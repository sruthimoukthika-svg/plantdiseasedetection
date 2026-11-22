"""FastAPI backend for the plant disease scanner.

Weights are loaded once at startup. No request triggers training. Uploads
are processed in memory and discarded unless PDAI_PERSIST_UPLOADS is set.

The service refuses to start without real trained weights, rather than
falling back to random or placeholder predictions.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "training"))

from backend.services.condition_info import CONFIRMATION_NOTE, get as info_get  # noqa: E402
from backend.services.predictor import Predictor  # noqa: E402

# ----------------------------------------------------------------- settings
BACKBONE = os.getenv("PDAI_BACKBONE", "efficientnet_b0")
MODEL_DIR = Path(os.getenv("PDAI_MODEL_DIR", REPO_ROOT / "checkpoints"))
MAX_UPLOAD_BYTES = int(os.getenv("PDAI_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_ORIGINS = [
    o.strip() for o in
    os.getenv("PDAI_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
SPECIES_MIN = float(os.getenv("PDAI_SPECIES_MIN_CONFIDENCE", 0.0))
CONDITION_MIN = float(os.getenv("PDAI_CONDITION_MIN_CONFIDENCE", 0.0))
PERSIST_UPLOADS = os.getenv("PDAI_PERSIST_UPLOADS", "false").lower() == "true"
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

DISCLAIMER = (
    "This is an AI-generated prediction and not a confirmed diagnosis."
)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    STATE["predictor"] = Predictor(
        backbone=BACKBONE,
        checkpoint_dir=MODEL_DIR,
        species_min_confidence=SPECIES_MIN,
        condition_min_confidence=CONDITION_MIN,
    )
    STATE["load_seconds"] = round(time.time() - t0, 2)
    # metrics are read from the evaluation artefacts, never hard-coded
    STATE["metrics"] = {}
    m = REPO_ROOT / "evaluation_results" / f"{BACKBONE}_test_metrics.json"
    if m.exists():
        STATE["metrics"] = json.loads(m.read_text())
    r = REPO_ROOT / "evaluation_results" / f"{BACKBONE}_rejection.json"
    STATE["rejection"] = json.loads(r.read_text()) if r.exists() else {}
    yield
    STATE.clear()


app = FastAPI(title="Plant Disease Detection AI", version="0.1.0",
              lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _predictor() -> Predictor:
    p = STATE.get("predictor")
    if p is None:
        raise HTTPException(503, "model not loaded")
    return p


REJECTION_MESSAGES = {
    "image_too_large": "That image is too large to process. Please upload a smaller image.",
    "image_too_small": "That image is too small to analyse. Please upload a larger, clearer photo.",
    "image_too_blurry": "That image is too blurry or low-contrast to analyse. Please upload a sharper photo.",
    "no_plant_tissue_detected": "No plant foliage could be detected in that image. Please upload a clear photo of a leaf.",
    "unsupported_image": "This image could not be confidently identified as a supported plant leaf. Please upload a clear image of a tomato, potato, or grape leaf.",
    "plant_species_uncertain": "The plant species could not be identified confidently enough. Please upload a clearer photo of a single tomato, potato, or grape leaf.",
    "condition_uncertain": "A condition could not be predicted confidently enough for this image. Please upload a clearer, closer photo of the affected leaf.",
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backbone": BACKBONE,
            "load_seconds": STATE.get("load_seconds")}


@app.get("/api/supported-plants")
def supported_plants() -> dict:
    """Derived from the trained class mapping, never a hard-coded list."""
    return {"plants": _predictor().supported_plants()}


@app.get("/api/diseases")
def diseases() -> dict:
    """Verified conditions per crop, with reference information."""
    out = []
    for entry in _predictor().supported_plants():
        crop = entry["crop"]
        conditions = []
        for c in entry["conditions"]:
            conditions.append({**c, "info": info_get(crop, c["condition"])})
        out.append({"crop": crop, "display": entry["display"],
                    "conditions": conditions})
    return {"crops": out, "confirmation_note": CONFIRMATION_NOTE}


@app.get("/api/model-info")
def model_info() -> dict:
    """Measured metrics only - read from the evaluation artefacts."""
    m = STATE.get("metrics") or {}
    stages = m.get("stages", {})
    e2e = stages.get("end_to_end", {})
    return {
        "backbone": BACKBONE,
        "architecture": "frozen ImageNet backbone; species gate plus "
                        "per-crop condition heads",
        "image_size": 224,
        "test_set_size": m.get("test_n"),
        "measured": {
            "end_to_end_accuracy": e2e.get("accuracy"),
            "end_to_end_macro_f1": e2e.get("f1_macro"),
            "species_gate_accuracy": stages.get("species_gate", {}).get("accuracy"),
            "per_source_dataset": m.get("per_source_dataset"),
            "inference_ms_per_image": m.get("inference_ms_per_image"),
        },
        "rejection": {
            "method": "Mahalanobis distance to class-conditional Gaussians "
                      "plus free-energy over species logits; calibrated on "
                      "validation data. Max-softmax is deliberately not used.",
            "thresholds": STATE.get("rejection", {}).get("thresholds"),
            "target_true_accept_rate": STATE.get("rejection", {}).get("target_tpr"),
            "false_accept_rate_by_set": {
                k: v.get("false_accept_rate")
                for k, v in (STATE.get("rejection", {}).get("ood_sets") or {}).items()
            },
        },
        "low_support_classes": m.get("low_support_classes"),
        "caveat": "Accuracy measured on lab-style imagery does not transfer "
                  "to field photographs. See per_source_dataset: the model is "
                  "markedly weaker on real-world photos than on the "
                  "standardised PlantVillage images.",
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={
            "success": False, "rejected": True, "reason": "file_too_large",
            "message": f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        })
    try:
        im = Image.open(io.BytesIO(raw))
        fmt = (im.format or "").upper()
        im.load()
    except (UnidentifiedImageError, OSError):
        return JSONResponse(status_code=415, content={
            "success": False, "rejected": True, "reason": "unsupported_file_type",
            "message": "Please upload a JPG, PNG or WEBP image.",
        })
    if fmt not in ALLOWED_FORMATS:
        return JSONResponse(status_code=415, content={
            "success": False, "rejected": True, "reason": "unsupported_file_type",
            "message": f"{fmt or 'That file type'} is not supported. "
                       f"Please upload a JPG, PNG or WEBP image.",
        })

    t0 = time.perf_counter()
    try:
        pred = _predictor().predict(im)
    except Exception:  # noqa: BLE001
        raise HTTPException(500, "inference failed")
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    if not PERSIST_UPLOADS:
        del raw   # not written to disk at any point

    if not pred.accepted:
        return JSONResponse(status_code=200, content={
            "success": False,
            "rejected": True,
            "reason": pred.reason,
            "message": REJECTION_MESSAGES.get(
                pred.reason,
                "This image could not be analysed. Please upload a clear "
                "image of a tomato, potato, or grape leaf."),
            "diagnostics": pred.diagnostics,
            "inference_ms": elapsed,
        })

    return JSONResponse(status_code=200, content={
        "success": True,
        "plant": {"name": pred.crop.capitalize(),
                  "id": pred.crop,
                  "confidence": pred.crop_confidence},
        "prediction": {"name": pred.condition_display,
                       "id": pred.condition,
                       "kind": pred.condition_kind,
                       "confidence": pred.condition_confidence},
        "alternatives": [
            {"name": a["display"], "id": a["condition"],
             "kind": a["kind"], "confidence": a["confidence"]}
            for a in pred.alternatives
        ],
        "info": info_get(pred.crop, pred.condition),
        "confidence_note": "Confidence is the model's prediction score for "
                           "this image. It is not the probability that the "
                           "plant truly has this condition.",
        "diagnostics": pred.diagnostics,
        "inference_ms": elapsed,
        "disclaimer": DISCLAIMER,
    })
