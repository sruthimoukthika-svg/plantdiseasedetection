"""Inference pipeline: validate, embed, gate on species, classify, reject.

Loaded once at process start. The model is never trained here and no
request causes training.

Order matters and mirrors training exactly:

  1. cheap model-free checks (size, blur, vegetation)
  2. embed with the SAME transform training used (imported from
     training/dataset.py, so the two cannot drift)
  3. out-of-distribution scoring - Mahalanobis + energy, calibrated
     offline. A closed-set head always returns some class, so this is what
     actually decides whether the image is supported.
  4. species gate picks the crop, then that crop's condition head runs
  5. confidence floors applied last

If any gate rejects, no prediction is returned at all - the caller gets a
rejection, never a disease name with a low score attached.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

import dataset as ds  # noqa: E402
import labels as L  # noqa: E402
import models  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from backend.validation.image_checks import run_checks  # noqa: E402


@dataclass
class Prediction:
    accepted: bool
    reason: str = ""
    crop: str = ""
    crop_confidence: float = 0.0
    condition: str = ""
    condition_display: str = ""
    condition_kind: str = ""
    condition_confidence: float = 0.0
    alternatives: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class Predictor:
    def __init__(
        self,
        backbone: str,
        checkpoint_dir: Path,
        species_min_confidence: float = 0.0,
        condition_min_confidence: float = 0.0,
        enable_ood: bool = True,
    ) -> None:
        self.backbone = backbone
        ck_path = Path(checkpoint_dir) / f"{backbone}_heads.pt"
        if not ck_path.exists():
            raise FileNotFoundError(
                f"no trained weights at {ck_path}. Run training first; the "
                f"API deliberately refuses to start without real weights."
            )
        ck = torch.load(ck_path, weights_only=False)
        self.crop_index: dict[str, int] = ck["crop_index"]
        self.condition_index: dict[str, dict[str, int]] = ck["condition_index"]
        self.inv_crop = {v: k for k, v in self.crop_index.items()}
        self.inv_cond = {
            c: {v: k for k, v in self.condition_index[c].items()}
            for c in self.crop_index
        }

        self.features, dim = models.build_backbone(backbone, finetune=False)
        self.features.eval()

        # The 3-class gate cannot express "none of these": it must assign an
        # apple leaf to tomato, potato or grape. The 4-class gate adds an
        # explicit unsupported class, trained by outlier exposure, and is
        # preferred whenever the checkpoint carries one.
        self.unsupported_index: int | None = ck.get("unsupported_class_index")
        if "species_gate4" in ck["state"] and self.unsupported_index is not None:
            self.gate = models.LinearHead(dim, self.unsupported_index + 1)
            self.gate.load_state_dict(ck["state"]["species_gate4"])
            self.gate_has_unsupported = True
        else:
            self.gate = models.LinearHead(dim, len(self.crop_index))
            self.gate.load_state_dict(ck["state"]["species_head"])
            self.gate_has_unsupported = False
        self.gate.eval()
        self.heads: dict[str, models.LinearHead] = {}
        for crop in self.crop_index:
            h = models.LinearHead(dim, len(self.condition_index[crop]))
            h.load_state_dict(ck["state"][f"condition_heads.{crop}"])
            h.eval()
            self.heads[crop] = h

        self.transform = ds.eval_transform(ck.get("image_size", ds.IMAGE_SIZE))
        self.species_min = species_min_confidence
        self.condition_min = condition_min_confidence

        # out-of-distribution parameters, fitted and calibrated offline
        self.ood = None
        ood_path = Path(checkpoint_dir) / f"{backbone}_ood.npz"
        if enable_ood and ood_path.exists():
            z = np.load(ood_path)
            self.ood = {
                "means": z["means"],
                "precision": z["precision"],
                "thr_mahalanobis": float(z["thr_mahalanobis"][0]),
                "thr_energy": float(z["thr_energy"][0]),
            }

    # ---------------------------------------------------------------- scoring
    def _mahalanobis(self, z: np.ndarray) -> float:
        means, prec = self.ood["means"], self.ood["precision"]
        d = z.astype(np.float64) - means
        return float(np.einsum("ij,jk,ik->i", d, prec, d).min())

    def predict(self, im: Image.Image, top_k: int = 3) -> Prediction:
        chk = run_checks(im)
        if not chk.ok:
            return Prediction(False, chk.reason,
                              diagnostics={"image_checks": chk.detail})

        with torch.no_grad():
            x = self.transform(im.convert("RGB")).unsqueeze(0)
            z = self.features(x)
            gate_logits = self.gate(z)
            gate_p = torch.softmax(gate_logits, 1)[0].numpy()

        diag: dict = {"image_checks": chk.detail}

        # Layer 1 of rejection: the gate's own explicit unsupported class.
        if self.gate_has_unsupported:
            diag["unsupported_score"] = round(
                float(gate_p[self.unsupported_index]), 4)
            if int(gate_p.argmax()) == self.unsupported_index:
                return Prediction(False, "unsupported_image", diagnostics=diag)

        if self.ood is not None:
            zn = z.numpy()
            maha = self._mahalanobis(zn)
            energy = float(-torch.logsumexp(gate_logits, 1).item())
            diag["ood"] = {
                "mahalanobis": round(maha, 2),
                "mahalanobis_threshold": round(self.ood["thr_mahalanobis"], 2),
                "energy": round(energy, 4),
                "energy_threshold": round(self.ood["thr_energy"], 4),
            }
            if maha > self.ood["thr_mahalanobis"] or energy > self.ood["thr_energy"]:
                return Prediction(False, "unsupported_image", diagnostics=diag)

        # restrict the argmax to the supported crops; renormalise so the
        # reported species confidence is over supported crops only
        n_sup = len(self.crop_index)
        sup_p = gate_p[:n_sup]
        total = float(sup_p.sum()) or 1.0
        ci = int(sup_p.argmax())
        crop = self.inv_crop[ci]
        crop_conf = float(sup_p[ci] / total)
        diag["species_probs"] = {
            self.inv_crop[i]: round(float(p / total), 4)
            for i, p in enumerate(sup_p)
        }
        if crop_conf < self.species_min:
            return Prediction(False, "plant_species_uncertain", diagnostics=diag)

        with torch.no_grad():
            cond_p = torch.softmax(self.heads[crop](z), 1)[0].numpy()
        order = np.argsort(-cond_p)
        best = int(order[0])
        best_id = self.inv_cond[crop][best]
        cond_conf = float(cond_p[best])
        if cond_conf < self.condition_min:
            return Prediction(False, "condition_uncertain", diagnostics=diag)

        conds = L.CONDITIONS[crop]
        # Alternatives stay within the identified crop by construction - a
        # condition head only knows its own crop's classes.
        alts = []
        for j in order[1:top_k]:
            cid = self.inv_cond[crop][int(j)]
            p = float(cond_p[int(j)])
            if p < 0.01:
                continue   # not a meaningful alternative
            alts.append({
                "condition": cid,
                "display": conds[cid].display,
                "kind": conds[cid].kind,
                "confidence": round(p, 4),
            })

        c = conds[best_id]
        return Prediction(
            accepted=True,
            crop=crop,
            crop_confidence=round(crop_conf, 4),
            condition=best_id,
            condition_display=c.display,
            condition_kind=c.kind,
            condition_confidence=round(cond_conf, 4),
            alternatives=alts,
            diagnostics=diag,
        )

    # ------------------------------------------------------------- metadata
    def supported_plants(self) -> list[dict]:
        out = []
        for crop in sorted(self.crop_index):
            out.append({
                "crop": crop,
                "display": crop.capitalize(),
                "conditions": [
                    {
                        "condition": cid,
                        "display": L.CONDITIONS[crop][cid].display,
                        "kind": L.CONDITIONS[crop][cid].kind,
                        "pathogen": L.CONDITIONS[crop][cid].pathogen,
                    }
                    for cid in sorted(self.condition_index[crop])
                ],
            })
        return out
