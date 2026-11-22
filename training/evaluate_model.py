#!/usr/bin/env python3
"""Stage 3c: evaluate on the untouched test split.

Reports accuracy, macro/weighted precision, recall and F1, per-class F1,
confusion matrices and end-to-end inference time. The test split is read
here and nowhere else - it was never used for training or model selection.

Hierarchical evaluation reports two numbers that are easy to conflate:

  condition_given_true_crop : condition accuracy when the correct crop head
                              is used (isolates the condition heads)
  end_to_end                : the species gate picks the head, so a gate
                              error cascades. This is what a user experiences.

Usage:
    python3 training/evaluate_model.py --backbone mobilenet_v3_small
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import labels as L  # noqa: E402
import models  # noqa: E402


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None,
                      labels=list(range(len(names))), zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
        "f1_macro": round(float(f_m), 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "f1_weighted": round(float(f_w), 4),
        "per_class_f1": {n: round(float(v), 4) for n, v in zip(names, per_f1)},
        "support": {n: int((y_true == i).sum()) for i, n in enumerate(names)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(names)))).tolist(),
        "confusion_matrix_labels": names,
        "report": classification_report(
            y_true, y_pred, labels=list(range(len(names))),
            target_names=names, zero_division=0, digits=3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=list(models.BACKBONES))
    args = ap.parse_args()
    torch.set_num_threads(4)

    ckpt_path = config.CHECKPOINT_DIR / f"{args.backbone}_heads.pt"
    if not ckpt_path.exists():
        print(f"missing {ckpt_path} - run train_model.py first", file=sys.stderr)
        return 1
    ck = torch.load(ckpt_path, weights_only=False)
    dim = ck["dim"]
    crop_idx = ck["crop_index"]
    cond_idx = ck["condition_index"]
    inv_crop = {v: k for k, v in crop_idx.items()}

    feat_dir = config.DATA_DIR / "features" / args.backbone
    d = np.load(feat_dir / "test.npz")
    X, y_crop, y_cond = d["X"], d["crop"], d["cond"]
    print(f"test set: {X.shape[0]} images (never seen in training or selection)")

    gate = models.LinearHead(dim, len(crop_idx))
    gate.load_state_dict(ck["state"]["species_head"])
    gate.eval()
    heads = {}
    for crop in crop_idx:
        h = models.LinearHead(dim, len(cond_idx[crop]))
        h.load_state_dict(ck["state"][f"condition_heads.{crop}"])
        h.eval()
        heads[crop] = h

    Xt = torch.from_numpy(X)
    with torch.no_grad():
        crop_pred = gate(Xt).argmax(1).numpy()

    out: dict = {"backbone": args.backbone, "test_n": int(len(X)), "stages": {}}

    # --- species gate -----------------------------------------------------
    crop_names = [inv_crop[i] for i in range(len(crop_idx))]
    out["stages"]["species_gate"] = metrics_block(y_crop, crop_pred, crop_names)
    print(f"\nspecies gate: acc {out['stages']['species_gate']['accuracy']:.4f} "
          f"macroF1 {out['stages']['species_gate']['f1_macro']:.4f}")

    # --- condition heads, given the TRUE crop ----------------------------
    print("\ncondition heads (given true crop):")
    for crop, ci in sorted(crop_idx.items()):
        m = y_crop == ci
        if not m.any():
            continue
        names = [k for k, _ in sorted(cond_idx[crop].items(), key=lambda kv: kv[1])]
        with torch.no_grad():
            pred = heads[crop](torch.from_numpy(X[m])).argmax(1).numpy()
        blk = metrics_block(y_cond[m], pred, names)
        out["stages"][f"condition_given_true_crop_{crop}"] = blk
        print(f"  {crop:7s} n={blk['n']:5d} acc {blk['accuracy']:.4f} "
              f"macroF1 {blk['f1_macro']:.4f}")
        worst = sorted(blk["per_class_f1"].items(), key=lambda kv: kv[1])[:3]
        print(f"          weakest: {', '.join(f'{k}={v:.2f}' for k, v in worst)}")

    # --- end to end: gate chooses the head -------------------------------
    # Flatten to a single global label space so cascade errors are counted.
    global_names: list[str] = []
    gindex: dict[tuple[str, str], int] = {}
    for crop, ci in sorted(crop_idx.items()):
        for cid, k in sorted(cond_idx[crop].items(), key=lambda kv: kv[1]):
            gindex[(crop, cid)] = len(global_names)
            global_names.append(f"{crop}/{cid}")
    inv_cond = {crop: {v: k for k, v in cond_idx[crop].items()} for crop in crop_idx}

    y_true_g = np.array([gindex[(inv_crop[c], inv_cond[inv_crop[c]][d_])]
                         for c, d_ in zip(y_crop, y_cond)])
    y_pred_g = np.empty_like(y_true_g)
    with torch.no_grad():
        for crop, ci in crop_idx.items():
            m = crop_pred == ci
            if not m.any():
                continue
            pr = heads[crop](torch.from_numpy(X[m])).argmax(1).numpy()
            y_pred_g[m] = [gindex[(crop, inv_cond[crop][p])] for p in pr]
    blk = metrics_block(y_true_g, y_pred_g, global_names)
    out["stages"]["end_to_end"] = blk
    print(f"\nend-to-end (gate + condition head over {len(global_names)} classes):")
    print(f"  acc {blk['accuracy']:.4f}  macroF1 {blk['f1_macro']:.4f}  "
          f"weightedF1 {blk['f1_weighted']:.4f}")

    # --- inference timing, full pipeline on real images ------------------
    import dataset as ds  # noqa: E402
    from PIL import Image
    prepared = config.DATA_DIR / "prepared"
    rows = [r for r in ds.load_manifest(prepared / "manifest.csv")
            if r.split == "test"][:200]
    feat, _ = models.build_backbone(args.backbone, finetune=False)
    feat.eval()
    tf = ds.eval_transform()
    times = []
    with torch.no_grad():
        for r in rows:
            t0 = time.perf_counter()
            im = Image.open(prepared / "images" / r.path).convert("RGB")
            z = feat(tf(im).unsqueeze(0))
            c = gate(z).argmax(1).item()
            heads[inv_crop[c]](z)
            times.append((time.perf_counter() - t0) * 1000)
    times_a = np.array(times)
    out["inference_ms_per_image"] = {
        "n": len(times), "mean": round(float(times_a.mean()), 2),
        "p50": round(float(np.percentile(times_a, 50)), 2),
        "p95": round(float(np.percentile(times_a, 95)), 2),
        "note": "single image, batch size 1, 4 CPU threads, no GPU; "
                "includes JPEG decode and preprocessing",
    }
    print(f"\ninference: mean {times_a.mean():.1f} ms  "
          f"p95 {np.percentile(times_a, 95):.1f} ms (batch 1, CPU)")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.EVAL_DIR / f"{args.backbone}_test_metrics.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
