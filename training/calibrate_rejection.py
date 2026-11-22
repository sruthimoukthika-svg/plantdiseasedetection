#!/usr/bin/env python3
"""Stage 4b: fit and calibrate out-of-distribution rejection.

The brief is explicit that a high softmax score is not evidence an image
belongs to a supported category - a closed-set classifier assigns SOME
class to every input, however unrelated. So rejection uses two signals
computed in the frozen embedding space, neither of which is max-softmax:

  Mahalanobis : distance to the nearest class-conditional Gaussian fitted
                on training embeddings, with a shared shrunk covariance.
                Measures "does this look like any class I was trained on".
  Energy      : -logsumexp over the species-gate logits. Low for confident
                in-distribution inputs, high for inputs the gate cannot
                place anywhere.

Thresholds are calibrated on the VALIDATION split at a chosen true-accept
rate (default 95%), then the false-accept rate is measured on each OOD set
separately. Test data is never used for calibration.

Gaussians are fitted only on the non-augmented (pass 0) training
embeddings; augmented views would inflate the covariance and loosen the
distance scale.

Usage:
    python3 training/calibrate_rejection.py --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataset as ds  # noqa: E402
import labels as L  # noqa: E402
import models  # noqa: E402

TARGET_TPR = 0.95
SHRINKAGE = 0.10


def fit_gaussians(X: np.ndarray, y: np.ndarray, n_classes: int):
    """Class means plus a single shared, shrunk covariance."""
    d = X.shape[1]
    means = np.zeros((n_classes, d), dtype=np.float64)
    acc = np.zeros((d, d), dtype=np.float64)
    total = 0
    for c in range(n_classes):
        m = y == c
        if m.sum() < 2:
            means[c] = X[m].mean(0) if m.any() else 0.0
            continue
        Xc = X[m].astype(np.float64)
        means[c] = Xc.mean(0)
        Z = Xc - means[c]
        acc += Z.T @ Z
        total += len(Xc)
    cov = acc / max(total - n_classes, 1)
    # shrink toward a scaled identity: d is large and some classes are small
    cov = (1 - SHRINKAGE) * cov + SHRINKAGE * np.eye(d) * np.trace(cov) / d
    prec = np.linalg.pinv(cov)
    return means, prec


def mahalanobis_min(X: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Minimum squared Mahalanobis distance to any class centroid."""
    out = np.empty((len(X), len(means)), dtype=np.float64)
    Xd = X.astype(np.float64)
    for c, mu in enumerate(means):
        Z = Xd - mu
        out[:, c] = np.einsum("ij,jk,ik->i", Z, prec, Z)
    return out.min(1)


def energy(logits: np.ndarray) -> np.ndarray:
    return -torch.logsumexp(torch.from_numpy(logits), dim=1).numpy()


def embed_dir(root: Path, backbone: str, batch: int = 64) -> np.ndarray:
    """Embed every image under root with the production eval transform."""
    paths = sorted(p for p in root.rglob("*.jpg"))
    if not paths:
        return np.zeros((0, 1), dtype=np.float32)
    feat, _ = models.build_backbone(backbone, finetune=False)
    feat.eval()
    tf = ds.eval_transform()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = []
            for p in paths[i:i + batch]:
                try:
                    xs.append(tf(Image.open(p).convert("RGB")))
                except Exception:  # noqa: BLE001
                    continue
            if xs:
                out.append(feat(torch.stack(xs)).numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0",
                    choices=list(models.BACKBONES))
    ap.add_argument("--target-tpr", type=float, default=TARGET_TPR)
    args = ap.parse_args()
    torch.set_num_threads(4)

    feat_dir = config.DATA_DIR / "features" / args.backbone
    ck_path = config.CHECKPOINT_DIR / f"{args.backbone}_heads.pt"
    for p in (feat_dir / "train.npz", ck_path):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
    meta = json.loads((feat_dir / "meta.json").read_text())
    ck = torch.load(ck_path, weights_only=False)
    crop_idx = ck["crop_index"]
    cond_idx = ck["condition_index"]

    tr = np.load(feat_dir / "train.npz")
    Xtr, ctr, dtr = tr["X"], tr["crop"], tr["cond"]
    # pass 0 is the deterministic eval-transform view of every train image
    n_clean = len(Xtr) // meta["aug_passes"]
    Xtr, ctr, dtr = Xtr[:n_clean], ctr[:n_clean], dtr[:n_clean]
    print(f"fitting on {len(Xtr)} non-augmented train embeddings "
          f"(dim {Xtr.shape[1]})")

    # global class id = (crop, condition) so Gaussians are per fine class
    inv_crop = {v: k for k, v in crop_idx.items()}
    gmap: dict[tuple[int, int], int] = {}
    for crop, ci in sorted(crop_idx.items()):
        for cid, k in sorted(cond_idx[crop].items(), key=lambda kv: kv[1]):
            gmap[(ci, k)] = len(gmap)
    ytr = np.array([gmap[(c, d_)] for c, d_ in zip(ctr, dtr)])

    t0 = time.time()
    means, prec = fit_gaussians(Xtr, ytr, len(gmap))
    print(f"fitted {len(gmap)} class Gaussians in {time.time() - t0:.0f}s "
          f"(shrinkage {SHRINKAGE})")

    gate = models.LinearHead(ck["dim"], len(crop_idx))
    gate.load_state_dict(ck["state"]["species_head"])
    gate.eval()

    def score(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            lg = gate(torch.from_numpy(X)).numpy()
        return mahalanobis_min(X, means, prec), energy(lg)

    va = np.load(feat_dir / "val.npz")
    m_val, e_val = score(va["X"])
    # calibrate each threshold independently at the target true-accept rate
    q = args.target_tpr * 100
    thr_m = float(np.percentile(m_val, q))
    thr_e = float(np.percentile(e_val, q))
    print(f"\ncalibrated on {len(m_val)} val images at TPR {args.target_tpr:.2f}")
    print(f"  mahalanobis threshold {thr_m:.2f}")
    print(f"  energy threshold      {thr_e:.4f}")

    accept_val = (m_val <= thr_m) & (e_val <= thr_e)
    print(f"  combined val accept rate (OR-reject): {accept_val.mean():.4f}")

    results = {
        "backbone": args.backbone,
        "target_tpr": args.target_tpr,
        "shrinkage": SHRINKAGE,
        "thresholds": {"mahalanobis": thr_m, "energy": thr_e},
        "val_accept_rate_combined": round(float(accept_val.mean()), 4),
        "ood_sets": {},
    }

    # in-distribution test split, for reference
    te = np.load(feat_dir / "test.npz")
    m_te, e_te = score(te["X"])
    acc_te = (m_te <= thr_m) & (e_te <= thr_e)
    results["test_accept_rate_in_distribution"] = round(float(acc_te.mean()), 4)
    print(f"  in-distribution TEST accept rate:     {acc_te.mean():.4f}")

    ood_root = config.DATA_DIR / "ood"
    print("\nfalse-accept rate per OOD set (lower is better):")
    for sub in sorted(p for p in ood_root.iterdir() if p.is_dir()) if ood_root.exists() else []:
        Xo = embed_dir(sub, args.backbone)
        if len(Xo) == 0 or Xo.shape[1] != Xtr.shape[1]:
            print(f"  {sub.name:22s} (no images)")
            continue
        m_o, e_o = score(Xo)
        fa_both = float(((m_o <= thr_m) & (e_o <= thr_e)).mean())
        results["ood_sets"][sub.name] = {
            "n": int(len(Xo)),
            "false_accept_rate": round(fa_both, 4),
            "false_accept_mahalanobis_only": round(float((m_o <= thr_m).mean()), 4),
            "false_accept_energy_only": round(float((e_o <= thr_e).mean()), 4),
            "mahalanobis_p50": round(float(np.percentile(m_o, 50)), 2),
        }
        print(f"  {sub.name:22s} n={len(Xo):5d}  FAR {fa_both:.4f}   "
              f"(maha-only {(m_o <= thr_m).mean():.4f}, "
              f"energy-only {(e_o <= thr_e).mean():.4f})")

    # persist what the backend needs at inference
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.CHECKPOINT_DIR / f"{args.backbone}_ood.npz",
        means=means, precision=prec,
        thr_mahalanobis=np.array([thr_m]), thr_energy=np.array([thr_e]),
    )
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.EVAL_DIR / f"{args.backbone}_rejection.json"
    dest.write_text(json.dumps(results, indent=2))
    (config.REPORT_DIR / f"{args.backbone}_rejection.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nwrote {dest}")
    print(f"wrote {config.CHECKPOINT_DIR / f'{args.backbone}_ood.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
