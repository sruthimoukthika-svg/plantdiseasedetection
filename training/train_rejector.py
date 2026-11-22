#!/usr/bin/env python3
"""Stage 4d: train a 4-class species gate with outlier exposure, then
jointly calibrate the rejection thresholds.

Two defects in the first attempt are fixed here.

1. Structural. A 3-class gate has to place an apple leaf in tomato, potato
   or grape. Distance scoring alone accepted 22.7% of unsupported
   PlantVillage species. The gate now has a fourth class, "unsupported",
   trained on outlier species.

2. Calibration. Thresholds were set at 95% each, independently, then
   OR-rejected. Two such rules compound, giving 0.905 combined accept
   rather than 0.95. Both thresholds are now searched jointly for the
   percentile at which the COMBINED rule hits the target true-accept rate.

Honesty of evaluation: the exposure species and the held-out species are
disjoint, so the reported false-accept rate is for species never seen as
an outlier during training.

Usage: python3 training/train_rejector.py --backbone efficientnet_b0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_rejection as CR  # noqa: E402
import config  # noqa: E402
import dataset as ds  # noqa: E402
import models  # noqa: E402

UNSUPPORTED = "unsupported"


def embed_dir(root: Path, backbone: str, cache: Path, batch: int = 64) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["X"]
    paths = sorted(p for p in root.rglob("*.jpg"))
    if not paths:
        return np.zeros((0, 1), dtype=np.float32)
    feat, _ = models.build_backbone(backbone, finetune=False)
    feat.eval()
    tf = ds.eval_transform()
    out = []
    t0 = time.time()
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
    X = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X)
    print(f"    embedded {len(X)} from {root.name} in {time.time() - t0:.0f}s")
    return X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0",
                    choices=list(models.BACKBONES))
    ap.add_argument("--target-tpr", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    feat_dir = config.DATA_DIR / "features" / args.backbone
    ck_path = config.CHECKPOINT_DIR / f"{args.backbone}_heads.pt"
    ck = torch.load(ck_path, weights_only=False)
    crop_idx = dict(ck["crop_index"])
    cond_idx = ck["condition_index"]
    meta = json.loads((feat_dir / "meta.json").read_text())
    dim = ck["dim"]

    tr = np.load(feat_dir / "train.npz")
    n_clean = len(tr["X"]) // meta["aug_passes"]
    Xs, cs, dsup = tr["X"][:n_clean], tr["crop"][:n_clean], tr["cond"][:n_clean]
    va = np.load(feat_dir / "val.npz")
    te = np.load(feat_dir / "test.npz")

    ood_root = config.DATA_DIR / "ood"
    cache_root = config.DATA_DIR / "features" / args.backbone / "ood"
    print("embedding OOD sets ...")
    sets = {}
    for name in ("unsupported_lab_exposure", "unsupported_lab_heldout",
                 "unsupported_field", "non_plant_objects"):
        d = ood_root / name
        if d.exists():
            sets[name] = embed_dir(d, args.backbone, cache_root / f"{name}.npz")
            print(f"  {name:28s} {sets[name].shape}")

    Xexp = sets.get("unsupported_lab_exposure")
    if Xexp is None or len(Xexp) == 0:
        print("no exposure set; run build_exposure_sets.py first", file=sys.stderr)
        return 1

    # ---------------- 4-class gate: grape / potato / tomato / unsupported
    unsup_id = len(crop_idx)
    Xg = np.concatenate([Xs, Xexp])
    yg = np.concatenate([cs, np.full(len(Xexp), unsup_id, dtype=cs.dtype)])
    # hold back a slice of exposure for the gate's own validation
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(len(Xexp))
    n_hold = max(1, len(Xexp) // 10)
    exp_val, exp_tr = perm[:n_hold], perm[n_hold:]
    Xg_tr = np.concatenate([Xs, Xexp[exp_tr]])
    yg_tr = np.concatenate([cs, np.full(len(exp_tr), unsup_id, dtype=cs.dtype)])
    Xg_va = np.concatenate([va["X"], Xexp[exp_val]])
    yg_va = np.concatenate([va["crop"],
                            np.full(len(exp_val), unsup_id, dtype=cs.dtype)])
    print(f"\ntraining 4-class gate: train {Xg_tr.shape}  val {Xg_va.shape}")

    gate4 = models.LinearHead(dim, unsup_id + 1)
    opt = torch.optim.AdamW(gate4.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(
        weight=CR.torch.tensor(
            (np.bincount(yg_tr).sum() / (len(np.bincount(yg_tr))
                                         * np.maximum(np.bincount(yg_tr), 1))
             ).astype(np.float32)))
    Xt, yt = torch.from_numpy(Xg_tr), torch.from_numpy(yg_tr).long()
    Xv = torch.from_numpy(Xg_va)
    best, best_state = -1.0, None
    g = torch.Generator().manual_seed(config.SEED)
    for ep in range(args.epochs):
        gate4.train()
        pm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 512):
            idx = pm[i:i + 512]
            opt.zero_grad()
            lossf(gate4(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        gate4.eval()
        with torch.no_grad():
            pred = gate4(Xv).argmax(1).numpy()
        mf1 = f1_score(yg_va, pred, average="macro", zero_division=0)
        if mf1 > best:
            best, best_state = mf1, {k: v.clone() for k, v in gate4.state_dict().items()}
    gate4.load_state_dict(best_state)
    print(f"  gate val macroF1 {best:.4f}")

    # ---------------- Mahalanobis on the 22 supported classes
    gmap = {}
    for crop, ci in sorted(crop_idx.items()):
        for cid, k in sorted(cond_idx[crop].items(), key=lambda kv: kv[1]):
            gmap[(ci, k)] = len(gmap)
    ysup = np.array([gmap[(c, d_)] for c, d_ in zip(cs, dsup)])
    means, prec = CR.fit_gaussians(Xs, ysup, len(gmap))

    def signals(X: np.ndarray):
        with torch.no_grad():
            lg = gate4(torch.from_numpy(X))
            pred = lg.argmax(1).numpy()
            en = (-torch.logsumexp(lg, 1)).numpy()
        return pred, CR.mahalanobis_min(X, means, prec), en

    # ---------------- joint calibration at the target true-accept rate
    pv, mv, ev = signals(va["X"])
    gate_ok_val = pv != unsup_id

    def accept_rate(p: float) -> tuple[float, float, float]:
        tm = float(np.percentile(mv, p))
        tev = float(np.percentile(ev, p))
        return float((gate_ok_val & (mv <= tm) & (ev <= tev)).mean()), tm, tev

    lo, hi = 50.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        r, _, _ = accept_rate(mid)
        if r < args.target_tpr:
            lo = mid
        else:
            hi = mid
    rate, thr_m, thr_e = accept_rate(hi)
    print(f"\njointly calibrated at percentile {hi:.2f}")
    print(f"  mahalanobis <= {thr_m:.2f}   energy <= {thr_e:.4f}")
    print(f"  val accept rate (combined rule): {rate:.4f}  "
          f"(target {args.target_tpr})")

    def far(X: np.ndarray) -> dict:
        p, m, e = signals(X)
        acc = (p != unsup_id) & (m <= thr_m) & (e <= thr_e)
        return {"n": int(len(X)),
                "false_accept_rate": round(float(acc.mean()), 4),
                "rejected_by_gate_alone": round(float((p == unsup_id).mean()), 4),
                "rejected_by_mahalanobis_alone": round(float((m > thr_m).mean()), 4),
                "rejected_by_energy_alone": round(float((e > thr_e).mean()), 4)}

    pt, mt, et = signals(te["X"])
    res = {
        "backbone": args.backbone,
        "target_tpr": args.target_tpr,
        "calibration_percentile": round(hi, 3),
        "thresholds": {"mahalanobis": thr_m, "energy": thr_e},
        "gate_val_macro_f1": round(float(best), 4),
        "exposure_species": json.loads(
            (config.REPORT_DIR / "exposure_summary.json").read_text()
        )["unsupported_lab_exposure"]["species"],
        "heldout_species": json.loads(
            (config.REPORT_DIR / "exposure_summary.json").read_text()
        )["unsupported_lab_heldout"]["species"],
        "in_distribution": {
            "val_accept_rate": round(rate, 4),
            "test_accept_rate": round(float(
                ((pt != unsup_id) & (mt <= thr_m) & (et <= thr_e)).mean()), 4),
        },
        "ood_sets": {},
    }
    print("\nfalse-accept rate per OOD set (lower is better):")
    for name in ("unsupported_lab_heldout", "unsupported_field",
                 "non_plant_objects", "unsupported_lab_exposure"):
        if name not in sets or len(sets[name]) == 0:
            continue
        res["ood_sets"][name] = far(sets[name])
        v = res["ood_sets"][name]
        tag = " (seen in training)" if name.endswith("exposure") else ""
        print(f"  {name:28s} n={v['n']:5d}  FAR {v['false_accept_rate']:.4f}"
              f"   gate alone rejects {v['rejected_by_gate_alone']:.4f}{tag}")

    ck["state"]["species_gate4"] = gate4.state_dict()
    ck["unsupported_class_index"] = unsup_id
    torch.save(ck, ck_path)
    np.savez_compressed(
        config.CHECKPOINT_DIR / f"{args.backbone}_ood.npz",
        means=means, precision=prec,
        thr_mahalanobis=np.array([thr_m]), thr_energy=np.array([thr_e]),
    )
    dest = config.EVAL_DIR / f"{args.backbone}_rejection.json"
    dest.write_text(json.dumps(res, indent=2))
    (config.REPORT_DIR / f"{args.backbone}_rejection.json").write_text(
        json.dumps(res, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
