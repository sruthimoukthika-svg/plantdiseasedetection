#!/usr/bin/env python3
"""Stage 3b: train the species gate and per-crop condition heads.

Trains on cached frozen-backbone features, so each head is a small linear
problem that fits in seconds rather than hours of CPU time.

Class imbalance is handled with inverse-frequency class weights in the loss.
Tomato is 75:1 (yellow_leaf_curl_virus 3,749 vs potassium_deficiency 50), so
unweighted training would simply learn the majority classes.

The best checkpoint is selected on validation MACRO F1, not accuracy:
with imbalance this severe, accuracy is dominated by the large classes and
would hide total failure on the small ones.

The test split is never read here. It is used only by evaluate_model.py.

Usage:
    python3 training/train_model.py --backbone mobilenet_v3_small
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
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import labels as L  # noqa: E402
import models  # noqa: E402


def load_split(feat_dir: Path, split: str):
    d = np.load(feat_dir / f"{split}.npz")
    return d["X"], d["crop"], d["cond"]


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 so the loss scale
    stays comparable across heads."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_head(
    name: str,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    n_classes: int,
    dim: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    head = models.LinearHead(dim, n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes))

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr).long()
    Xva_t = torch.from_numpy(Xva)

    best = {"macro_f1": -1.0, "epoch": -1}
    best_state = None
    history = []
    bs = 512
    n = len(Xtr_t)
    g = torch.Generator().manual_seed(seed)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        train_loss = tot / n

        head.eval()
        with torch.no_grad():
            logits = head(Xva_t)
            val_loss = lossf(logits, torch.from_numpy(yva).long()).item()
            pred = logits.argmax(1).numpy()
        mf1 = f1_score(yva, pred, average="macro", zero_division=0)
        acc = float((pred == yva).mean())
        history.append({"epoch": ep, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "val_macro_f1": round(mf1, 4), "val_acc": round(acc, 4)})

        if mf1 > best["macro_f1"]:
            best = {"macro_f1": mf1, "acc": acc, "epoch": ep,
                    "val_loss": val_loss, "train_loss": train_loss}
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        elif ep - best["epoch"] >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    print(f"  {name:22s} best epoch {best['epoch']:3d}  "
          f"val macroF1 {best['macro_f1']:.4f}  val acc {best['acc']:.4f}  "
          f"({len(history)} epochs run)")
    return head, {"best": best, "history": history}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=list(models.BACKBONES))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    feat_dir = config.DATA_DIR / "features" / args.backbone
    if not (feat_dir / "train.npz").exists():
        print(f"missing {feat_dir}/train.npz - run extract_features.py first",
              file=sys.stderr)
        return 1
    meta = json.loads((feat_dir / "meta.json").read_text())
    dim = meta["dim"]

    Xtr, ctr, dtr = load_split(feat_dir, "train")
    Xva, cva, dva = load_split(feat_dir, "val")
    print(f"backbone {args.backbone} dim {dim}")
    print(f"train {Xtr.shape}  val {Xva.shape}  (aug passes {meta['aug_passes']})")

    crop_idx = L.crop_index()
    results: dict = {"backbone": args.backbone, "dim": dim,
                     "aug_passes": meta["aug_passes"], "seed": args.seed,
                     "heads": {}}
    state: dict = {}
    t0 = time.time()

    # --- species gate -----------------------------------------------------
    print("\ntraining species gate ...")
    gate, r = train_head("species_gate", Xtr, ctr, Xva, cva,
                         len(crop_idx), dim, args.epochs, args.lr,
                         args.patience, args.seed)
    results["heads"]["species"] = r
    state["species_head"] = gate.state_dict()

    # --- per-crop condition heads ----------------------------------------
    print("\ntraining condition heads ...")
    for crop, ci in sorted(crop_idx.items()):
        n_cls = len(L.condition_index(crop))
        mtr, mva = ctr == ci, cva == ci
        head, r = train_head(f"condition/{crop}", Xtr[mtr], dtr[mtr],
                             Xva[mva], dva[mva], n_cls, dim,
                             args.epochs, args.lr, args.patience, args.seed)
        results["heads"][f"condition_{crop}"] = r
        state[f"condition_heads.{crop}"] = head.state_dict()

    elapsed = time.time() - t0
    results["train_seconds"] = round(elapsed, 1)
    print(f"\ntrained all heads in {elapsed:.0f}s")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = config.CHECKPOINT_DIR / f"{args.backbone}_heads.pt"
    torch.save({
        "backbone": args.backbone,
        "dim": dim,
        "crop_index": crop_idx,
        "condition_index": {c: L.condition_index(c) for c in L.CONDITIONS},
        "state": state,
        "image_size": meta["image_size"],
        "seed": args.seed,
    }, ckpt)
    print(f"checkpoint: {ckpt}")

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    hist = config.EVAL_DIR / f"{args.backbone}_training.json"
    hist.write_text(json.dumps(results, indent=2))
    print(f"history   : {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
