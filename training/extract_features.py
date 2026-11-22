#!/usr/bin/env python3
"""Stage 3a: cache frozen-backbone embeddings.

With no GPU, this is the difference between minutes and hours. The backbone
is frozen, so its output for a given image never changes - computing it once
and caching turns head training into seconds of linear algebra.

Train images are embedded several times under augmentation (the frozen path
cannot otherwise benefit from augmentation, since features would be fixed).
Val/test are embedded once, deterministically, via the same eval_transform
the backend uses at inference.

The cached embeddings are reused by Stage 4 for out-of-distribution scoring.

Usage:
    python3 training/extract_features.py --backbone mobilenet_v3_small
    python3 training/extract_features.py --backbone efficientnet_b0 --aug 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataset as ds  # noqa: E402
import labels as L  # noqa: E402
import models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=list(models.BACKBONES))
    ap.add_argument("--aug", type=int, default=2,
                    help="augmented passes over the train split (1 = none)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.set_num_threads(4)

    prepared = config.DATA_DIR / "prepared"
    rows = ds.load_manifest(prepared / "manifest.csv")
    img_root = prepared / "images"
    crop_idx = L.crop_index()
    cond_idx = {c: L.condition_index(c) for c in L.CONDITIONS}

    feat, dim = models.build_backbone(args.backbone, finetune=False)
    feat.eval()
    print(f"backbone {args.backbone}  embedding dim {dim}")

    out_dir = config.DATA_DIR / "features" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r.split == split]
        n_passes = args.aug if split == "train" else 1
        embeds, crops, conds, sources = [], [], [], []
        for p in range(n_passes):
            # pass 0 is always the deterministic eval transform, so the cache
            # always contains a clean, non-augmented view of every image
            tf = ds.eval_transform() if (p == 0 or split != "train") else ds.train_transform()
            dl = DataLoader(
                ds.LeafDataset(split_rows, img_root, crop_idx, cond_idx, tf),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers,
            )
            t0 = time.time()
            with torch.no_grad():
                for xb, cb, db in dl:
                    embeds.append(feat(xb).numpy().astype(np.float32))
                    crops.append(cb.numpy())
                    conds.append(db.numpy())
            sources.extend(r.source_dataset for r in split_rows)
            kind = "eval-tf" if (p == 0 or split != "train") else "train-tf"
            print(f"  {split} pass {p + 1}/{n_passes} ({kind}) "
                  f"{len(split_rows)} imgs in {time.time() - t0:.0f}s")

        X = np.concatenate(embeds)
        np.savez_compressed(
            out_dir / f"{split}.npz",
            X=X,
            crop=np.concatenate(crops),
            cond=np.concatenate(conds),
        )
        (out_dir / f"{split}_sources.json").write_text(json.dumps(sources))
        print(f"  -> {split}.npz  X{X.shape}")

    (out_dir / "meta.json").write_text(json.dumps({
        "backbone": args.backbone, "dim": dim, "aug_passes": args.aug,
        "seed": config.SEED, "image_size": ds.IMAGE_SIZE,
        "crop_index": crop_idx,
        "condition_index": cond_idx,
    }, indent=2))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
