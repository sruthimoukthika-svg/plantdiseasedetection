#!/usr/bin/env python3
"""Stage 2: build the training corpus with leakage-safe splits.

What this does, and why:

1. Reads images straight out of the Kaggle zips and the Tomato-Village
   sparse checkout. PlantVillage uses the 'color' variant only.
2. Maps every raw on-disk label through training/labels.py. Anything not in
   that table is excluded by design, never silently guessed.
3. Computes a 64-bit perceptual hash (dHash) for every image.
4. Groups near-duplicates with LSH banding + union-find, then assigns whole
   GROUPS to splits. This is the key anti-leakage step: an augmented or
   re-encoded sibling can never land in a different split from its original.
5. Stratifies by (crop, condition) so rare classes keep representation.
6. Normalises every image to 256px on the short side and writes a manifest
   carrying split, crop, condition, kind, source dataset and dHash, so the
   origin of every training image stays traceable.

Usage:
    python3 training/prepare_data.py
    python3 training/prepare_data.py --limit-per-class 50   # fast smoke run
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import labels as L  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")
SHORT_SIDE = 256
VAL_FRAC, TEST_FRAC = 0.15, 0.15
NEAR_DUP_HAMMING = 4
BANDS = 4  # 64-bit hash -> 4 x 16-bit bands for LSH
# dHash alone false-positives badly on this corpus: PlantVillage leaves sit on
# uniform grey backgrounds, so the 8x9 downsample encodes mostly smooth
# background and collides. Measured on the real data, same-class "duplicate"
# pairs had a median mean-absolute pixel difference of 23.8/255 - i.e. most
# were NOT duplicates. Only ~5% fell at or below 4.0, with a clear gap before
# coincidental matches set in. So every candidate pair is verified at pixel
# level and only merged when it is genuinely near-identical.
THUMB = 32
MAX_PIXEL_MAD = 4.0


# ------------------------------------------------------------------ hashing
def dhash_and_thumb(data: bytes, size: int = 8):
    """Return (dHash, 32x32 grayscale thumbnail) or (None, None)."""
    try:
        base = Image.open(io.BytesIO(data)).convert("L")
    except Exception:  # noqa: BLE001 - unreadable/corrupt image
        return None, None
    thumb = np.asarray(base.resize((THUMB, THUMB), Image.LANCZOS), dtype=np.uint8)
    im = base.resize((size + 1, size), Image.LANCZOS)
    px = list(getattr(im, "get_flattened_data", im.getdata)())
    h = 0
    for row in range(size):
        for col in range(size):
            a = px[row * (size + 1) + col]
            b = px[row * (size + 1) + col + 1]
            h = (h << 1) | (1 if a < b else 0)
    return h, thumb


def pixel_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class Union:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_near_duplicates(
    hashes: list[int], thumbs: list[np.ndarray]
) -> tuple[list[int], dict]:
    """Connected components of VERIFIED near-duplicates.

    Exact O(n^2) comparison is infeasible at this scale, so candidates are
    bucketed by 16-bit LSH bands and compared only inside a bucket. Each
    candidate must then pass a pixel-level check, which is what stops dHash
    collisions from chaining unrelated images into one giant cluster.
    """
    u = Union(len(hashes))
    cand = merged = 0
    seen: set[tuple[int, int]] = set()
    for band in range(BANDS):
        shift = band * 16
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> shift) & 0xFFFF].append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 400:
                continue  # huge bucket = degenerate low-entropy hash
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    ia, ib = members[a_i], members[b_i]
                    key = (ia, ib) if ia < ib else (ib, ia)
                    if key in seen:
                        continue
                    seen.add(key)
                    if hamming(hashes[ia], hashes[ib]) > NEAR_DUP_HAMMING:
                        continue
                    cand += 1
                    if pixel_mad(thumbs[ia], thumbs[ib]) <= MAX_PIXEL_MAD:
                        u.union(ia, ib)
                        merged += 1
    stats = {"hash_candidate_pairs": cand, "pixel_verified_pairs": merged,
             "rejected_as_hash_collision": cand - merged}
    return [u.find(i) for i in range(len(hashes))], stats


# ------------------------------------------------------------- collection
class Item:
    __slots__ = ("source", "raw_label", "ref", "crop", "cond", "kind",
                 "dhash", "group", "split", "out_name")

    def __init__(self, source: str, raw_label: str, ref: str) -> None:
        self.source, self.raw_label, self.ref = source, raw_label, ref


def collect_plantvillage(limit: int | None) -> tuple[list[Item], zipfile.ZipFile]:
    arc = config.RAW_DIR / config.DATASETS["plantvillage"]["archive"]
    z = zipfile.ZipFile(arc)
    root = config.DATASETS["plantvillage"]["root_in_archive"]
    per: Counter = Counter()
    out: list[Item] = []
    for n in z.namelist():
        if not n.startswith(root + "/") or not n.lower().endswith(IMG_EXT):
            continue
        parts = n.split("/")
        if len(parts) < 4:
            continue
        raw = parts[2]
        if L.resolve("plantvillage", raw) is None:
            continue
        if limit and per[raw] >= limit:
            continue
        per[raw] += 1
        out.append(Item("plantvillage", raw, n))
    return out, z


def collect_potato_pld(limit: int | None) -> tuple[list[Item], zipfile.ZipFile]:
    arc = config.RAW_DIR / config.DATASETS["potato_pld"]["archive"]
    z = zipfile.ZipFile(arc)
    per: Counter = Counter()
    out: list[Item] = []
    for n in z.namelist():
        if not n.lower().endswith(IMG_EXT):
            continue
        parts = n.split("/")
        if len(parts) < 3:
            continue
        raw = parts[2]  # PLD_3_Classes_256/<Split>/<Class>/img
        if L.resolve("potato_pld", raw) is None:
            continue
        if limit and per[raw] >= limit:
            continue
        per[raw] += 1
        out.append(Item("potato_pld", raw, n))
    return out, z


def collect_tomato_village(limit: int | None) -> list[Item]:
    base = config.RAW_DIR / "tv_images" / config.DATASETS["tomato_village"]["root_in_archive"]
    per: Counter = Counter()
    out: list[Item] = []
    if not base.exists():
        print(f"  WARNING: {base} missing - Tomato-Village skipped", file=sys.stderr)
        return out
    for p in sorted(base.rglob("*")):
        if not p.is_file() or not p.suffix.lower() in IMG_EXT:
            continue
        raw = p.parent.name
        if L.resolve("tomato_village", raw) is None:
            continue
        if limit and per[raw] >= limit:
            continue
        per[raw] += 1
        out.append(Item("tomato_village", raw, str(p)))
    return out


def read_bytes(item: Item, zips: dict[str, zipfile.ZipFile]) -> bytes | None:
    try:
        if item.source in zips:
            return zips[item.source].read(item.ref)
        return Path(item.ref).read_bytes()
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------- splitting
def assign_splits(items: list[Item], seed: int) -> dict[str, int]:
    """Stratify by (crop, condition); assign whole duplicate-groups together."""
    rng = random.Random(seed)

    # Each duplicate group is assigned to EXACTLY ONE split. A group is filed
    # under its majority (crop, condition) so stratification still holds, but
    # the group - not the image - is the unit, which makes it structurally
    # impossible for a duplicate to straddle train and test.
    groups_all: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        groups_all[it.group].append(it)

    by_class: dict[tuple[str, str], dict[int, list[Item]]] = defaultdict(dict)
    for gid, members in groups_all.items():
        primary = Counter((m.crop, m.cond) for m in members).most_common(1)[0][0]
        by_class[primary][gid] = members

    counts: Counter = Counter()
    for (_crop, _cond), groups in sorted(by_class.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        n_img = sum(len(groups[k]) for k in keys)
        want_test = n_img * TEST_FRAC
        want_val = n_img * VAL_FRAC
        got_test = got_val = 0
        for k in keys:
            members = groups[k]
            # Smallest classes still need at least one val/test group.
            if got_test < want_test:
                split = "test"
                got_test += len(members)
            elif got_val < want_val:
                split = "val"
                got_val += len(members)
            else:
                split = "train"
            for it in members:
                it.split = split
                counts[split] += 1
    return dict(counts)


# ------------------------------------------------------------------- output
def normalise_and_save(data: bytes, dest: Path) -> tuple[int, int] | None:
    try:
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    w, h = im.size
    if min(w, h) > SHORT_SIDE:
        if w < h:
            nw, nh = SHORT_SIDE, round(h * SHORT_SIDE / w)
        else:
            nh, nw = SHORT_SIDE, round(w * SHORT_SIDE / h)
        im = im.resize((nw, nh), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=92)
    return im.size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap images per raw class (smoke testing)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    out_root = config.DATA_DIR / "prepared"
    img_root = out_root / "images"

    print("collecting ...")
    zips: dict[str, zipfile.ZipFile] = {}
    pv, z1 = collect_plantvillage(args.limit_per_class)
    zips["plantvillage"] = z1
    pld, z2 = collect_potato_pld(args.limit_per_class)
    zips["potato_pld"] = z2
    tv = collect_tomato_village(args.limit_per_class)
    items = pv + pld + tv
    print(f"  plantvillage {len(pv)}  potato_pld {len(pld)}  tomato_village {len(tv)}")

    for it in items:
        crop, cond = L.resolve(it.source, it.raw_label)
        it.crop, it.cond, it.kind = crop, cond.id, cond.kind

    print("hashing (perceptual + thumbnails) ...")
    kept: list[Item] = []
    thumbs: list[np.ndarray] = []
    unreadable = 0
    for i, it in enumerate(items):
        data = read_bytes(it, zips)
        h, th = dhash_and_thumb(data) if data else (None, None)
        if h is None:
            unreadable += 1
            continue
        it.dhash = h
        kept.append(it)
        thumbs.append(th)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(items)}")
    print(f"  hashed {len(kept)}, unreadable {unreadable}")

    print("grouping near-duplicates (pixel-verified) ...")
    groups, dup_stats = group_near_duplicates([it.dhash for it in kept], thumbs)
    print(f"  {dup_stats}")
    for it, g in zip(kept, groups):
        it.group = g
    n_groups = len(set(groups))
    print(f"  {len(kept)} images -> {n_groups} groups "
          f"({len(kept) - n_groups} near/exact duplicates absorbed)")

    # cross-source duplicate report
    by_group: dict[int, set[str]] = defaultdict(set)
    for it in kept:
        by_group[it.group].add(it.source)
    cross = sum(1 for v in by_group.values() if len(v) > 1)
    print(f"  groups spanning >1 source dataset: {cross}")

    print("assigning splits (group-aware, stratified) ...")
    split_counts = assign_splits(kept, args.seed)
    print("  ", split_counts)

    print("writing images + manifest ...")
    img_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.csv"
    written = 0
    dist: Counter = Counter()
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "split", "crop", "condition", "kind",
                    "source_dataset", "raw_label", "dhash", "dup_group"])
        for idx, it in enumerate(kept):
            data = read_bytes(it, zips)
            if data is None:
                continue
            rel = Path(it.crop) / it.cond / f"{it.source}_{idx:06d}.jpg"
            if normalise_and_save(data, img_root / rel) is None:
                continue
            w.writerow([str(rel), it.split, it.crop, it.cond, it.kind,
                        it.source, it.raw_label, it.dhash, it.group])
            written += 1
            dist[(it.split, it.crop, it.cond)] += 1
            if written % 5000 == 0:
                print(f"  {written}/{len(kept)}")
    print(f"  wrote {written} images")

    # class mapping, consumed by training AND served by the API
    mapping = {
        "crops": L.crop_index(),
        "conditions": {
            crop: {
                cid: {
                    "index": i,
                    "display": L.CONDITIONS[crop][cid].display,
                    "kind": L.CONDITIONS[crop][cid].kind,
                    "pathogen": L.CONDITIONS[crop][cid].pathogen,
                }
                for cid, i in L.condition_index(crop).items()
            }
            for crop in sorted(L.CONDITIONS)
        },
        "preprocessing": {"short_side": SHORT_SIDE, "jpeg_quality": 92},
        "split_fractions": {"train": round(1 - VAL_FRAC - TEST_FRAC, 2),
                            "val": VAL_FRAC, "test": TEST_FRAC},
        "near_dup_hamming_threshold": NEAR_DUP_HAMMING,
        "seed": args.seed,
    }
    (out_root / "class_mapping.json").write_text(json.dumps(mapping, indent=2))
    # datasets/ is gitignored, but the backend needs the class mapping at
    # runtime and it must be reviewable in diffs - so keep a committed copy.
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2)
    )

    stats = {
        "n_images": written,
        "n_dup_groups": n_groups,
        "duplicates_absorbed": len(kept) - n_groups,
        "groups_spanning_multiple_sources": cross,
        "unreadable": unreadable,
        "duplicate_detection": dup_stats,
        "split_counts": split_counts,
        "per_split_crop_condition": {
            f"{s}/{c}/{cd}": n for (s, c, cd), n in sorted(dist.items())
        },
    }
    (out_root / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    (config.REPORT_DIR / "prepare_stats.json").write_text(
        json.dumps(stats, indent=2)
    )
    print(f"\nmanifest: {manifest}")
    print(f"mapping : {out_root / 'class_mapping.json'}")
    print(f"stats   : {out_root / 'prepare_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
