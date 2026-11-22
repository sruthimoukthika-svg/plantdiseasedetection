#!/usr/bin/env python3
"""Stage 1: inspect every dataset and emit a reproducible report.

Reads archives by index where possible (no full extraction), lists label
directories as they actually exist on disk, counts images per class, and runs
perceptual-hash duplicate detection within and across datasets.

Usage:
    python3 training/inspect_datasets.py            # inspect everything
    python3 training/inspect_datasets.py --quick    # skip perceptual hashing
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___")


# ---------------------------------------------------------------- helpers
def _is_img(name: str) -> bool:
    return name.lower().endswith(IMG_EXT) and not name.endswith("/")


def dhash(data: bytes, size: int = 8) -> int:
    """Perceptual hash: catches resized / re-encoded duplicates that byte
    hashing misses. Returns a 64-bit int."""
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("L").resize(
        (size + 1, size), Image.LANCZOS
    )
    px = list(getattr(im, "get_flattened_data", im.getdata)())
    h = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            h = (h << 1) | (1 if left < right else 0)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def blobless_clone(url: str, dest: Path) -> Path:
    """Clone tree metadata only - lets us read label directories without
    downloading gigabytes of images."""
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
         url, str(dest)],
        check=True, capture_output=True,
    )
    return dest


def git_ls(repo: Path, treeish: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "--name-only", treeish],
        capture_output=True, text=True,
    )
    return [l for l in out.stdout.splitlines() if l] if out.returncode == 0 else []


# ------------------------------------------------------- zip-based datasets
def inspect_zip_tree(archive: Path, root: str, depth_class: int) -> dict:
    """Count images per class directory inside a zip, by index only."""
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if _is_img(n)]
    per_class: Counter = Counter()
    variants: Counter = Counter()
    for n in names:
        parts = n.split("/")
        if len(parts) > depth_class:
            variants[parts[1]] += 1
            key = "/".join(parts[2:depth_class + 1]) if depth_class > 2 else parts[2]
            if n.startswith(root) or root == "":
                per_class[key] += 1
    return {
        "total_image_entries": len(names),
        "top_level_groups": dict(variants),
        "classes": dict(sorted(per_class.items())),
        "n_classes": len(per_class),
        "n_images": sum(per_class.values()),
    }


def inspect_plantvillage() -> dict:
    arc = config.RAW_DIR / config.DATASETS["plantvillage"]["archive"]
    if not arc.exists():
        return {"error": f"missing archive {arc}"}
    with zipfile.ZipFile(arc) as z:
        names = [n for n in z.namelist() if _is_img(n)]
    variants: Counter = Counter()
    per: dict[str, Counter] = defaultdict(Counter)
    for n in names:
        p = n.split("/")
        if len(p) >= 4:
            variants[p[1]] += 1
            per[p[1]][p[2]] += 1
    used = config.DATASETS["plantvillage"]["variant_used"]
    classes = dict(sorted(per.get(used, Counter()).items()))
    crops = defaultdict(lambda: {"classes": {}, "n_images": 0})
    for cls, n in classes.items():
        crop = cls.split("___")[0].lower()
        for want in config.SUPPORTED_CROPS:
            if crop.startswith(want):
                crops[want]["classes"][cls] = n
                crops[want]["n_images"] += n
    return {
        "variants": dict(variants),
        "variant_used": used,
        "n_classes": len(classes),
        "n_images": sum(classes.values()),
        "classes": classes,
        "supported_crops": {k: dict(v) for k, v in crops.items()},
    }


def inspect_potato_pld() -> dict:
    arc = config.RAW_DIR / config.DATASETS["potato_pld"]["archive"]
    if not arc.exists():
        return {"error": f"missing archive {arc}"}
    with zipfile.ZipFile(arc) as z:
        names = [n for n in z.namelist() if _is_img(n)]
    per: Counter = Counter()
    for n in names:
        p = n.split("/")
        if len(p) >= 3:
            per[f"{p[1]}/{p[2]}"] += 1
    uuid_hits = sum(1 for n in names if UUID_RE.search(n.split("/")[-1]))
    return {
        "n_images": len(names),
        "splits_and_classes": dict(sorted(per.items())),
        "plantvillage_naming_hits": uuid_hits,
    }


# ----------------------------------------------------- github-based datasets
def inspect_github_imagefolder(key: str, dest: Path, subroot: str = "") -> dict:
    url = config.DATASETS[key]["ref"]
    try:
        repo = blobless_clone(url, dest)
    except subprocess.CalledProcessError as e:
        return {"error": f"clone failed: {e.stderr[:200] if e.stderr else e}"}
    base = f"HEAD:{subroot}" if subroot else "HEAD"
    splits = [s for s in git_ls(repo, base) if "." not in s]
    out: dict = {"splits": {}}
    for sp in splits:
        prefix = f"{subroot}/{sp}" if subroot else sp
        classes = git_ls(repo, f"HEAD:{prefix}")
        counts = {c: len(git_ls(repo, f"HEAD:{prefix}/{c}")) for c in classes}
        out["splits"][sp] = {
            "n_classes": len(counts),
            "n_images": sum(counts.values()),
            "classes": dict(sorted(counts.items())),
        }
    return out


# ---------------------------------------------------- huggingface datasets
def inspect_hf(repo_id: str) -> dict:
    url = f"https://huggingface.co/api/datasets/{repo_id}?full=true"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            meta = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    files = [s.get("rfilename", "") for s in meta.get("siblings", [])]
    imgs = [f for f in files if _is_img(f)]
    info = (meta.get("cardData") or {}).get("dataset_info") or {}
    out: dict = {
        "downloads": meta.get("downloads"),
        "n_files": len(files),
        "n_image_files": len(imgs),
        "declared_features": info.get("features"),
        "declared_splits": info.get("splits"),
    }
    if imgs:
        per: Counter = Counter()
        for f in imgs:
            p = f.split("/")
            if len(p) >= 3:
                per[f"{p[1]}/{p[2]}"] += 1
        out["splits_and_classes"] = dict(sorted(per.items()))
        # augmentation / leakage analysis on PlantVillage-style source ids
        src: dict[str, set] = defaultdict(set)
        for f in imgs:
            m = UUID_RE.search(f)
            if m:
                src[m.group(0)[:-3]].add(f.split("/")[1])
        if src:
            out["unique_source_images"] = len(src)
            out["copies_per_source"] = round(len(imgs) / len(src), 2)
            out["sources_in_multiple_splits"] = sum(
                1 for v in src.values() if len(v) > 1
            )
    return out


# ------------------------------------------------------- duplicate detection
def potato_cross_dataset_dupes(limit: int | None = None) -> dict:
    """Is PLD independent of PlantVillage, or a re-encoded copy of it?"""
    pv_arc = config.RAW_DIR / config.DATASETS["plantvillage"]["archive"]
    pld_arc = config.RAW_DIR / config.DATASETS["potato_pld"]["archive"]
    if not (pv_arc.exists() and pld_arc.exists()):
        return {"error": "archives not present"}
    with zipfile.ZipFile(pv_arc) as pv, zipfile.ZipFile(pld_arc) as pld:
        pv_names = [n for n in pv.namelist()
                    if n.startswith("plantvillage dataset/color/Potato___")
                    and _is_img(n)][:limit]
        pld_names = [n for n in pld.namelist() if _is_img(n)][:limit]
        pv_h: dict[int, list[str]] = defaultdict(list)
        for n in pv_names:
            pv_h[dhash(pv.read(n))].append(n)
        pld_h: dict[int, list[str]] = defaultdict(list)
        for n in pld_names:
            pld_h[dhash(pld.read(n))].append(n)
    exact = set(pv_h) & set(pld_h)
    near = 0
    pv_keys = list(pv_h)
    for k2 in pld_h:
        if any(hamming(k1, k2) <= 4 for k1 in pv_keys):
            near += 1
    cross_split = sum(
        1 for v in pld_h.values() if len({n.split("/")[1] for n in v}) > 1
    )
    internal_dupes = sum(len(v) - 1 for v in pld_h.values() if len(v) > 1)
    return {
        "plantvillage_potato_hashed": len(pv_names),
        "pld_hashed": len(pld_names),
        "exact_dhash_matches_across_datasets": len(exact),
        "near_dhash_matches_hamming_le_4": near,
        "pld_dhashes_in_multiple_own_splits": cross_split,
        "pld_internal_duplicate_images": internal_dupes,
    }


# ----------------------------------------------------------------- reporting
def build_report(quick: bool = False) -> dict:
    work = config.RAW_DIR / "_git_meta"
    rep: dict = {"generated_by": "training/inspect_datasets.py", "datasets": {}}
    rep["datasets"]["plantvillage"] = inspect_plantvillage()
    rep["datasets"]["potato_pld"] = inspect_potato_pld()
    rep["datasets"]["tomato_village"] = inspect_github_imagefolder(
        "tomato_village", work / "tomatovillage",
        config.DATASETS["tomato_village"]["root_in_archive"],
    )
    rep["datasets"]["plantdoc"] = inspect_github_imagefolder(
        "plantdoc", work / "plantdoc",
    )
    rep["datasets"]["agml_potato"] = inspect_hf(config.DATASETS["agml_potato"]["ref"])
    rep["datasets"]["grape_hf"] = inspect_hf(config.DATASETS["grape_hf"]["ref"])
    rep["duplicate_analysis"] = (
        {"skipped": "--quick"} if quick else potato_cross_dataset_dupes()
    )
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip perceptual-hash duplicate detection")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = build_report(quick=args.quick)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else config.REPORT_DIR / "dataset_report.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(f"wrote {out}")
    for name, d in rep["datasets"].items():
        if "error" in d:
            print(f"  {name:16s} ERROR: {d['error']}")
        else:
            if "splits" in d:
                n = sum(v["n_images"] for v in d["splits"].values())
                cls = max(v["n_classes"] for v in d["splits"].values())
                print(f"  {name:16s} ok  ({n} images, {cls} classes)")
            else:
                n = (d.get("n_images") or d.get("n_image_files")
                     or d.get("n_files"))
                print(f"  {name:16s} ok  ({n} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
