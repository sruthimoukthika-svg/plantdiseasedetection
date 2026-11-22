#!/usr/bin/env python3
"""Stage 4a: assemble out-of-distribution evaluation sets.

The brief requires rejection to be tested against animals, people,
buildings, food, random objects, unsupported plants and images with no
visible leaf. Those are not equally hard, so they are kept as separate
named sets and reported separately - an aggregate number would hide the
only case that really matters.

Sets, easiest to hardest:

  non_plant_objects   random photographs (animals, buildings, objects).
                      Trivially far from leaf features; a sanity floor.
  unsupported_field   PlantDoc species outside the three supported crops,
                      photographed in the wild.
  unsupported_lab     PlantVillage's other 24 classes - apple, corn,
                      orange, soybean and so on. THE hard case: real leaf
                      close-ups, same camera setup, same grey background,
                      differing only in species. If rejection works here it
                      works anywhere.

Usage:
    python3 training/build_ood_sets.py
    python3 training/build_ood_sets.py --max-per-set 1500
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import labels as L  # noqa: E402

SHORT_SIDE = 256
IMG_EXT = (".jpg", ".jpeg", ".png")

SUPPORTED_PREFIXES = ("Tomato", "Potato", "Grape")


def save_norm(data: bytes, dest: Path) -> bool:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:  # noqa: BLE001
        return False
    w, h = im.size
    if min(w, h) > SHORT_SIDE:
        if w < h:
            im = im.resize((SHORT_SIDE, round(h * SHORT_SIDE / w)), Image.LANCZOS)
        else:
            im = im.resize((round(w * SHORT_SIDE / h), SHORT_SIDE), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=92)
    return True


def build_unsupported_lab(out: Path, cap: int | None) -> dict:
    """PlantVillage classes outside the three supported crops."""
    arc = config.RAW_DIR / config.DATASETS["plantvillage"]["archive"]
    root = config.DATASETS["plantvillage"]["root_in_archive"]
    counts: dict[str, int] = {}
    n = 0
    with zipfile.ZipFile(arc) as z:
        names = [x for x in z.namelist()
                 if x.startswith(root + "/") and x.lower().endswith(IMG_EXT)]
        # deterministic, class-balanced sampling
        by_class: dict[str, list[str]] = {}
        for x in names:
            parts = x.split("/")
            if len(parts) < 4:
                continue
            cls = parts[2]
            if cls.startswith(SUPPORTED_PREFIXES):
                continue   # supported crop: in-distribution, not OOD
            by_class.setdefault(cls, []).append(x)
        per = max(1, (cap // len(by_class))) if cap else None
        for cls, members in sorted(by_class.items()):
            take = sorted(members)[:per] if per else sorted(members)
            for i, m in enumerate(take):
                if save_norm(z.read(m), out / cls / f"{i:05d}.jpg"):
                    n += 1
                    counts[cls] = counts.get(cls, 0) + 1
    return {"n_images": n, "n_classes": len(counts), "per_class": counts}


def build_unsupported_field(out: Path, cap: int | None) -> dict:
    """PlantDoc species outside the supported crops (real-world backgrounds)."""
    repo = config.RAW_DIR / "plantdoc_images"
    if not repo.exists():
        print("  fetching PlantDoc images (sparse) ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
             config.DATASETS["plantdoc"]["ref"], str(repo)],
            check=True, capture_output=True,
        )
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set",
                        "train", "test"], check=True, capture_output=True)
    supported = set(L.PLANTDOC_EVAL)
    counts: dict[str, int] = {}
    n = 0
    by_class: dict[str, list[Path]] = {}
    for split in ("train", "test"):
        base = repo / split
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name in supported:
                continue
            # also skip the deliberately excluded supported-crop class
            if d.name.startswith(("Tomato", "Potato", "grape")):
                continue
            by_class.setdefault(d.name, []).extend(
                sorted(p for p in d.iterdir()
                       if p.suffix.lower() in IMG_EXT)
            )
    if not by_class:
        return {"n_images": 0, "note": "PlantDoc images unavailable"}
    per = max(1, (cap // len(by_class))) if cap else None
    for cls, members in sorted(by_class.items()):
        take = members[:per] if per else members
        for i, p in enumerate(take):
            if save_norm(p.read_bytes(), out / cls / f"{i:05d}.jpg"):
                n += 1
                counts[cls] = counts.get(cls, 0) + 1
    return {"n_images": n, "n_classes": len(counts), "per_class": counts}


# Caltech-101 classes that are PLANTS. They belong in the unsupported-plant
# category, not the non-plant one, so they are excluded here to keep the
# reported categories honest.
CALTECH_PLANT_CLASSES = {
    "bonsai", "joshua_tree", "lotus", "sunflower", "water_lilly", "strawberry",
}

# Grouping used only for reporting coverage of the categories the brief names.
CALTECH_GROUPS = {
    "people": {"faces", "faces_easy"},
    "buildings": {"minaret", "pagoda", "pyramid"},
    "food": {"pizza"},
    "vehicles": {"airplanes", "car_side", "ferry", "helicopter", "motorbikes",
                 "schooner", "ketch", "inline_skate", "wheelchair"},
    "animals": {
        "ant", "bass", "beaver", "brontosaurus", "butterfly", "cougar_body",
        "cougar_face", "crab", "crayfish", "crocodile", "crocodile_head",
        "dalmatian", "dolphin", "dragonfly", "elephant", "emu", "flamingo",
        "flamingo_head", "gerenuk", "hawksbill", "hedgehog", "ibis",
        "kangaroo", "leopards", "llama", "lobster", "mayfly", "nautilus",
        "octopus", "okapi", "panda", "pigeon", "platypus", "rhino", "rooster",
        "scorpion", "sea_horse", "starfish", "stegosaurus", "tick",
        "trilobite", "wild_cat",
    },
}


def _caltech_group(name: str) -> str:
    for g, members in CALTECH_GROUPS.items():
        if name in members:
            return g
    return "objects"


def build_non_plant(out: Path, cap: int | None) -> dict:
    """Random non-plant photographs: people, animals, buildings, food, objects.

    Sourced from Caltech-101, which is parquet-native on the Hub and
    full-resolution. Full resolution matters: a low-resolution set such as
    CIFAR would be rejected by the blur pre-check rather than by the model,
    which would flatter these numbers.
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        return {"n_images": 0, "error": f"datasets unavailable: {e}"}
    target = cap or 1000
    counts: dict[str, int] = {}
    groups: dict[str, int] = {}
    skipped_plant = 0
    n = 0
    try:
        dsx = load_dataset("dpdl-benchmark/caltech101", split="test",
                           streaming=True)
        names = dsx.features["label"].names
        for ex in dsx:
            if n >= target:
                break
            cls = names[ex["label"]]
            if cls in CALTECH_PLANT_CLASSES:
                skipped_plant += 1
                continue
            grp = _caltech_group(cls)
            buf = io.BytesIO()
            ex["image"].convert("RGB").save(buf, "JPEG", quality=92)
            if save_norm(buf.getvalue(), out / grp / f"{cls}_{n:05d}.jpg"):
                n += 1
                counts[cls] = counts.get(cls, 0) + 1
                groups[grp] = groups.get(grp, 0) + 1
    except Exception as e:  # noqa: BLE001
        return {"n_images": n, "per_group": groups,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"n_images": n, "n_classes": len(counts), "per_group": groups,
            "plant_classes_excluded": sorted(CALTECH_PLANT_CLASSES),
            "plant_images_skipped": skipped_plant}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-set", type=int, default=1500)
    args = ap.parse_args()
    root = config.DATA_DIR / "ood"
    summary = {}
    for name, fn in (
        ("unsupported_lab", build_unsupported_lab),
        ("unsupported_field", build_unsupported_field),
        ("non_plant_objects", build_non_plant),
    ):
        dest = root / name
        if dest.exists() and any(dest.rglob("*.jpg")):
            n = len(list(dest.rglob("*.jpg")))
            print(f"{name}: already built ({n} images)")
            summary[name] = {"n_images": n, "cached": True}
            continue
        print(f"building {name} ...")
        summary[name] = fn(dest, args.max_per_set)
        print(f"  {name}: {summary[name].get('n_images')} images"
              + (f"  ERROR {summary[name]['error']}" if "error" in summary[name] else ""))
    root.mkdir(parents=True, exist_ok=True)
    (root / "ood_summary.json").write_text(json.dumps(summary, indent=2))
    (config.REPORT_DIR / "ood_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {root / 'ood_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
