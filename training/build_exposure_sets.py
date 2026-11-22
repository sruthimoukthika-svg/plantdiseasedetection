#!/usr/bin/env python3
"""Stage 4c: build outlier-exposure and held-out rejection sets.

Calibrated distance scoring rejected random photographs perfectly (FAR
0.0000) but accepted 22.7% of unsupported PlantVillage species. That
failure is structural: a 3-class species gate must assign an apple leaf to
tomato, potato or grape, because it has no other option. The fix is to give
it a fourth, explicit "unsupported" class, trained by outlier exposure.

The split is BY SPECIES, never by image. Held-out species are absent from
training entirely, so the reported false-accept rate measures whether
rejection generalises to a species never seen as an outlier - not whether
it memorised the ones it was shown.

  exposure (train) : Apple, Corn, Orange, Soybean, Squash, Strawberry
  held-out (eval)  : Blueberry, Cherry, Peach, Pepper bell, Raspberry

Usage: python3 training/build_exposure_sets.py --per-species 1200
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SHORT_SIDE = 256
EXPOSURE_SPECIES = ["Apple", "Corn_(maize)", "Orange", "Soybean",
                    "Squash", "Strawberry"]
HELDOUT_SPECIES = ["Blueberry", "Cherry_(including_sour)", "Peach",
                   "Pepper,_bell", "Raspberry"]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=1200)
    args = ap.parse_args()

    arc = config.RAW_DIR / config.DATASETS["plantvillage"]["archive"]
    root = config.DATASETS["plantvillage"]["root_in_archive"]
    out_root = config.DATA_DIR / "ood"

    groups = {"unsupported_lab_exposure": EXPOSURE_SPECIES,
              "unsupported_lab_heldout": HELDOUT_SPECIES}
    # a species must never appear on both sides
    assert not (set(EXPOSURE_SPECIES) & set(HELDOUT_SPECIES))

    by_species: dict[str, list[str]] = {}
    with zipfile.ZipFile(arc) as z:
        for n in z.namelist():
            if not n.startswith(root + "/") or not n.lower().endswith(
                    (".jpg", ".jpeg", ".png")):
                continue
            p = n.split("/")
            if len(p) < 4 or p[2].startswith(("Tomato", "Potato", "Grape")):
                continue
            by_species.setdefault(p[2].split("___")[0], []).append(n)

        summary = {}
        for gname, species in groups.items():
            dest_root = out_root / gname
            counts = {}
            total = 0
            for sp in species:
                members = sorted(by_species.get(sp, []))
                if not members:
                    print(f"  WARNING: species {sp!r} not found", file=sys.stderr)
                    continue
                # stride-sample so all of the species' disease classes appear
                step = max(1, len(members) // args.per_species)
                take = members[::step][:args.per_species]
                for i, m in enumerate(take):
                    if save_norm(z.read(m), dest_root / sp / f"{i:05d}.jpg"):
                        total += 1
                        counts[sp] = counts.get(sp, 0) + 1
            summary[gname] = {"species": species, "n_images": total,
                              "per_species": counts}
            print(f"{gname}: {total} images across {len(counts)} species")

    (out_root / "exposure_summary.json").write_text(json.dumps(summary, indent=2))
    (config.REPORT_DIR / "exposure_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {out_root / 'exposure_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
