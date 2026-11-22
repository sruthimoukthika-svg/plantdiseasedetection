#!/usr/bin/env python3
"""Stage 3d: compare trained backbones and record the selection.

Selection is made on VALIDATION macro F1, per the brief. The test split is
reported for transparency but never used to choose, which would leak it.

Usage: python3 training/compare_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

def main() -> int:
    rows = []
    for tr in sorted(config.EVAL_DIR.glob("*_training.json")):
        name = tr.name.replace("_training.json", "")
        te_path = config.EVAL_DIR / f"{name}_test_metrics.json"
        if not te_path.exists():
            continue
        tr_d = json.loads(tr.read_text())
        te_d = json.loads(te_path.read_text())
        # validation macro F1, averaged over the heads that do the real work
        heads = tr_d["heads"]
        val_cond = [heads[k]["best"]["macro_f1"] for k in heads
                    if k.startswith("condition_")]
        rows.append({
            "backbone": name,
            "val_species_macro_f1": round(heads["species"]["best"]["macro_f1"], 4),
            "val_condition_macro_f1_mean": round(sum(val_cond) / len(val_cond), 4),
            "val_tomato_macro_f1": round(heads["condition_tomato"]["best"]["macro_f1"], 4),
            "test_end_to_end_acc": te_d["stages"]["end_to_end"]["accuracy"],
            "test_end_to_end_macro_f1": te_d["stages"]["end_to_end"]["f1_macro"],
            "test_acc_lab_plantvillage":
                te_d.get("per_source_dataset", {}).get("plantvillage", {}).get("end_to_end_accuracy"),
            "test_acc_field_tomato_village":
                te_d.get("per_source_dataset", {}).get("tomato_village", {}).get("end_to_end_accuracy"),
            "inference_ms_mean": te_d["inference_ms_per_image"]["mean"],
            "train_seconds": tr_d.get("train_seconds"),
        })
    if not rows:
        print("no evaluated models found", file=sys.stderr)
        return 1

    # select on validation only
    best = max(rows, key=lambda r: (r["val_condition_macro_f1_mean"],
                                    r["val_species_macro_f1"]))
    out = {"models": rows, "selected": best["backbone"],
           "selection_criterion":
               "highest mean validation macro F1 across condition heads; "
               "test split not used for selection"}
    dest = config.EVAL_DIR / "model_comparison.json"
    dest.write_text(json.dumps(out, indent=2))
    (config.REPORT_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))

    w = max(len(r["backbone"]) for r in rows)
    print(f"{'backbone'.ljust(w)}  valCondF1  valTomF1  testAcc  testF1  "
          f"labAcc  fieldAcc  ms")
    for r in sorted(rows, key=lambda r: -r["val_condition_macro_f1_mean"]):
        print(f"{r['backbone'].ljust(w)}  "
              f"{r['val_condition_macro_f1_mean']:9.4f}  "
              f"{r['val_tomato_macro_f1']:8.4f}  "
              f"{r['test_end_to_end_acc']:7.4f}  "
              f"{r['test_end_to_end_macro_f1']:6.4f}  "
              f"{r['test_acc_lab_plantvillage']:6.4f}  "
              f"{r['test_acc_field_tomato_village']:8.4f}  "
              f"{r['inference_ms_mean']:5.1f}")
    print(f"\nSELECTED: {best['backbone']} (on validation macro F1)")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
