"""Centralised label mapping across datasets.

Design rules, each traceable to a Stage 1 finding:

* Equivalent conditions are merged ONLY when they are the same biological
  entity. PlantVillage `Tomato___Early_blight` and Tomato-Village
  `Early_blight` are both Alternaria solani on tomato, so they merge.
* Biologically distinct conditions are NEVER merged. Spotted Wilt Virus,
  Tomato Yellow Leaf Curl Virus and Tomato mosaic virus stay three classes.
* Not every class is a disease. Spider mites and Leaf Miner are pests; the
  three Deficiency classes are nutrient disorders. Each class carries a
  `kind` so the UI and API can say "condition", not "disease".
* Upstream typos are fixed for display only; the raw on-disk label is kept
  for provenance.

Nothing here was inferred from a dataset name - every raw label below was
read off the real archive directory listings.
"""
from __future__ import annotations

from dataclasses import dataclass

DISEASE, PEST, DEFICIENCY, HEALTHY = "disease", "pest", "deficiency", "healthy"


@dataclass(frozen=True)
class Condition:
    id: str            # stable machine id, unique within its crop
    display: str       # user-facing name
    kind: str          # disease | pest | deficiency | healthy
    pathogen: str = "" # causal agent, empty for healthy/deficiency


# ----------------------------------------------------------------- tomato
TOMATO: dict[str, Condition] = {
    "bacterial_spot":    Condition("bacterial_spot", "Bacterial Spot", DISEASE, "Xanthomonas spp."),
    "early_blight":      Condition("early_blight", "Early Blight", DISEASE, "Alternaria solani"),
    "late_blight":       Condition("late_blight", "Late Blight", DISEASE, "Phytophthora infestans"),
    "leaf_mold":         Condition("leaf_mold", "Leaf Mold", DISEASE, "Passalora fulva"),
    "septoria_leaf_spot":Condition("septoria_leaf_spot", "Septoria Leaf Spot", DISEASE, "Septoria lycopersici"),
    "target_spot":       Condition("target_spot", "Target Spot", DISEASE, "Corynespora cassiicola"),
    "yellow_leaf_curl_virus": Condition("yellow_leaf_curl_virus", "Yellow Leaf Curl Virus", DISEASE, "TYLCV"),
    "mosaic_virus":      Condition("mosaic_virus", "Mosaic Virus", DISEASE, "ToMV"),
    "spotted_wilt_virus":Condition("spotted_wilt_virus", "Spotted Wilt Virus", DISEASE, "TSWV"),
    "spider_mites":      Condition("spider_mites", "Two-Spotted Spider Mites", PEST, "Tetranychus urticae"),
    "leaf_miner":        Condition("leaf_miner", "Leaf Miner", PEST, "Liriomyza spp."),
    "magnesium_deficiency": Condition("magnesium_deficiency", "Magnesium Deficiency", DEFICIENCY),
    "nitrogen_deficiency":  Condition("nitrogen_deficiency", "Nitrogen Deficiency", DEFICIENCY),
    "potassium_deficiency": Condition("potassium_deficiency", "Potassium Deficiency", DEFICIENCY),
    "healthy":           Condition("healthy", "Healthy", HEALTHY),
}

# ----------------------------------------------------------------- potato
POTATO: dict[str, Condition] = {
    "early_blight": Condition("early_blight", "Early Blight", DISEASE, "Alternaria solani"),
    "late_blight":  Condition("late_blight", "Late Blight", DISEASE, "Phytophthora infestans"),
    "healthy":      Condition("healthy", "Healthy", HEALTHY),
}

# ------------------------------------------------------------------ grape
GRAPE: dict[str, Condition] = {
    "black_rot":   Condition("black_rot", "Black Rot", DISEASE, "Guignardia bidwellii"),
    "esca":        Condition("esca", "Esca (Black Measles)", DISEASE, "Phaeomoniella chlamydospora"),
    "leaf_blight": Condition("leaf_blight", "Leaf Blight (Isariopsis Leaf Spot)", DISEASE, "Pseudocercospora vitis"),
    "healthy":     Condition("healthy", "Healthy", HEALTHY),
}

CONDITIONS: dict[str, dict[str, Condition]] = {
    "tomato": TOMATO, "potato": POTATO, "grape": GRAPE,
}

# ------------------------------------------------------------------------
# Raw label -> (crop, condition id). Keys are EXACT on-disk directory names.
# A raw label absent from this table is excluded from training by design.
# ------------------------------------------------------------------------
RAW_TO_CANONICAL: dict[tuple[str, str], tuple[str, str]] = {
    # --- PlantVillage (variant: color) ---
    ("plantvillage", "Tomato___Bacterial_spot"):                   ("tomato", "bacterial_spot"),
    ("plantvillage", "Tomato___Early_blight"):                     ("tomato", "early_blight"),
    ("plantvillage", "Tomato___Late_blight"):                      ("tomato", "late_blight"),
    ("plantvillage", "Tomato___Leaf_Mold"):                        ("tomato", "leaf_mold"),
    ("plantvillage", "Tomato___Septoria_leaf_spot"):               ("tomato", "septoria_leaf_spot"),
    ("plantvillage", "Tomato___Spider_mites Two-spotted_spider_mite"): ("tomato", "spider_mites"),
    ("plantvillage", "Tomato___Target_Spot"):                      ("tomato", "target_spot"),
    ("plantvillage", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"):    ("tomato", "yellow_leaf_curl_virus"),
    ("plantvillage", "Tomato___Tomato_mosaic_virus"):              ("tomato", "mosaic_virus"),
    ("plantvillage", "Tomato___healthy"):                          ("tomato", "healthy"),
    ("plantvillage", "Potato___Early_blight"):                     ("potato", "early_blight"),
    ("plantvillage", "Potato___Late_blight"):                      ("potato", "late_blight"),
    ("plantvillage", "Potato___healthy"):                          ("potato", "healthy"),
    ("plantvillage", "Grape___Black_rot"):                         ("grape", "black_rot"),
    ("plantvillage", "Grape___Esca_(Black_Measles)"):              ("grape", "esca"),
    ("plantvillage", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"):("grape", "leaf_blight"),
    ("plantvillage", "Grape___healthy"):                           ("grape", "healthy"),

    # --- Tomato-Village, Variant-a. Early/Late blight and Healthy merge into
    #     the PlantVillage classes (same biology, different environment).
    #     'Pottassium' is the upstream spelling; display name is corrected.
    ("tomato_village", "Early_blight"):           ("tomato", "early_blight"),
    ("tomato_village", "Late_blight"):            ("tomato", "late_blight"),
    ("tomato_village", "Healthy"):                ("tomato", "healthy"),
    ("tomato_village", "Leaf Miner"):             ("tomato", "leaf_miner"),
    ("tomato_village", "Magnesium Deficiency"):   ("tomato", "magnesium_deficiency"),
    ("tomato_village", "Nitrogen Deficiency"):    ("tomato", "nitrogen_deficiency"),
    ("tomato_village", "Pottassium Deficiency"):  ("tomato", "potassium_deficiency"),
    ("tomato_village", "Spotted Wilt Virus"):     ("tomato", "spotted_wilt_virus"),

    # --- Potato PLD (corrected slug). Merges with PlantVillage potato;
    #     verified independent (0 exact / 0 near perceptual-hash matches).
    ("potato_pld", "Early_Blight"): ("potato", "early_blight"),
    ("potato_pld", "Late_Blight"):  ("potato", "late_blight"),
    ("potato_pld", "Healthy"):      ("potato", "healthy"),
}

# ------------------------------------------------------------------------
# Evaluation-only mappings. These datasets are NEVER used for training.
# ------------------------------------------------------------------------
# PlantDoc: real-world images. Only the supported crops are mapped.
PLANTDOC_EVAL: dict[str, tuple[str, str]] = {
    "Tomato Early blight leaf":        ("tomato", "early_blight"),
    "Tomato Septoria leaf spot":       ("tomato", "septoria_leaf_spot"),
    "Tomato leaf bacterial spot":      ("tomato", "bacterial_spot"),
    "Tomato leaf late blight":         ("tomato", "late_blight"),
    "Tomato leaf mosaic virus":        ("tomato", "mosaic_virus"),
    "Tomato leaf yellow virus":        ("tomato", "yellow_leaf_curl_virus"),
    "Tomato mold leaf":                ("tomato", "leaf_mold"),
    "Tomato leaf":                     ("tomato", "healthy"),
    "Potato leaf early blight":        ("potato", "early_blight"),
    "Potato leaf late blight":         ("potato", "late_blight"),
    "grape leaf black rot":            ("grape", "black_rot"),
    "grape leaf":                      ("grape", "healthy"),
    # 'Tomato two spotted spider mites leaf' deliberately absent:
    # 2 train / 0 test images (see config.EXCLUDED_CLASSES).
}

# AgML potato: pathogen-level labels, a different taxonomic level from named
# diseases. Only these two map cleanly; 'Fungi' is a SUPERSET that contains
# Early blight and must not be mapped onto it.
AGML_POTATO_EVAL: dict[str, tuple[str, str]] = {
    "Phytopthora": ("potato", "late_blight"),
    "Healthy":     ("potato", "healthy"),
}
AGML_POTATO_UNMAPPABLE = {
    "Fungi": "superset containing Early blight and other fungal diseases",
    "Bacteria": "no PlantVillage potato equivalent",
    "Virus": "no PlantVillage potato equivalent",
    "Pest": "no PlantVillage potato equivalent",
    "Nematode": "no equivalent; only 68 images",
}


def crop_index() -> dict[str, int]:
    """Stable species-gate class index."""
    return {c: i for i, c in enumerate(sorted(CONDITIONS))}


def condition_index(crop: str) -> dict[str, int]:
    """Stable per-crop condition class index."""
    return {cid: i for i, cid in enumerate(sorted(CONDITIONS[crop]))}


def resolve(dataset: str, raw_label: str) -> tuple[str, Condition] | None:
    """Map a raw on-disk label to (crop, Condition), or None if excluded."""
    hit = RAW_TO_CANONICAL.get((dataset, raw_label))
    if hit is None:
        return None
    crop, cid = hit
    return crop, CONDITIONS[crop][cid]
