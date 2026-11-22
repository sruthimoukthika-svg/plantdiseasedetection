"""Central configuration: dataset registry, supported crops, label mapping.

Every fact in DATASETS below was verified by inspecting the real archives in
this environment (see training/inspect_datasets.py and
training/reports/dataset_report.md). Nothing here is assumed from a name.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path("/home/user/datasets_raw")   # archives; never committed
DATA_DIR = REPO_ROOT / "datasets"           # extracted/prepared; never committed
REPORT_DIR = REPO_ROOT / "training" / "reports"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
EVAL_DIR = REPO_ROOT / "evaluation_results"

SEED = 42
IMAGE_SIZE = 224

# Crops the product supports. Nothing outside this list may be predicted.
SUPPORTED_CROPS = ("tomato", "potato", "grape")

DATASETS = {
    "plantvillage": {
        "source": "kaggle",
        "ref": "abdallahalidev/plantvillage-dataset",
        "archive": "plantvillage.zip",
        "bytes": 2189386419,
        "license": "CC BY-NC-SA 4.0",
        # 3 variants x 54,305 images; we use 'color' only.
        "variant_used": "color",
        "root_in_archive": "plantvillage dataset/color",
        "n_classes_total": 38,
        "role": "primary training data for all three crops",
    },
    "tomato_village": {
        "source": "github",
        "ref": "https://github.com/mamta-joshi-gehlot/Tomato-Village",
        "root_in_archive": "Variant-a(Multiclass Classification)",
        "n_classes": 8,
        "n_images": 4526,
        "license": "see repository / Gehlot et al. 2023",
        "role": "real-world tomato data; adds pest and nutrient-deficiency classes",
    },
    "potato_pld": {
        "source": "kaggle",
        # NOTE: the slug 'potato-disease-leaf-dataset' in the original brief
        # does not exist (404). The real dataset has a 'pld' suffix.
        "ref": "rizwan123456789/potato-disease-leaf-datasetpld",
        "archive": "pld.zip",
        "bytes": 38922037,
        "license": "ODbL (Database), Database Contents License",
        "root_in_archive": "PLD_3_Classes_256",
        "n_classes": 3,
        "n_images": 4072,
        "role": "independent potato data; supplies the healthy-potato images "
                "PlantVillage lacks (1,020 vs 152)",
    },
    "agml_potato": {
        "source": "huggingface",
        "ref": "Project-AgML/potato_leaf_disease_classification",
        "n_classes": 7,
        "n_images": 3076,
        "license": "CC BY-4.0",
        # Labels are pathogen groups, not named diseases -> different taxonomic
        # level from PlantVillage. Only Phytopthora and Healthy map cleanly.
        "role": "held-out real-world evaluation only; NOT merged into training",
        "excluded_from_training": True,
        "exclusion_reason": "pathogen-level labels (Fungi/Bacteria/Virus/...) "
                            "cannot be reliably mapped onto named diseases",
    },
    "grape_hf": {
        "source": "huggingface",
        "ref": "adamkatchee/grape-leaf-disease-augmented-dataset",
        "n_files": 18054,
        "n_unique_sources": 4062,
        "license": "CC BY-NC-SA 4.0",
        "role": "EXCLUDED",
        "excluded_from_training": True,
        "exclusion_reason": "augmented copies of PlantVillage grape (4,062 unique "
                            "sources == PlantVillage grape count exactly); 1,344 "
                            "source images appear in both its train and test dirs",
    },
    "plantdoc": {
        "source": "github",
        "ref": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "n_classes_train": 28,
        "n_classes_test": 27,
        "license": "CC BY 4.0",
        "role": "real-world generalization evaluation; only matching crops used",
    },
}

# Classes excluded outright, with the measured reason.
EXCLUDED_CLASSES = {
    ("plantdoc", "Tomato two spotted spider mites leaf"):
        "only 2 train / 0 test images - too few to train or evaluate",
}

# Display-name normalisation for labels that are misspelled or inconsistently
# cased upstream. Keys are the raw on-disk names; values are what users see.
LABEL_DISPLAY_FIXES = {
    "Pottassium Deficiency": "Potassium Deficiency",   # upstream typo
    "grape leaf": "Grape (healthy)",
    "grape leaf black rot": "Grape Black Rot",
}

# Condition categories. Tomato-Village contributes a pest and three nutrient
# disorders, which are NOT diseases - the UI must say "condition", not "disease".
CONDITION_KINDS = {
    "disease": "infectious disease (fungal, bacterial, viral, oomycete)",
    "pest": "pest damage",
    "deficiency": "nutrient deficiency",
    "healthy": "no condition detected",
}
