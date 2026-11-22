# Stage 1 — Dataset Inspection Report

Generated from the real archives in this environment by
`training/inspect_datasets.py`. Machine-readable companion:
`training/reports/dataset_report.json`.

Every count below was measured, not taken from a dataset name, README or paper.

## Summary

| # | Dataset | Source | Measured size | Verdict |
|---|---|---|---|---|
| 1 | PlantVillage | Kaggle `abdallahalidev/plantvillage-dataset` | 38 classes, 54,305 images × 3 variants | **Primary training data** |
| 2 | Tomato-Village | GitHub `mamta-joshi-gehlot/Tomato-Village` | 8 classes, 4,526 images | **Train** (adds pest + deficiency classes) |
| 3 | Potato PLD | Kaggle `rizwan123456789/potato-disease-leaf-datasetpld` | 3 classes, 4,072 images | **Train** (independent; fixes healthy-potato shortage) |
| 4 | AgML Potato | HF `Project-AgML/potato_leaf_disease_classification` | 7 classes, 3,076 images | **Eval only** (incompatible label taxonomy) |
| 5 | Grape (augmented) | HF `adamkatchee/grape-leaf-disease-augmented-dataset` | 18,054 files / 4,062 sources | **Excluded** (PlantVillage copy + split leakage) |
| 6 | PlantDoc | GitHub `pratikkayal/PlantDoc-Dataset` | 28/27 classes, 2,578 images | **Eval only** (real-world generalization) |

## Corrections to the original brief

1. **Dataset 3's URL is dead.** `rizwan123456789/potato-disease-leaf-dataset`
   returns HTTP 404 (web page) / 403 (API). The real dataset is the same
   owner's **`potato-disease-leaf-datasetpld`** — note the `pld` suffix.
   38,922,037 bytes, ODbL. All work uses the corrected slug.

## 1. PlantVillage

Ships three parallel variants of the same 54,305 images:
`color` (54,305), `grayscale` (54,305), `segmented` (54,306).
**Only `color` is used** — the other two trade 3× the disk for no new
information. The segmented count being one higher indicates a stray extra
file, not extra data.

Of the 38 classes, these map to the three supported crops:

| Crop | Classes | Images |
|---|---|---|
| Tomato | 10 | 18,160 |
| Grape | 4 | 4,062 |
| Potato | 3 | 2,152 |

**Class imbalance.** Worst case is `Potato___healthy` at **152** images
against `Potato___Early_blight` and `Potato___Late_blight` at 1,000 each —
a 6.6:1 ratio inside one crop. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
(5,357) versus `Tomato___Tomato_mosaic_virus` (373) is 14:1. Both need
addressing in Stage 2 via class weighting or resampling.

## 2. Tomato-Village

Variant-a (Multiclass Classification) is the relevant variant; Variant-b
(multilabel) and Variant-c (object detection) are not needed.
Pre-split train/val/test: 3,163 / 902 / 461.

| Class | Train | Val | Test | Kind |
|---|---|---|---|---|
| Early_blight | 347 | 99 | 50 | disease |
| Healthy | 151 | 43 | 22 | healthy |
| Late_blight | 632 | 180 | 92 | disease |
| Leaf Miner | 716 | 204 | 104 | **pest** |
| Magnesium Deficiency | 655 | 187 | 95 | **deficiency** |
| Nitrogen Deficiency | 251 | 72 | 37 | **deficiency** |
| Pottassium Deficiency | 50 | 14 | 8 | **deficiency** |
| Spotted Wilt Virus | 361 | 103 | 53 | disease |

**Finding: four of eight classes are not diseases.** Leaf Miner is pest
damage; the three Deficiency classes are nutrient disorders. None has a
PlantVillage equivalent, so they are additive classes, never merges. The
product must therefore talk about a *condition*, not a *disease* — the UI
copy and API field naming follow from this.

**Finding: `Pottassium Deficiency` is misspelled upstream.** The on-disk
directory name is preserved for provenance; the display label is normalised
to "Potassium Deficiency" in `config.LABEL_DISPLAY_FIXES`.

**Finding: Spotted Wilt Virus must not be merged** with PlantVillage's
`Tomato_Yellow_Leaf_Curl_Virus` or `Tomato_mosaic_virus`. These are three
distinct pathogens.

## 3. Potato PLD (corrected slug)

3 classes, 4,072 images, 256 px, all JPG, pre-split
Training / Validation / Testing.

| Class | Training | Validation | Testing |
|---|---|---|---|
| Early_Blight | 1,303 | 163 | 162 |
| Healthy | 816 | 102 | 102 |
| Late_Blight | 1,132 | 151 | 141 |

**Why this dataset matters:** it carries **1,020 healthy potato images**
against PlantVillage's 152. It is the single biggest fix available for the
worst class imbalance in the project.

**Independence verified.** Because its filenames are sequentially renamed
(`Early_Blight_1.jpg`) and both datasets are 256 px, byte-level comparison
alone could miss a re-encoded copy. Perceptual hashing (dHash, 64-bit) of
**all** 4,072 PLD images against **all** 2,152 PlantVillage potato images
found:

- exact dHash matches: **0**
- near matches (Hamming ≤ 4): **0**

So PLD is genuinely independent data and safe to combine.

**Its own split is slightly dirty:** 20 internal duplicate images, and 4
dHashes appearing in more than one of its own Training/Validation/Testing
directories. Stage 2 must dedupe and re-split rather than trust the shipped
split.

## 4. AgML Potato — excluded from training

Declared: 3,076 examples, train split only, no official test split, 754 MB
in two parquet shards. Labels:

`Bacteria (569) · Fungi (748) · Healthy (201) · Nematode (68) · Pest (611) · Phytopthora (347) · Virus (532)`

**Finding: this is a different taxonomic level.** These are pathogen
groups, not named diseases:

| AgML label | Maps to | Verdict |
|---|---|---|
| `Phytopthora` | Late blight (*Phytophthora infestans*) | clean 1:1 |
| `Healthy` | healthy | clean 1:1 |
| `Fungi` | — | **cannot map**: a superset *containing* Early blight plus other fungal diseases |
| `Bacteria`, `Nematode`, `Pest`, `Virus` | — | no PlantVillage potato equivalent |

Mapping `Fungi` → Early blight would be exactly the biologically-wrong merge
the brief forbids. The dataset is also captured in an uncontrolled
environment (Shabrina et al. 2024), which makes it valuable as a **held-out
real-world evaluation set** for `Healthy` and `Late blight`. It is not
merged into the training label space. `Nematode` at 68 images would be
unusable for training regardless.

## 5. Grape augmented — excluded entirely

18,054 image files, in 4 classes (Black Rot, ESCA, Healthy, Leaf Blight),
shipped as train (14,444) / test (3,610).

**Finding: this is PlantVillage grape, augmented.** Evidence:

- **100%** of the 18,054 filenames carry PlantVillage's `UUID___` prefix
  (e.g. `00090b0f-…___FAM_B.Rot 0376_aug1.png`)
- they reduce to **4,062 unique source images**, ~4.44 augmented copies each
- 4,062 is *exactly* PlantVillage's grape total
  (1,180 Black_rot + 1,383 Esca + 1,076 Leaf_blight + 423 healthy)
- its 4 classes correspond 1:1 to PlantVillage's 4 grape classes
- a sampled file is 600×800, upscaled from PlantVillage's native 256×256

**Finding: its shipped split leaks badly.** **1,344 of 4,062 source images
(33%) appear in both its train and test directories** as different
augmentations. Any accuracy measured on that split is inflated and
meaningless.

Combining it with PlantVillage grape would double-count the same
photographs. **Decision:** exclude it; use PlantVillage grape as the source
of truth and augment at training time, where augmentation is applied after
splitting and cannot leak.

## 6. PlantDoc — evaluation only

2,578 images; 28 train classes but only **27** test classes, across 13
species. Real-world backgrounds and lighting, which is its value.

Classes matching the supported crops:

| Crop | Class | Train | Test |
|---|---|---|---|
| Tomato | Early blight leaf | 79 | 9 |
| Tomato | Septoria leaf spot | 140 | 11 |
| Tomato | leaf bacterial spot | 101 | 9 |
| Tomato | leaf late blight | 101 | 10 |
| Tomato | leaf mosaic virus | 44 | 10 |
| Tomato | leaf yellow virus | 70 | 6 |
| Tomato | mold leaf | 85 | 6 |
| Tomato | leaf (healthy) | 55 | 8 |
| Tomato | two spotted spider mites leaf | **2** | **0** |
| Potato | leaf early blight | 109 | 8 |
| Potato | leaf late blight | 97 | 8 |
| Grape | leaf black rot | 56 | 8 |
| Grape | leaf (healthy) | 57 | 12 |

**Finding: `Tomato two spotted spider mites leaf` has 2 train / 0 test
images.** It is the class missing from the test set and is excluded — see
`config.EXCLUDED_CLASSES`.

**Finding: no healthy-potato class.** PlantDoc has only the two potato
blights, so healthy potato must come from PlantVillage and PLD.

**Finding: inconsistent naming.** Lowercase (`grape leaf black rot`) and
underscored (`Bell_pepper leaf spot`) names sit alongside capitalised ones,
so normalisation is required before any cross-dataset merge.

## Consequences for Stage 2

1. Use PlantVillage `color` only; ignore `grayscale` and `segmented`.
2. Exclude dataset 5 entirely; grape comes from PlantVillage.
3. Hold AgML and PlantDoc out of training as real-world eval sets.
4. Combine PlantVillage + PLD for potato; dedupe PLD's 20 internal
   duplicates and re-split by perceptual hash.
5. Keep Leaf Miner and the three deficiency classes as distinct
   non-disease conditions; carry a `kind` field (disease/pest/deficiency/
   healthy) through to the API.
6. Never merge Spotted Wilt Virus, Yellow Leaf Curl Virus and mosaic virus.
7. Address imbalance: potato healthy 152 vs 1,000; tomato mosaic 373 vs
   yellow curl 5,357.
8. Record the originating dataset for every image.
