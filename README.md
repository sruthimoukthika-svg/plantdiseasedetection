# Plant Disease Detection AI

An AI web application that identifies tomato, potato and grape leaves and
predicts their condition. Upload a leaf photograph; the system validates the
image, identifies the plant, predicts the condition with a confidence score,
and **rejects anything it does not support rather than guessing**.

Built on real trained weights — there are no hard-coded or sample predictions
anywhere in the scanning path, and the API refuses to start without a trained
checkpoint.

---

## Measured performance

On 4,956 held-out test images, never used for training or model selection:

| Metric | Value |
|---|---|
| End-to-end accuracy (22 classes) | **94.57%** |
| End-to-end macro F1 | 92.74% |
| Species identification accuracy | 99.35% |
| Inference | 15.9 ms mean, batch 1, CPU |

### The headline number is misleading on its own

Most training data is standardised laboratory imagery. Real field photographs
are much harder:

| Image source | End-to-end accuracy |
|---|---|
| PlantVillage (laboratory) | 96.60% |
| Potato PLD | 99.00% |
| **Tomato-Village (real field photos)** | **79.80%** |

If you photograph a leaf outdoors on a phone, **79.8% predicts your experience,
not 94.6%**. The site states this on its About page rather than advertising the
higher figure.

### Rejection of unsupported images

Measured false-accept rate (lower is better), at the shipped operating point:

| Category | False-accept |
|---|---|
| Random non-plant photographs (people, animals, buildings, food, objects, vehicles) | **0.00%** |
| Unsupported species, field photographs | 0.62% |
| **Unsupported leaf species never seen in training** | **6.56%** |

Rejecting unfamiliar *leaf species* is the weakest part of the system. Random
objects are turned away reliably; an unfamiliar plant is not always.

The operating point is tuned to accept ~90% of genuine supported leaves. The
full curve is in `evaluation_results/efficientnet_b0_rejection_operating_points.json`:

| Target accept rate | Unseen-species false-accept |
|---|---|
| 0.80 | 1.95% |
| 0.90 | **6.56%** ← shipped |
| 0.95 | 12.59% |
| 0.99 | 27.44% |

0.90 was chosen deliberately: halving the false-accept rate is worth turning
away 5% more genuine leaves. The cost is uneven — `tomato/early_blight`, the
weakest class, is accepted only ~80% of the time.

---

## Architecture

```
frontend/   Next.js 15 + TypeScript + Tailwind  (Home, Scanner, Plants, About)
backend/    FastAPI inference service
training/   dataset inspection, preparation, training, evaluation, calibration
datasets/   prepared corpus and OOD sets        (gitignored)
checkpoints/ trained weights                     (gitignored)
evaluation_results/ measured metrics
```

**Hierarchical classification.** A species gate decides tomato / potato /
grape / **unsupported**, then one condition head per crop predicts the
condition using only that crop's classes. This makes it structurally
impossible to suggest a grape disease for a potato leaf.

**The backbone is frozen.** With no GPU available, features are cached once and
the linear heads train in minutes rather than hours. Pass `--finetune` to
unfreeze for a GPU run.

**Three independent rejection layers**, because a closed-set classifier returns
some class for every input and a high softmax score is not evidence an image is
supported:

1. Model-free checks — sharpness (Laplacian variance) and vegetation fraction
2. The gate's explicit `unsupported` class, trained by outlier exposure
3. Mahalanobis distance to class-conditional Gaussians + free energy, calibrated
   on validation data

Max-softmax is deliberately never used as the rejection signal.

---

## Setup

Requires Python 3.13 and Node 22.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r backend/requirements.txt
```

```bash
cd frontend && npm install
```

### Datasets

Datasets are **not** committed. Two need a Kaggle API key
(`~/.kaggle/kaggle.json`):

```bash
mkdir -p ~/datasets_raw && cd ~/datasets_raw
curl -L -o plantvillage.zip "https://www.kaggle.com/api/v1/datasets/download/abdallahalidev/plantvillage-dataset"
curl -L -o pld.zip "https://www.kaggle.com/api/v1/datasets/download/rizwan123456789/potato-disease-leaf-datasetpld"
```

```bash
git clone --filter=blob:none --sparse --depth 1 https://github.com/mamta-joshi-gehlot/Tomato-Village ~/datasets_raw/tv_images
```

Note: the commonly-cited slug `rizwan123456789/potato-disease-leaf-dataset`
does **not** exist (404). The real dataset has a `pld` suffix.

### Train

```bash
python3 training/inspect_datasets.py
```

```bash
python3 training/prepare_data.py
```

```bash
python3 training/extract_features.py --backbone efficientnet_b0 --aug 3
```

```bash
python3 training/train_model.py --backbone efficientnet_b0
```

```bash
python3 training/evaluate_model.py --backbone efficientnet_b0
```

```bash
python3 training/build_exposure_sets.py --per-species 1200
```

```bash
python3 training/train_rejector.py --backbone efficientnet_b0 --target-tpr 0.90
```

On 4 CPU cores: feature extraction ~30 min, head training ~4 min.

### Run

```bash
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend && NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

The site is then at `http://localhost:3000`.

### Test

```bash
python3 tests/test_api_e2e.py 60
```

---

## Configuration

All configuration is by environment variable; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `PDAI_BACKBONE` | `efficientnet_b0` | Which trained model to serve |
| `PDAI_MODEL_DIR` | `./checkpoints` | Where weights are loaded from |
| `PDAI_MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `PDAI_ALLOWED_ORIGINS` | `localhost:3000,127.0.0.1:3000` | CORS origins — **both are needed**; they are distinct origins to a browser |
| `PDAI_PERSIST_UPLOADS` | `false` | Uploads are processed in memory and discarded |

---

## Datasets

| Dataset | Role |
|---|---|
| [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | Primary training data, all three crops |
| [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) | Real-world tomato; source of pest and deficiency classes |
| [Potato Disease Leaf (PLD)](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld) | Independent potato data; healthy-leaf images |
| [AgML Potato](https://huggingface.co/datasets/Project-AgML/potato_leaf_disease_classification) | Evaluation only — pathogen-group labels |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | Evaluation only — generalisation and rejection |
| [Grape augmented](https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset) | **Excluded** — see below |

Each dataset was inspected before use; see
[`training/reports/dataset_report.md`](training/reports/dataset_report.md).
Findings that changed the design:

- **The grape dataset was excluded.** All 18,054 of its filenames carry
  PlantVillage's UUID prefix and reduce to 4,062 unique source images —
  exactly PlantVillage's grape count. 33% of those sources appear in **both**
  its train and test directories.
- **AgML's labels are pathogen groups, not diseases.** `Phytopthora` maps to
  late blight, but `Fungi` is a superset *containing* early blight and cannot
  be mapped onto it. Held out for evaluation instead.
- **PLD is genuinely independent** of PlantVillage — 0 exact and 0 near
  perceptual-hash matches across all 4,072 × 2,152 potato images — and supplies
  1,020 healthy potato images against PlantVillage's 152.
- **Four of Tomato-Village's eight classes are not diseases** (one pest, three
  nutrient deficiencies), so the system reports a *condition*, with a `kind`
  field, rather than a disease.

### Leakage control

Splits are assigned per **duplicate group**, not per image. Near-duplicates are
found by perceptual hash and then **verified at pixel level** — dHash alone
false-positived badly here, since PlantVillage leaves sit on uniform
backgrounds: 1,725 hash candidates reduced to 115 real duplicates, rejecting
1,610 collisions. Verified: **0 duplicate groups span splits.**

---

## Limitations

- Substantially weaker on field photographs than laboratory images (79.8% vs 96.6%).
- Rejecting leaf species never seen in training is the weakest component (6.56% false-accept).
- Only tomato, potato and grape are supported.
- `tomato/potassium_deficiency` has 11 test images; its per-class score is not statistically meaningful.
- Confidence scores are uncalibrated and read optimistically. Confidence is the
  model's score for a class, **not** the probability the plant has the condition.
- The model sees one leaf in isolation — not the whole plant, conditions or history.

**This is an experimental research tool, not a diagnosis service.** Do not make
treatment, spraying or crop-destruction decisions from its output.
