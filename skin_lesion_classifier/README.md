# Dermalyze Training Pipeline

ML training and evaluation pipeline for skin lesion classification using EfficientNet, EfficientNetV2, ResNeSt, and ConvNeXt architectures on the HAM10000 dataset.

> ⚠️ **DISCLAIMER**: Educational/research purposes only. Not for medical diagnosis. Consult healthcare professionals for medical advice.

## Architecture

This is the **training component** of the Dermalyze project:

- **Training Pipeline** (this directory): Model training, evaluation, hyperparameter tuning
- **Inference Service**: [`../inference_service/`](../inference_service/README.md) - Standalone API for production deployment
- **Frontend Application**: [`../frontend/`](../frontend/README.md) - Web UI for end users

After training models here, copy the best checkpoint to `../inference_service/model/` for deployment.

## Key Features

- **EfficientNet-B0 to B7** and **EfficientNetV2-S/M/L** with transfer learning
- **Classification heads**: MLP (512→256→7) with BatchNorm & Dropout (0.35)
- **Augmentation**: Mixup, CutMix, TTA, domain-specific transforms
- **Training**: Mixed precision (AMP), ModelEMA, cosine annealing, early stopping
- **Inference**: Ensemble prediction, TTA, uncertainty quantification
- **Data**: Lesion-aware splitting (prevents leakage), weighted sampling for imbalance
- **Loss**: Cross-Entropy, Focal Loss, Label Smoothing
- **Evaluation**: Per-class metrics, ROC curves, confusion matrix, calibration
- **Utilities**: Training visualization, fit analysis, benchmarking

## Classes

| Label | Description |
|-------|-------------|
| akiec | Actinic keratoses/intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

## Quick Start

```bash
# 1. Install PyTorch (auto-detect CUDA or CPU)
bash scripts/install_pytorch.sh

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Prepare HAM10000 data (download HAM10000 dataset first)
# Training set (with metadata)
python src/prepare_data.py \
    --data-dir data/HAM10000_Training \
    --include-metadata

# Validation set
python src/prepare_data.py \
    --data-dir data/HAM10000_Val

# Test set
python src/prepare_data.py \
    --data-dir data/HAM10000_Test

# 4. Train
python src/train.py --config config.yaml

# 5. Evaluate
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000_Training/images

# 6. Predict
python src/inference.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --image path/to/image.jpg
```

## Exporting Models for Inference

After training, copy your best checkpoint to the inference service:

```bash
# Copy checkpoint to inference service
cp outputs/run_xxx/checkpoint_best.pt ../inference_service/model/

# Or copy an ensemble/kfold checkpoint
cp outputs/kfold_sweep_1/fold_0/checkpoint_best.pt ../inference_service/model/
```

Then follow [`../inference_service/README.md`](../inference_service/README.md) to run the API server for frontend integration.

Windows (PowerShell):

```powershell
# 1. Install PyTorch (auto-detect CUDA or CPU)
.\scripts\install_pytorch.ps1

# 2. Install remaining dependencies
pip install -r requirements.txt
```

### PyTorch CUDA/ROCm/CPU Channel Selection

Installers:
- macOS/Linux: `scripts/install_pytorch.sh`
- Windows: `scripts/install_pytorch.ps1`

Auto-selection (highest compatible channel):
- CUDA `>=13.0` → `cu130`
- CUDA `>=12.8` → `cu128`
- CUDA `>=12.6` → `cu126`
- ROCm `>=7.1` → `rocm7.1` (Linux Bash installer)
- otherwise → `cpu`

Manual override examples:

```bash
TORCH_CHANNEL=cu130 bash scripts/install_pytorch.sh
TORCH_CHANNEL=cu128 bash scripts/install_pytorch.sh
TORCH_CHANNEL=cu126 bash scripts/install_pytorch.sh
TORCH_CHANNEL=rocm7.1 bash scripts/install_pytorch.sh
TORCH_CHANNEL=cpu bash scripts/install_pytorch.sh
```

```powershell
$env:TORCH_CHANNEL='cu130'  # or cu128 / cu126 / cpu
.\scripts\install_pytorch.ps1
```

## Dataset

Download [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and organize:
```
data/HAM10000_Training/
├── images/
│   └── *.jpg
├── ground_truth.csv (one-hot encoded)
└── metadata.csv (age_approx, sex, anatom_site_general)

data/HAM10000_Val/
├── images/
│   └── *.jpg
└── ground_truth.csv

data/HAM10000_Test/
├── images/
│   └── *.jpg
└── ground_truth.csv
```

The `prepare_data.py` script will convert the one-hot encoded ground_truth.csv to single-label format and merge with metadata for the training set.

## Training

**Basic:**
```bash
python src/train.py --config config.yaml
```

**Resume:**
```bash
python src/train.py --resume outputs/run_xxx/checkpoint_latest.pt
```

**ConvNeXt variant (same training entrypoint):**
```bash
python src/train.py --config config_conv.yaml
```

`config_conv.yaml` should set:
```yaml
model:
    backbone: convnext_tiny
```

**Ensemble (3 models with different seeds):**
```bash
python scripts/train_ensemble.py
```

Each ensemble model is automatically assigned a different fold index (`data.kfold.fold_index`) so models train on different train/val folds. You can shift the sequence with `--start-fold`.

**5-fold sweep (folds 0-4) + automatic aggregation:**
```bash
python scripts/run_kfold_sweep.py --config config.yaml
```
This creates `outputs/kfold_sweep_*/kfold_command_plan.sh` and aggregates fold evaluation metrics into `outputs/kfold_sweep_*/kfold_summary.json`.

**Run folds in parallel (faster on multi-GPU / multi-node setups):**
```bash
python scripts/run_kfold_sweep.py --config config.yaml --max-parallel-folds 5
```
Each fold still runs `train -> evaluate -> evaluate_tta` in order, but different folds run concurrently. Logs are written per fold at `outputs/kfold_sweep_*/fold_<k>/fold_run.log`.
The CLI shows a live fold-level progress bar while concurrent folds are running.

**Optuna hyperparameter tuning (wired to `src/train.py`):**
```bash
python scripts/tune_optuna.py \
    --config config.yaml \
    --n-trials 40 \
    --objective-metric macro_recall_f1_mean
```
This runs full training per trial using trial-specific YAML configs and stores artifacts under `outputs/optuna_*`.

To stream actual training/evaluation logs from each trial to terminal:
```bash
python scripts/tune_optuna.py \
    --config config.yaml \
    --n-trials 5 \
    --objective-metric macro_recall_f1_mean \
    --verbose-subprocess
```

Useful options:
- `--storage sqlite:///outputs/optuna/study.db` (resume-able study DB)
- `--study-name ham10000_optuna`
- `--timeout 43200` (12 hours)
- `--objective-metric macro_recall|macro_f1|weighted_recall|weighted_f1|macro_recall_f1_mean`
- `--show-progress-bar/--no-show-progress-bar` (trial-level progress bar)
- `--verbose-subprocess` (show live `train.py` + `evaluate.py` output)

Recommended objective for HAM10000 with imbalance controls: `macro_recall_f1_mean`.
After tuning, inspect `outputs/optuna_*/optuna_summary.json` and retrain best config at your full epoch budget.

**Key config options:**
- `model.backbone`: `efficientnet_b0`, `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l`, or `convnext_tiny` (all use `src/train.py`)
- `model.dropout_rate`: 0.35 (regularization)
- `training.batch_size`: 32
- `training.epochs`: 40
- `training.lr`: 0.0003
- `training.use_weighted_sampling`: true (class balance)
- `loss.type`: cross_entropy, focal, or label_smoothing
- `data.use_stratified_group_kfold`: true/false (fold-based training split)
- `data.kfold.n_splits`, `data.kfold.fold_index`, `data.kfold.group_column`

See `config.md` for a complete parameter reference, valid options, and defaults.


## Evaluation

**Basic:**
```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000_Training/images
```
Supports checkpoints trained with any backbone (EfficientNet B0-B7, EfficientNetV2 S/M/L, ConvNeXt-Tiny, ResNeSt-101, SE-ResNeXt-101) via `src/train.py`, including mixed-architecture ensembles.

**Config-first evaluation (recommended):**
`src/evaluate.py` now reads TTA defaults from `config.yaml` (via `evaluation.tta`) so you don't need to pass TTA flags every run.

```yaml
evaluation:
  tta:
    use_tta: true
    mode: medium
    aggregation: geometric_mean
    use_clahe_tta: true
    clahe_clip_limit: 2.0
    clahe_grid_size: 8
```

Then run:
```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000_Training/images
```

CLI flags still override config for one-off runs (for example `--tta-mode full`).

**With TTA (Test-Time Augmentation):**
```bash
python src/evaluate.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --test-csv outputs/run_xxx/test_split.csv \
    --images-dir data/HAM10000_Training/images \
    --use-tta --tta-mode medium
```
*TTA modes: `light` (4 branches: original + flips), `medium` (8 branches: light + 90°/180°/270° rotations + center zoom-crop), `full` (12 branches: medium + 4 corner zoom-crops).*

**Ensemble:**
```bash
python src/evaluate.py \
    --checkpoint \
        outputs/model_1/checkpoint_best.pt \
        outputs/model_2/checkpoint_best.pt \
        outputs/model_3/checkpoint_best.pt \
    --test-csv outputs/model_1/test_split.csv \
    --images-dir data/HAM10000_Training/images
```
*Default uses `weighted_mean` aggregation with automatic weight computation from validation accuracy*

**Ensemble with custom weights:**
```bash
python src/evaluate.py \
    --checkpoint model1.pt model2.pt model3.pt \
    --test-csv test_split.csv \
    --images-dir data/HAM10000_Training/images \
    --ensemble-weights 0.5 0.3 0.2
```

**Ensemble aggregation methods:**
- `weighted_mean` (default): Automatically weights by validation accuracy from checkpoints
- `mean`: Uniform averaging (all models equal weight)
- `geometric_mean`: Geometric average (robust to outliers)

**Ensemble with TTA:**
```bash
python src/evaluate.py \
    --checkpoint \
        outputs/model_1/checkpoint_best.pt \
        outputs/model_2/checkpoint_best.pt \
        outputs/model_3/checkpoint_best.pt \
    --test-csv outputs/model_1/test_split.csv \
    --images-dir data/HAM10000_Training/images \
    --use-tta \
    --tta-mode medium \
    --ensemble-aggregation weighted_mean
```

**Outputs:** `evaluation_metrics.json`, `confusion_matrix.png`, `roc_curves.png`, `per_class_metrics.png`, `calibration_curve.png`

## Inference

**Python API:**
```python
from src.inference import SkinLesionPredictor

predictor = SkinLesionPredictor("outputs/run_xxx/checkpoint_best.pt")
result = predictor.predict("path/to/image.jpg")

print(f"{result['predicted_class']}: {result['confidence']:.2%}")
```

**With TTA:**
```python
result = predictor.predict_with_tta("image.jpg", mode="medium")
```

**Ensemble:**
```python
from src.inference import load_ensemble_predictor

ensemble = load_ensemble_predictor([
    "outputs/run_1/checkpoint_best.pt",
    "outputs/run_2/checkpoint_best.pt"
])
result = ensemble.predict("image.jpg")
```

**CLI:**
```bash
python src/inference.py \
    --checkpoint outputs/run_xxx/checkpoint_best.pt \
    --image path/to/image.jpg \
    --use-tta --tta-mode medium
```

## Utilities

```bash
# Visualize training curves
python scripts/visualize_training.py --run outputs/run_xxx

# Check for overfitting/underfitting
python scripts/check_fit.py outputs/run_xxx/training_history.json

# Benchmark performance
python scripts/benchmark.py --num-batches 20
```

## Architecture

- **Backbone**: EfficientNet (B0-B7), EfficientNetV2 (S/M/L), ResNeSt-101, SE-ResNeXt-101, or ConvNeXt-Tiny (ImageNet pretrained)
- **Classifier**: MLP (512→256→7) with BatchNorm & Dropout (0.3)
- **Loss**: Cross-Entropy / Focal / Label Smoothing with class weights
- **Optimizer**: AdamW (lr=0.00025, weight_decay=0.02)
- **Scheduler**: Cosine annealing with warm restarts (T_0=20)
- **Augmentation**: Flip, rotate, color jitter, blur, affine, cutout
- **Class imbalance**: Weighted sampling + class-weighted loss

## Ensemble Weighting

When using `--ensemble-aggregation weighted_mean` without `--ensemble-weights`:
- **Automatic weight computation** from validation accuracy in checkpoints
- Models with higher validation accuracy get higher weights
- Falls back to uniform weights if validation metrics are unavailable
- Example: Model A (92% acc) gets weight 0.52, Model B (88% acc) gets weight 0.48

## Expected Performance

| Metric | Score |
|--------|-------|
| Accuracy | 75-85% |
| Macro F1 | 65-75% |
| ROC-AUC (macro) | 85-95% |

*Note: Class-specific performance varies due to data imbalance in HAM10000 dataset*

## Ethical Considerations

- ⚠️ **Not for clinical use** - Educational tool only, not validated for diagnosis
- **Dataset biases**: Limited skin tone diversity
- **Limited generalization**: May not work on different equipment, lighting, or demographics
- **Probabilistic outputs**: High confidence ≠ correctness
- **Transparency required**: Always disclose AI involvement in healthcare contexts

**This system should augment, never replace, professional medical judgment.**


## References

1. **HAM10000**: Tschandl et al. (2018). [DOI:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
2. **EfficientNetV2**: Tan & Le (2021). [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
3. **Focal Loss**: Lin et al. (2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

## License

Educational/research purposes. Check [HAM10000 license](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) for data usage terms.