# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training
```bash
# Fresh run
python train.py --config configs/isic2017.yaml

# Resume from last checkpoint (auto-detects runs/<run_name>/checkpoints/last.pth)
python train.py --config configs/isic2017.yaml --resume auto

# Resume from specific checkpoint
python train.py --config configs/isic2017.yaml --resume runs/my_run/checkpoints/last.pth
```

### Evaluation
```bash
python evaluate.py --config configs/isic2017.yaml --ckpt runs/<run_name>/checkpoints/best.pth
```

### Data preparation (one-time)
```bash
# Precompute BERT features for all captions (recommended before training)
python scripts/precompute_text_features.py \
  --captions <path/to/captions.jsonl> \
  --out cache/text_features.pt

# Verify caption coverage over dataset images
python scripts/verify_caption_coverage.py \
  --captions <path/to/captions.jsonl> \
  --images <path/to/isic2017/images/>
```

## Architecture

**TextSwinUMambaD** (`models/text_swin_umamba_d.py`) is the main model:
- **Encoder**: `VSSMEncoder` from `models/swin_umamba_d.py` (upstream Swin-UMamba, Apache 2.0, frozen for the first N epochs per config)
- **Decoder**: `TextUNetResDecoder` — a modified UNet residual decoder that injects a `TGCM` at each upsampling stage
- **Text branch**: `FrozenBertTextEncoder` (`models/text_encoder.py`) wraps BERT-base-uncased; always frozen during training; outputs `(pooled [B,768], tokens [B,L,768])`

**TGCM** (`models/tgcm.py`) is the core novel module:
- Takes image features `(B, H, W, C)` and text pooled embedding `(B, 768)`
- Projects text → K scalar weights → K parallel depthwise 1D convolutions on flattened spatial dimension
- Two iterative refinement passes (ViTexNet-style)
- Learnable residual mix via `beta` parameter

**Deep supervision**: The decoder outputs a list of feature maps at 4 scales; `DiceBCEWithDeepSupervision` in `utils/losses.py` applies weighted loss at each scale. `utils/metrics.py:first_scale()` extracts the highest-resolution output for metric computation.

### Text input modes (`data/isic_dataset.py`)
- `"features"`: Loads precomputed `.pt` cache (dict of `image_id → {pooled, tokens, attention_mask}`). Fast; recommended for Colab.
- `"tokens"`: Tokenizes and runs BERT forward on every batch. Simpler but slower.

Set via `data.text_mode` in the YAML config.

## Data & Artifacts

**Git-tracked**: source code, `configs/`, `notebooks/`, `requirements.txt`

**Not git-tracked** (store in Google Drive or local paths outside the repo):
- `datasets/isic2017/` — raw images and masks
- `captions/captions.jsonl` — GPT-4o-generated dermoscopy captions (one JSON object per line: `image_id`, `caption`)
- `cache/text_features.pt` — precomputed BERT embeddings
- `pretrained/vmamba_tiny_e292.pth` — VMamba-Tiny ImageNet pretrained weights (Swin-UMamba encoder init)
- `runs/<run_name>/` — checkpoints (`last.pth`, `best.pth`) and TensorBoard logs

Paths for all of the above are configured in `configs/isic2017.yaml` under `data.root`, `data.captions`, `data.text_features_cache`, `model.pretrained_encoder`, and `training.run_dir`.

## Key Files

| File | Role |
|------|------|
| `train.py` | Main training loop; AMP, cosine LR + warmup, encoder freeze schedule, wall-clock budget |
| `evaluate.py` | Standalone eval; reports Dice and IoU |
| `configs/isic2017.yaml` | All hyperparameters and paths |
| `models/text_swin_umamba_d.py` | Full model assembly |
| `models/tgcm.py` | Text-Gated Channel Module (novel contribution) |
| `models/swin_umamba_d.py` | Upstream Swin-UMamba (do not modify; Apache 2.0) |
| `models/text_encoder.py` | Frozen BERT wrapper |
| `utils/checkpoint.py` | Atomic save/load with full RNG state for deterministic resume |
| `notebooks/colab_train.ipynb` | Colab entry point (mounts Drive, installs deps, trains) |
