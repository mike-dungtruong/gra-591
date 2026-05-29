# TextSwinUMamba

Text-guided medical image segmentation for ISIC 2017. This project adds a
Text-Gated Channel Module (TGCM) to the Swin-UMamba-D decoder so dermoscopy
captions can guide lesion segmentation through frozen BERT features.

The Swin-UMamba encoder stays compatible with VMamba-Tiny ImageNet pretrained
weights. TGCM consumes the pooled BERT embedding for each image caption at every
decoder upsampling stage.

## Repo Layout

```text
TextSwinUMamba/
├── configs/
│   ├── isic2017.yaml                 # TextSwinUMambaD config
│   ├── isic2017_text_swin_umamba.yaml # TextSwinUMamba CNN-decoder config
│   ├── isic2017_swin_umamba.yaml     # no-text SwinUMamba baseline
│   └── isic2017_swin_umamba_d.yaml   # no-text SwinUMamba-D baseline
├── src/
│   ├── models/                       # model definitions
│   ├── data/                         # ISIC dataset, transforms, captions
│   └── utils/                        # losses, metrics, checkpointing, logging
├── scripts/
│   ├── precompute_text_features.py
│   └── verify_caption_coverage.py
├── training/                         # training entrypoints by model family
│   ├── train_text_swin_umamba_d.py
│   ├── train_text_swin_umamba.py
│   ├── train_swin_umamba.py
│   └── train_swin_umamba_d.py
├── evaluation/                       # evaluation entrypoints by model family
│   ├── evaluate_text_swin_umamba_d.py
│   ├── evaluate_text_swin_umamba.py
│   ├── evaluate_swin_umamba.py
│   └── evaluate_swin_umamba_d.py
├── THIRD_PARTY_LICENSES/
│   └── Swin-UMamba-LICENSE
└── notebooks/
```

Important model files are `src/models/text_swin_umamba_d.py`,
`src/models/tgcm.py`, `src/models/text_encoder.py`, and
`src/models/swin_umamba_d.py`. The upstream Swin-UMamba code is redistributed
under Apache 2.0; preserve the license context when editing it.

Local-only research assets may exist in `caption_gen/`, `notebooks/`, `docs/`,
`specs/`, `papers/`, `outputs/`, `cache/`, `checkpoints/`, and `runs/`.

## What Lives In Git

Tracked source should stay code-focused:

| Goes in git | Stays local or in Drive |
| --- | --- |
| `src/`, `training/`, `evaluation/`, `configs/isic2017*.yaml` | `runs/<run_name>/` checkpoints and logs |
| `README.md`, `AGENTS.md`, `CLAUDE.md`, `.gitignore` | `cache/text_features.pt` |
| `requirements.txt`, `environment.yml` | ISIC images, masks, captions, pretrained weights |
| `THIRD_PARTY_LICENSES/` | generated outputs |

Do not commit `OPENAI_API_KEY`, datasets, checkpoints, generated captions with
sensitive metadata, or text-feature caches.

## Setup

Create the canonical conda environment:

```bash
conda env create -f environment.yml
conda activate textswinumamba
```

For pip-only setup, install the CUDA 11.8 PyTorch wheels first:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify the Mamba kernel after installing dependencies:

```bash
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
```

## Data Preparation

Check caption coverage:

```bash
python scripts/verify_caption_coverage.py \
  --isic_root /path/to/isic2017 \
  --captions /path/to/captions.jsonl
```

Precompute BERT features for faster training:

```bash
python scripts/precompute_text_features.py \
  --captions /path/to/captions.jsonl \
  --out cache/text_features.pt
```

The canonical text config keys are in `configs/isic2017.yaml`:
`data.isic_root`, `data.captions_jsonl`, `data.text_features_cache`,
`model.pretrained_ckpt`, and `output.base_dir`.

## Training

Train TextSwinUMambaD:

```bash
python training/train_text_swin_umamba_d.py --config configs/isic2017.yaml
python training/train_text_swin_umamba_d.py --config configs/isic2017.yaml --resume auto
```

Train no-text baselines:

```bash
python training/train_swin_umamba.py --config configs/isic2017_swin_umamba.yaml
python training/train_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml
```

Train the text-guided CNN-decoder variant:

```bash
python training/train_text_swin_umamba.py --config configs/isic2017_text_swin_umamba.yaml
python training/train_text_swin_umamba.py --config configs/isic2018_text_swin_umamba.yaml
```

Training writes directly to `runs/<run_name>/` by default:
`last.pth`, `best.pth`, `config.yaml`, `training_log.txt`, `history.csv`,
and `progress.png`.

## Evaluation

Evaluate TextSwinUMambaD:

```bash
python evaluation/evaluate_text_swin_umamba_d.py \
  --config configs/isic2017.yaml \
  --ckpt runs/textswinumamba_isic2017_bert_base/best.pth
```

Evaluate baselines:

```bash
python evaluation/evaluate_swin_umamba.py \
  --config configs/isic2017_swin_umamba.yaml \
  --ckpt runs/swin_umamba_isic2017/best.pth

python evaluation/evaluate_swin_umamba_d.py \
  --config configs/isic2017_swin_umamba_d.yaml \
  --ckpt runs/swin_umamba_d_isic2017/best.pth
```

Evaluate the text-guided CNN-decoder variant:

```bash
python evaluation/evaluate_text_swin_umamba.py \
  --config configs/isic2017_text_swin_umamba.yaml \
  --ckpt runs/text_swin_umamba_isic2017/best.pth

python evaluation/evaluate_text_swin_umamba.py \
  --config configs/isic2018_text_swin_umamba.yaml \
  --ckpt runs/text_swin_umamba_isic2018/best.pth
```

## ISIC 2017 Evaluation Snapshot

Current `best.pth` results on the 650-case validation split:

| Model | Epoch | mIoU(%)↑ | DSC(%)↑ | Acc(%)↑ | Spe(%)↑ | Sen(%)↑ |
|-------|------:|---------:|--------:|--------:|--------:|--------:|
| SwinUMamba | 42 | 80.56 | 89.23 | 96.42 | 97.96 | 88.73 |
| SwinUMambaD | 41 | 80.68 | 89.31 | 96.45 | 98.03 | 88.61 |
| TextSwinUMambaD | 40 | 82.20 | 90.23 | 96.74 | 98.08 | 90.06 |

## Colab Workflow

1. Push the tracked repo to GitHub.
2. Put datasets, captions, pretrained weights, cache, and run outputs in Drive.
3. Override paths in `configs/isic2017.yaml` or notebook cells as needed.
4. Train with `--resume auto`; rerunning after disconnect resumes from
   `runs/<run_name>/last.pth`.

## Agent Docs

`AGENTS.md` is the canonical cross-agent guide. `CLAUDE.md` mirrors the same
facts for Claude Code. Keep both aligned with the actual `src/` layout and CLI
flags when changing project structure.

## Acknowledgements

The Swin-UMamba and VMamba components come from Liu et al. 2024. The upstream
license is preserved in `THIRD_PARTY_LICENSES/Swin-UMamba-LICENSE`. TGCM is
inspired by ViTexNet.
