# TextSwinUMamba

Text-guided medical image segmentation. Adds a Text-Gated Channel Module (TGCM) into
the decoder of [Swin-UMamba†](https://github.com/JiarunLiu/Swin-UMamba) for skin lesion
segmentation on ISIC 2017, with GPT-4o-generated dermoscopy captions as text input.

The encoder is unchanged from Swin-UMamba† so VMamba-Tiny ImageNet pretrained weights
load cleanly. The decoder consumes pooled BERT features of the per-image caption at
every upsampling stage via TGCM (depthwise-conv fusion gated by text, ViTexNet-style).

## Repo layout

```
TextSwinUMamba/
├── configs/isic2017.yaml          # all hyperparameters
├── models/
│   ├── text_encoder.py            # frozen BERT-base wrapper
│   ├── tgcm.py                    # Text-Gated Channel Module (our contribution)
│   ├── text_swin_umamba_d.py      # TextSwinUMambaD = encoder + TGCM-decoder
│   └── swin_umamba_d.py           # upstream Swin-UMamba net code (Apache 2.0)
├── data/                          # ISIC dataset, transforms, caption loader
├── utils/                         # losses, metrics, checkpoint, misc
├── scripts/
│   ├── precompute_text_features.py
│   └── verify_caption_coverage.py
├── THIRD_PARTY_LICENSES/
│   └── Swin-UMamba-LICENSE        # Apache 2.0, preserved as required
├── notebooks/colab_train.ipynb    # Colab entry: clone from git, Drive for artifacts
├── train.py
└── evaluate.py
```

## What lives in git vs. Drive

This repo (code only) lives in **git / GitHub**. Everything large or per-run lives in
**Google Drive** and is gitignored:

| Goes in git                 | Goes in Drive                          |
|-----------------------------|----------------------------------------|
| `.py`, `.yaml`, `.ipynb`    | `runs/<run_name>/` (checkpoints, logs) |
| `README.md`, `.gitignore`   | `cache/text_features.pt`               |
| `requirements.txt`          | `datasets/isic2017/`                   |
|                             | `captions/captions.jsonl`              |
|                             | `pretrained/vmamba_tiny_e292.pth`      |
|                             | TensorBoard event files                |

`models/swin_umamba_d.py` is upstream Swin-UMamba code (Apache 2.0), redistributed
under the terms of that license; the full license text is preserved under
`THIRD_PARTY_LICENSES/Swin-UMamba-LICENSE` and the file's docstring lists the
modifications we made to detach it from nnUNet.

## Local setup (one time)

```bash
# 1. Clone this repo
git clone <your-remote-url> TextSwinUMamba
cd TextSwinUMamba

# 2. Install deps
pip install -r requirements.txt

# 3. (Optional) sanity check captions coverage
python scripts/verify_caption_coverage.py \
    --isic_root /path/to/isic2017 \
    --captions  /path/to/captions.jsonl

# 4. Precompute BERT text features (one time, ~1 min on GPU)
python scripts/precompute_text_features.py \
    --captions /path/to/captions.jsonl \
    --out      cache/text_features.pt
```

## Initialize git and push to remote

```bash
cd TextSwinUMamba
git init
git add .
git commit -m "Initial commit: TextSwinUMambaD scaffold"
git branch -M main
git remote add origin git@github.com:<you>/TextSwinUMamba.git
git push -u origin main
```

The `.gitignore` already excludes datasets, checkpoints, large weights, and the
generated `models/swin_umamba_d.py`.

## Training

```bash
# Local
python train.py --config configs/isic2017.yaml

# Resume (auto-detects last.pth in run_dir, or pass --resume <path>)
python train.py --config configs/isic2017.yaml --resume auto

# Colab: open notebooks/colab_train.ipynb
```

## Colab workflow

1. Push this repo to GitHub.
2. Put dataset + captions + pretrained weights into Drive at
   `MyDrive/TextSwinUMamba/{datasets,captions,pretrained}/`.
3. Open `notebooks/colab_train.ipynb`. It will:
   - Mount Drive (for artifacts).
   - `git clone` this repo from GitHub into `/content/`.
   - Precompute text features into Drive (cached across sessions).
   - Train with `--resume auto`, writing checkpoints + TensorBoard logs to Drive.
4. If Colab disconnects, re-run the notebook from top to bottom — everything is
   idempotent and training resumes from `last.pth`.

## Acknowledgements

The encoder, decoder, and VSS / SS2D blocks in `models/swin_umamba_d.py` come from
[Liu et al. 2024 — Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) with the
nnUNet integration removed. The upstream code is Apache 2.0 licensed and the full
license is preserved at `THIRD_PARTY_LICENSES/Swin-UMamba-LICENSE`. TGCM is inspired
by ViTexNet (Bhardwaj et al., MICCAI 2025).
