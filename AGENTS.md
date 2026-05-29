# Repository Guidelines

## Project Structure

This repo implements TextSwinUMambaD for text-guided ISIC 2017 lesion
segmentation. Core entry points live in `training/` and `evaluation/`.

Code lives under `src/`:

- `src/models/text_swin_umamba_d.py` assembles TextSwinUMambaD.
- `src/models/tgcm.py` contains the Text-Gated Channel Module.
- `src/models/text_encoder.py` wraps frozen BERT.
- `src/models/swin_umamba_d.py` contains upstream Swin-UMamba-D code.
- `src/data/` contains dataset loading, transforms, and caption utilities.
- `src/utils/` contains losses, metrics, checkpointing, logging, and misc helpers.

Configs are in `configs/`. One-off utilities are in `scripts/`. Optional
local-only tooling may exist in `caption_gen/`, `notebooks/`, `docs/`, `specs/`,
`papers/`, and `outputs/`.

## Commands

Create the canonical environment:

```bash
conda env create -f environment.yml
conda activate textswinumamba
```

For pip-only setup, install the CUDA 11.8 PyTorch wheels first, then run
`pip install -r requirements.txt`. Verify the pinned Mamba kernel with:

```bash
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
```

Train and resume TextSwinUMambaD:

```bash
python training/train_text_swin_umamba_d.py --config configs/isic2017.yaml
python training/train_text_swin_umamba_d.py --config configs/isic2017.yaml --resume auto
```

Evaluate TextSwinUMambaD:

```bash
python evaluation/evaluate_text_swin_umamba_d.py --config configs/isic2017.yaml --ckpt runs/textswinumamba_isic2017_bert_base/best.pth
```

Train or evaluate no-text baselines with `training/train_swin_umamba.py`,
`training/train_swin_umamba_d.py`, `evaluation/evaluate_swin_umamba.py`, and
`evaluation/evaluate_swin_umamba_d.py` using their matching
`configs/isic2017_swin_umamba*.yaml` files.

Precompute caption embeddings:

```bash
python scripts/precompute_text_features.py --captions <captions.jsonl> --out cache/text_features.pt
```

## Configuration

Keep behavior config-driven. The main path keys in `configs/isic2017.yaml` are:

- `data.isic_root`
- `data.captions_jsonl`
- `data.text_features_cache`
- `model.pretrained_ckpt`
- `output.base_dir`

Training writes directly to `runs/<run_name>/`: `last.pth`, `best.pth`,
`config.yaml`, `training_log.txt`, `history.csv`, and `progress.png`.

## Coding Style

Use Python 3.10, 4-space indentation, type hints where they clarify interfaces,
and concise module docstrings for entry points. Use `snake_case` for functions,
variables, and files; `PascalCase` for classes; and short lowercase package
names.

Preserve the upstream license context in `src/models/swin_umamba_d.py` and avoid
unrelated refactors there.

## Testing Guidelines

There is no formal unit-test suite yet. Before training changes, run relevant
smoke checks:

```bash
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
python scripts/verify_caption_coverage.py --isic_root <isic2017> --captions <captions.jsonl>
```

Run a short training or evaluation pass only when the dataset, captions,
pretrained weights, and suitable compute are available.

For caption generation changes, run:

```bash
python caption_gen/generate.py --config configs/caption_gen.yaml --dry-run
python caption_gen/evaluate.py --captions outputs/captions/captions.jsonl
```

## Security And Artifacts

Do not commit `OPENAI_API_KEY`, generated captions with sensitive metadata,
datasets, checkpoints, `cache/text_features.pt`, pretrained weights, or run
outputs. Prefer local or Drive paths configured through YAML and document any
path overrides in PR notes.
