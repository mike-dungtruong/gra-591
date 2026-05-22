# CLAUDE.md

This file mirrors `AGENTS.md` for Claude Code. Keep both files aligned when
project structure, commands, or artifact paths change.

## Commands

```bash
# Text-guided model
python train.py --config configs/isic2017.yaml
python train.py --config configs/isic2017.yaml --resume auto
python evaluate.py --config configs/isic2017.yaml --ckpt runs/textswinumamba_isic2017_bert_base/best.pth

# No-text baselines
python train_swin_umamba.py --config configs/isic2017_swin_umamba.yaml
python train_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml
python evaluate_swin_umamba.py --config configs/isic2017_swin_umamba.yaml --ckpt runs/swin_umamba_isic2017/best.pth
python evaluate_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml --ckpt runs/swin_umamba_d_isic2017/best.pth
```

Prepare text features:

```bash
python scripts/verify_caption_coverage.py --isic_root <isic2017> --captions <captions.jsonl>
python scripts/precompute_text_features.py --captions <captions.jsonl> --out cache/text_features.pt
```

Verify the environment:

```bash
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
```

## Architecture

`TextSwinUMambaD` lives in `src/models/text_swin_umamba_d.py`.

- Encoder: `VSSMEncoder` from `src/models/swin_umamba_d.py`.
- Decoder: `TextUNetResDecoder`, a Swin-UMamba-D decoder with TGCM injected at
  each upsampling stage.
- Text branch: `FrozenBertTextEncoder` from `src/models/text_encoder.py`.
- TGCM: `src/models/tgcm.py`, using pooled BERT features to gate parallel
  depthwise convolutions over image features.

Deep supervision uses `DiceBCEWithDeepSupervision` in `src/utils/losses.py`.
Metric code uses `first_scale()` from `src/utils/metrics.py` to evaluate the
highest-resolution output.

## Configuration And Artifacts

Main config keys in `configs/isic2017.yaml`:

- `data.isic_root`
- `data.captions_jsonl`
- `data.text_features_cache`
- `model.pretrained_ckpt`
- `output.base_dir`

Training writes directly to `runs/<run_name>/`: `last.pth`, `best.pth`,
`config.yaml`, `training_log.txt`, `history.csv`, `progress.png`, and optional
`tb/`.

Do not commit datasets, checkpoints, pretrained weights, TensorBoard logs,
caption caches, text-feature caches, generated outputs, or `OPENAI_API_KEY`.

## Key Files

| File | Role |
| --- | --- |
| `train.py` | TextSwinUMambaD training loop |
| `evaluate.py` | TextSwinUMambaD evaluation |
| `configs/isic2017.yaml` | Main text-guided config |
| `src/models/text_swin_umamba_d.py` | Full text-guided model |
| `src/models/tgcm.py` | Text-Gated Channel Module |
| `src/models/swin_umamba_d.py` | Upstream Swin-UMamba-D code; preserve license context |
| `src/data/isic_dataset.py` | Dataset builder with text modes |
| `src/utils/checkpoint.py` | Atomic checkpoint save/load |

Local-only folders may include `caption_gen/`, `notebooks/`, `docs/`, `specs/`,
`papers/`, `outputs/`, `cache/`, `checkpoints/`, and `runs/`.
