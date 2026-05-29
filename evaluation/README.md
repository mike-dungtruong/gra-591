# Evaluation Entrypoints

Run these scripts from the repository root.

| Model key | Script | Model |
|---|---|---|
| `text` | `evaluation/evaluate_text_swin_umamba_d.py` | TextSwinUMambaD: Mamba encoder + Mamba decoder + decoder TGCM text guidance |
| `text_swin_umamba` | `evaluation/evaluate_text_swin_umamba.py` | TextSwinUMamba: Mamba encoder + CNN decoder + decoder TGCM text guidance |
| `swin_umamba` | `evaluation/evaluate_swin_umamba.py` | SwinUMamba no-text baseline with CNN decoder |
| `swin_umamba_d` | `evaluation/evaluate_swin_umamba_d.py` | SwinUMamba-D no-text baseline with Mamba decoder |
| `text_lvit_add` | `evaluation/evaluate_text_swin_umamba_d.py` | TextSwinUMambaD with encoder additive text fusion, selected by config |
| `text_lvit_film` | `evaluation/evaluate_text_swin_umamba_d.py` | TextSwinUMambaD with encoder FiLM text fusion, selected by config |
| `text_hybrid_add` | `evaluation/evaluate_text_swin_umamba_d.py` | TextSwinUMambaD with encoder additive fusion plus decoder TGCM, selected by config |
| `text_hybrid_film` | `evaluation/evaluate_text_swin_umamba_d.py` | TextSwinUMambaD with encoder FiLM fusion plus decoder TGCM, selected by config |

Evaluation scripts load `--ckpt`, print checkpoint monitor metadata when present, and recompute validation metrics from the configured dataset.
