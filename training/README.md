# Training Entrypoints

Run these scripts from the repository root.

| Model key | Script | Model |
|---|---|---|
| `text` | `training/train_text_swin_umamba_d.py` | TextSwinUMambaD: Mamba encoder + Mamba decoder + decoder TGCM text guidance |
| `text_swin_umamba` | `training/train_text_swin_umamba.py` | TextSwinUMamba: Mamba encoder + CNN decoder + decoder TGCM text guidance |
| `swin_umamba` | `training/train_swin_umamba.py` | SwinUMamba no-text baseline with CNN decoder |
| `swin_umamba_d` | `training/train_swin_umamba_d.py` | SwinUMamba-D no-text baseline with Mamba decoder |
| `text_lvit_add` | `training/train_text_swin_umamba_d.py` | TextSwinUMambaD with encoder additive text fusion, selected by config |
| `text_lvit_film` | `training/train_text_swin_umamba_d.py` | TextSwinUMambaD with encoder FiLM text fusion, selected by config |
| `text_hybrid_add` | `training/train_text_swin_umamba_d.py` | TextSwinUMambaD with encoder additive fusion plus decoder TGCM, selected by config |
| `text_hybrid_film` | `training/train_text_swin_umamba_d.py` | TextSwinUMambaD with encoder FiLM fusion plus decoder TGCM, selected by config |

All training scripts write `last.pth`, `best.pth`, `history.csv`, `training_log.txt`, and `progress.png` to `output.base_dir/run_name`.
