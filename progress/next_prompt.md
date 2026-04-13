# Next Prompt

Read `AGENTS.md`, `handoff.md`, and the refreshed Stage 5 artifacts, then continue from the new full-dataset baseline of record.

Use this Python command pattern for any shell execution in this repository:
`python ...`

Current result of record:
- run name: `stage5_refresh_resnet18_ft_linear_lr2e4`
- backbone: pretrained `resnet18`
- strategy: full fine-tuning
- head: linear
- optimizer: `AdamW`
- LR: `2e-4`
- epochs: `1`
- mean 5-fold validation accuracy: `0.9218`

Mission:
1. refresh downstream outputs so they match the new baseline
2. do not reopen a broad architecture search
3. only consider one narrow follow-up accuracy pass after the reporting pipeline is current again

Priority order:
- regenerate Stage 6 visualizations from the refreshed Stage 5 checkpoints
- regenerate Stage 7 feature maps from the refreshed best checkpoint
- refresh the Stage 9 notebook so it reflects the refreshed full-dataset baseline
- only then consider a tightly scoped follow-up such as:
  - a 2-epoch rerun of the selected `resnet18` strategy
  - or a single direct `resnet50` full 5-fold confirmation if runtime is acceptable

Important facts:
- the canonical dataset remains `data/processed/rank14_from_local_raw/`
- the refreshed strategy screen already ran and its summary artifacts live under `outputs/logs/refresh_full_dataset_screen_*`
- the refreshed Stage 5 artifacts already exist under:
  - `outputs/folds/stage5_refresh_resnet18_ft_linear_lr2e4_*`
  - `outputs/logs/stage5_refresh_resnet18_ft_linear_lr2e4_*`
  - `outputs/checkpoints/stage5_refresh_resnet18_ft_linear_lr2e4_fold_*`
- the environment is CPU-only
- `num_workers > 0` currently fails with `PermissionError: [WinError 5] Access is denied`

Rules:
- keep the final project requirement of real 5-fold CV intact
- do not compare refreshed results to stale subset-era metrics without labeling the dataset refresh
- do not resume packaging until the refreshed notebook and visualization outputs are aligned with the new baseline
