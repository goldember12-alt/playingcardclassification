# Next Prompt

Read `AGENTS.md`, `handoff.md`, and the current canonical full-dataset artifacts, then continue from the current result of record.

Use the active project interpreter for any shell execution in this repository rather than assuming a specific launcher alias.

Current result of record:
- run name: `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
- backbone: pretrained `resnet50`
- strategy: full fine-tuning
- head: linear
- optimizer: `AdamW`
- LR: `1e-4`
- epochs: `3`
- mean 5-fold validation accuracy: `0.9619813205 +/- 0.0067484664`

Mission:
1. keep public-facing docs and generated artifacts aligned with the current canonical run
2. do not reopen a broad architecture search
3. if more modeling work is needed, keep it to one bounded follow-up experiment

Priority order:
- verify documentation and notebook text against the current dataset and result-of-record artifacts
- refresh any downstream summaries if a stronger canonical run is added
- only then consider a tightly scoped follow-up such as:
  - one bounded higher-accuracy rerun of the current `resnet50` recipe
  - or one direct comparison against a single alternative architecture if runtime is acceptable

Important facts:
- the canonical dataset remains `data/processed/rank14_from_local_raw/`
- the current result-of-record artifacts already exist under:
  - `outputs/folds/stage8_refresh_resnet50_ft_linear_lr1e4_e3_*`
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_*`
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_*`
- the environment is CPU-only
- `num_workers > 0` currently fails with `PermissionError: [WinError 5] Access is denied`

Rules:
- keep the final project requirement of real 5-fold CV intact
- do not compare canonical full-dataset results to subset-era metrics without labeling the dataset refresh
- keep the notebook/report aligned with the strongest canonical full-dataset run
