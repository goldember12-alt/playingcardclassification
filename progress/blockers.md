# Blockers

The main accuracy blocker is no longer dataset availability or sub-90% performance.

Current open constraints:
- use the repo venv interpreter:
  - `C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe`
- the canonical data source remains `data/processed/rank14_from_local_raw/`
- the environment is CPU-only:
  - `torch.cuda.is_available() == False`
- Windows multiprocessing data loading currently fails here:
  - `num_workers > 0` raised `PermissionError: [WinError 5] Access is denied`
  - practical implication: runs currently use `num_workers=0`

Current procedural blocker:
- downstream reporting is now stale relative to the refreshed Stage 5 baseline
- specifically, Stage 6, Stage 7, and the Stage 9 notebook still need to be regenerated from:
  - `stage5_refresh_resnet18_ft_linear_lr2e4`

Packaging remains intentionally blocked until:
- refreshed visualizations are regenerated
- refreshed feature maps are regenerated
- the notebook is updated to reflect the refreshed baseline of record
