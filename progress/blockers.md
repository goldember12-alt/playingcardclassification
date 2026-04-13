# Blockers

The main project blockers are no longer dataset availability or sub-90% performance.

Current open constraints:
- use the active project interpreter
- the canonical data source remains `data/processed/rank14_from_local_raw/`
- the environment is CPU-only:
  - `torch.cuda.is_available() == False`
- Windows multiprocessing data loading currently fails here:
  - `num_workers > 0` raised `PermissionError: [WinError 5] Access is denied`
  - practical implication: runs currently use `num_workers=0`

Current procedural caution:
- if any new canonical run is added, downstream summaries must be refreshed from that run rather than left pointing at older artifacts
- packaging or sharing should use the current record run:
  - `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
