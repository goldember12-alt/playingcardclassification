# handoff.md

## Project status
Current status: the bounded `resnet50` escalation requested for the canonical full dataset is complete, downstream artifacts have been refreshed, and the notebook has been rebuilt with visible outputs.

Follow-up status on 2026-04-13: a one-fold `EfficientNet-B0` feasibility probe is also complete, and it does not currently justify a full 5-fold escalation.

The current result of record is now:
- canonical dataset: `data/processed/rank14_from_local_raw/`
- run name: `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
- pretrained `resnet50`
- full fine-tuning
- simple linear head
- `AdamW`
- LR `1e-4`
- weight decay `1e-4`
- no augmentation
- no balancing
- plateau scheduler
- 3 epochs per fold
- strict real 5-fold CV
- seed `42`
- `num_workers=0`
- CPU execution
- mean 5-fold validation accuracy: `0.9619813205 +/- 0.0067484664`

Environment note: use the active project interpreter for shell commands in this repository.

## What was completed
- Read `AGENTS.md` and `handoff.md`, then continued from the prior result of record:
  - `stage8_refresh_resnet18_ft_linear_lr2e4_e3`
- Ran the required bounded experiment on the canonical full dataset:
  - `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
- Preserved the requested narrow search discipline:
  - no augmentation
  - no class balancing
  - no larger classifier head
  - no broader architecture search
- Recovered the long CPU run in a bounded way:
  - the first full-run attempt timed out at 6 hours after folds `0` to `3` completed and fold `4` had already written a best checkpoint
  - reran only fold `4`
  - rebuilt the aggregate Stage 5-style summary from the saved fold artifacts
- Compared the `resnet50` result only against canonical full-dataset runs:
  - `stage5_refresh_resnet18_ft_linear_lr2e4`
  - `stage8_refresh_resnet18_ft_linear_lr2e4_e2`
  - `stage8_refresh_resnet18_ft_linear_lr2e4_e3`
- Because the `resnet50` run became the new result of record, regenerated:
  - Stage 6 visualizations
  - Stage 7 feature maps
  - Stage 9 notebook with visible outputs
- Updated discovery logic so refreshed notebook and visualization builders can follow the strongest canonical full-dataset run rather than assuming the old refreshed `stage5` baseline.
- Ran one additional bounded feasibility check after the `resnet50` result-of-record discussion:
  - fold `0` only
  - `efficientnet_b0`
  - full fine-tuning
  - linear head
  - `AdamW`
  - LR `1e-4`
  - weight decay `1e-4`
  - no augmentation
  - no balancing
  - plateau scheduler
  - 3 epochs
  - CPU
  - `num_workers=0`
- Compared that fold-0 `EfficientNet-B0` probe directly against fold `0` from the recent canonical `resnet18` and `resnet50` runs.

## Exact code and notebook changes
Modified files:
- `src/evaluation/visualizations.py`
  - Stage 6 default summary discovery now selects the strongest canonical full-dataset aggregate summary instead of only `stage5_*`
  - Stage 6 notes now refer generically to the completed cross-validation run rather than a hardcoded baseline
- `src/models/feature_maps.py`
  - Stage 7 default summary discovery now selects the strongest canonical full-dataset aggregate summary instead of only `stage5_*`
  - Stage 7 notes now refer generically to saved cross-validation checkpoints
- `scripts/build_stage9_notebook.py`
  - notebook build now discovers the strongest canonical full-dataset run dynamically
  - comparison discovery now supports cases where the current record is the improved Stage 8 run
  - notebook narrative no longer hardcodes refreshed `ResNet18` as the record model
  - conclusion cell now reports the correct record status when the improved run is the current winner
- the executed notebook artifact under `notebooks/`
  - regenerated and executed with the new `resnet50` result of record and refreshed downstream artifacts
- the HTML export under `notebooks/exports/`
  - regenerated from the executed notebook
- `handoff.md`
  - rewritten for the current state

## Exact experiment config
Run name:
- `stage8_refresh_resnet50_ft_linear_lr1e4_e3`

Dataset:
- root: `data/processed/rank14_from_local_raw/`
- classes: all 14 rank labels

Training configuration:
- model: `resnet50`
- pretrained backbone: `true`
- freeze backbone: `false`
- unfreeze_from: `None`
- classifier head: linear only
- classifier hidden dim: `None`
- classifier dropout: `0.0`
- optimizer: `adamw`
- batch size: `64`
- epochs: `3`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `plateau`
- scheduler plateau factor: `0.5`
- scheduler plateau patience: `1`
- scheduler min LR: `1e-6`
- label smoothing: `0.0`
- augmentation: `false`
- class weighting: `none`
- sampling strategy: `none`
- folds: `5`
- seed: `42`
- device: `cpu`
- num_workers: `0`

## Bounded resnet50 escalation results
Aggregate metrics:
- mean validation accuracy: `0.9619813205`
- validation accuracy std: `0.0067484664`
- mean validation loss: `0.1269737159`
- mean train accuracy at best epoch: `0.9863870925`
- mean train loss at best epoch: `0.0573080570`
- mean elapsed time per fold: `4645.31s`

Per-fold validation accuracy:
- fold `0`: `0.9717964439`
- fold `1`: `0.9638258737`
- fold `2`: `0.9540159411`
- fold `3`: `0.9625996321`
- fold `4`: `0.9576687117`

Per-fold runtime note:
- every fold selected epoch `3` as the best epoch
- best fold is now fold `0`

## Canonical full-dataset comparison
Compared runs:
- `stage5_refresh_resnet18_ft_linear_lr2e4`
  - mean val accuracy: `0.9217560080`
  - delta vs resnet50 run: `+0.0402253125`
- `stage8_refresh_resnet18_ft_linear_lr2e4_e2`
  - mean val accuracy: `0.9478788654`
  - delta vs resnet50 run: `+0.0141024551`
- `stage8_refresh_resnet18_ft_linear_lr2e4_e3`
  - mean val accuracy: `0.9578127010`
  - delta vs resnet50 run: `+0.0041686195`

Direct fold-by-fold comparison vs previous result of record `stage8_refresh_resnet18_ft_linear_lr2e4_e3`:
- fold `0`: `+0.0141017781`
- fold `1`: `+0.0079705702`
- fold `2`: `+0.0006131208`
- fold `3`: `+0.0030656039`
- fold `4`: `-0.0049079755`

Interpretation:
- the `resnet50` escalation is a modest but real improvement over the prior `resnet18` record
- it improves mean 5-fold validation accuracy by about `0.42` percentage points
- it improves `4` of `5` folds versus the prior record
- it is materially slower than the `resnet18` recipe, but it is now the strongest canonical full-dataset run in the repository
- it still does not reach the stretch target of `0.97+` mean validation accuracy

## Artifacts
Primary run artifacts:
- aggregate summary JSON:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_aggregate_summary.json`
- aggregate summary Markdown:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_aggregate_summary.md`
- per-fold CSV:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_per_fold_results.csv`
- summary table CSV:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_summary_table.csv`
- fold assignments:
  - `outputs/folds/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_assignments.csv`
- fold overview:
  - `outputs/folds/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_overview.csv`
- validation class counts:
  - `outputs/folds/stage8_refresh_resnet50_ft_linear_lr1e4_e3_5fold_seed42_validation_class_counts.csv`
- comparison JSON:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_comparison.json`
- comparison Markdown:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_comparison.md`
- comparison CSV:
  - `outputs/logs/stage8_refresh_resnet50_ft_linear_lr1e4_e3_per_fold_comparison.csv`
- checkpoints:
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_00_best.pt`
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_01_best.pt`
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_02_best.pt`
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_03_best.pt`
  - `outputs/checkpoints/stage8_refresh_resnet50_ft_linear_lr1e4_e3_fold_04_best.pt`

Refreshed Stage 6 artifacts:
- folder:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/`
- summary JSON:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/stage6_summary.json`
- summary Markdown:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/stage6_summary.md`
- curves:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/curves/training_validation_curves.png`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/curves/fold_summary_chart.png`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/curves/fold_summary_table.csv`
- confusion:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/confusion/aggregate_confusion_matrix_counts.png`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/confusion/aggregate_confusion_matrix_normalized.png`
- predictions:
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/predictions/aggregate_validation_predictions.csv`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/predictions/aggregate_misclassifications.csv`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/predictions/prediction_gallery.png`
  - `outputs/visualizations/stage8_refresh_resnet50_ft_linear_lr1e4_e3/predictions/misclassification_gallery.png`

Stage 6 summary:
- validation predictions aggregated: `8154`
- aggregated validation accuracy from saved predictions: `0.9619818494`

Refreshed Stage 7 artifacts:
- folder:
  - `outputs/feature_maps/stage8_refresh_resnet50_ft_linear_lr1e4_e3/best_fold_00/`
- summary JSON:
  - `outputs/feature_maps/stage8_refresh_resnet50_ft_linear_lr1e4_e3/best_fold_00/stage7_summary.json`
- summary Markdown:
  - `outputs/feature_maps/stage8_refresh_resnet50_ft_linear_lr1e4_e3/best_fold_00/stage7_summary.md`
- selected examples CSV:
  - `outputs/feature_maps/stage8_refresh_resnet50_ft_linear_lr1e4_e3/best_fold_00/selected_examples.csv`
- overview PNG:
  - `outputs/feature_maps/stage8_refresh_resnet50_ft_linear_lr1e4_e3/best_fold_00/feature_map_overview.png`

Stage 7 summary:
- selected checkpoint fold: `0`
- checkpoint validation accuracy: `0.9717964439`
- selected face-card classes:
  - `jack`
  - `queen`
- selected number-card classes:
  - `seven`
  - `five`
- hooked layers:
  - `backbone.layer3`
  - `backbone.layer4`

Refreshed Stage 9 notebook artifacts:
- notebook:
  - executed project notebook under `notebooks/`
- HTML export:
  - rendered export under `notebooks/exports/`

## Runtime notes
- The first end-to-end `resnet50` run attempt hit the command wall-clock timeout at `21600s` (`6` hours).
- That timeout happened after folds `0` to `3` had completed and fold `4` had already written its best checkpoint.
- To keep the search bounded, I did not restart the full 5-fold run.
- Instead, I reran only fold `4` and rebuilt the aggregate run summary from the saved fold artifacts.
- The fold-4 recovery step took `4405.8s`.
- Mean elapsed time per fold for the saved run is `4645.31s`, which is about `2.88x` the prior `resnet18` 3-epoch record.
- Notebook execution through direct `nbconvert --execute` is still blocked by the Windows secure-write permission failure.
- Successful notebook execution and HTML export were completed through an in-process `ExecutePreprocessor` call with a temporary monkeypatch to Jupyter's Windows permission helper.
- The notebook execution also emitted non-blocking Windows/ZMQ event-loop warnings plus IPython permission warnings after the monkeypatch.
- The one-fold `EfficientNet-B0` probe used the same clean protocol as the current `resnet50` record except for the backbone swap.
- That probe took `1986.48s` total for fold `0` across `3` epochs:
  - epoch `1`: `648.52s`
  - epoch `2`: `634.17s`
  - epoch `3`: `703.78s`
- Relative CPU runtime on fold `0`:
  - `EfficientNet-B0` was about `1.23x` slower than `resnet18`
  - `EfficientNet-B0` was about `0.42x` the runtime of `resnet50`

## One-fold EfficientNet-B0 probe
Probe run:
- `stage8_probe_effb0_ft_linear_lr1e4_e3`

Exact probe config:
- fold: `0`
- dataset: `data/processed/rank14_from_local_raw/`
- model: `efficientnet_b0`
- pretrained backbone: `true`
- freeze backbone: `false`
- classifier head: linear only
- optimizer: `adamw`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `plateau`
- scheduler plateau factor: `0.5`
- scheduler plateau patience: `1`
- scheduler min LR: `1e-6`
- epochs: `3`
- batch size: `64`
- augmentation: `false`
- class weighting: `none`
- sampling strategy: `none`
- seed: `42`
- device: `cpu`
- num_workers: `0`

Fold-0 result:
- best epoch: `3`
- validation accuracy: `0.9190680564`
- total elapsed seconds: `1986.48`

Artifacts:
- metrics CSV:
  - `outputs/logs/stage8_probe_effb0_ft_linear_lr1e4_e3_fold_00_metrics.csv`
- summary JSON:
  - `outputs/logs/stage8_probe_effb0_ft_linear_lr1e4_e3_fold_00_summary.json`
- checkpoint:
  - `outputs/checkpoints/stage8_probe_effb0_ft_linear_lr1e4_e3_fold_00_best.pt`

Fold-0 comparison against recent canonical runs:
- `stage8_probe_effb0_ft_linear_lr1e4_e3`
  - model: `efficientnet_b0`
  - validation accuracy: `0.9190680564`
  - elapsed seconds: `1986.48`
- `stage8_refresh_resnet18_ft_linear_lr2e4_e3`
  - model: `resnet18`
  - validation accuracy: `0.9576946658`
  - elapsed seconds: `1615.57`
- `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
  - model: `resnet50`
  - validation accuracy: `0.9717964439`
  - elapsed seconds: `4721.34`

Interpretation:
- `EfficientNet-B0` is runtime-feasible on CPU under the clean 14-class 5-fold protocol.
- On this first fair fold test, it is not competitive with either `resnet18` or `resnet50` on validation accuracy.
- Because the probe is about `3.86` accuracy points behind `resnet18` fold `0` and about `5.27` points behind `resnet50` fold `0`, a full 5-fold escalation is not recommended.

## Verification status
- Verified the final aggregate summary exists for `stage8_refresh_resnet50_ft_linear_lr1e4_e3`.
- Verified the comparison artifact against `stage5_refresh_resnet18_ft_linear_lr2e4` exists.
- Verified the `resnet50` run exceeds all canonical comparison runs on mean validation accuracy.
- Verified refreshed Stage 6 outputs were rebuilt from the `resnet50` record run.
- Verified refreshed Stage 7 feature maps were rebuilt from the `resnet50` best checkpoint.
- Verified the notebook was regenerated and executed with visible outputs.
- Verified the HTML export was regenerated.
- Verified the notebook now loads:
  - `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
  - `stage8_refresh_resnet50_ft_linear_lr1e4_e3_comparison.json`
- Verified the one-fold `EfficientNet-B0` probe completed and saved its checkpoint, metrics CSV, and summary JSON.

## Blockers or open risks
- The environment remains CPU-only:
  - `torch.cuda.is_available() == False`
- Windows multiprocessing data loading is still blocked:
  - `num_workers > 0` previously raised `PermissionError: [WinError 5] Access is denied`
  - practical implication: runs still use `num_workers=0`
- Vanilla notebook execution through `jupyter nbconvert --execute` is still blocked by the Windows secure-write permission error:
  - `PermissionError: [WinError 5] Access is denied`
- The current record is stronger than the prior `resnet18` result, but still below the stretch target of `0.97+` mean validation accuracy.
- Because the improvement over the prior `resnet18` record is modest and runtime is already high on CPU, further accuracy search is not recommended before packaging unless the project scope explicitly demands more experimentation.
- The one-fold `EfficientNet-B0` result reduces confidence that a new backbone swap will beat the current `resnet50` record under the strict canonical protocol without broader tuning.

## Exact recommended next prompt
Read `AGENTS.md` and `handoff.md`, then continue from the current result of record:
`stage8_refresh_resnet50_ft_linear_lr1e4_e3`.

Use this Python command pattern for shell execution:
`python ...`

Mission:
- stop accuracy search
- freeze `stage8_refresh_resnet50_ft_linear_lr1e4_e3` as the final result of record
- complete Stage 10 release packaging

Required packaging work:
- create `playingcardclassification`
- copy the final notebook with visible outputs
- include the supporting files needed for review and reproducibility
- keep the folder layout zip-ready
- prepare `playingcardclassification.zip`

Do not:
- start a new architecture sweep
- add augmentation or balancing experiments
- replace the current result of record unless you find a concrete artifact integrity problem
