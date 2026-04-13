# playingcardclassification

`playingcardclassification` is a standalone computer-vision project for classifying playing-card images into 14 rank classes using transfer learning with pretrained CNNs.

The repository is organized around a reproducible workflow:
- preprocess and validate the dataset
- train pretrained image classifiers with real 5-fold cross-validation
- visualize quantitative results
- inspect intermediate feature maps
- publish a notebook-backed project report with visible outputs

## Project scope
The current pipeline supports:
- pretrained image-classification backbones
- bottlenecking and fine-tuning workflows
- all 14 target classes
- stratified 5-fold cross-validation
- saved metrics, plots, confusion matrices, and prediction galleries
- feature-map visualizations for face cards and number cards

## Dataset
Source dataset:

[Cards Image Dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

This project works at the rank level rather than the full 53-card identity level. The target classes are:
- `ace`
- `two`
- `three`
- `four`
- `five`
- `six`
- `seven`
- `eight`
- `nine`
- `ten`
- `jack`
- `queen`
- `king`
- `joker`

## Local dataset audit
The repository contains local raw data under `data/raw/`, but the raw split tree is not treated as the direct training source.

Observed local state:
- `data/raw/train/` exists
- `data/raw/test/` exists
- `data/raw/valid/` does not exist
- `data/raw/cards.csv` exists and describes a 53-label deck dataset that can be normalized into 14 rank targets
- `data/raw/14card types-14-(200 X 200)-94.61.h5` and `data/raw/53cards-53-(200 X 200)-100.00.h5` are Keras artifacts, not image directories for the PyTorch pipeline

Current reconciliation result:
- the folder tree on disk contains only `1,866` of the `8,155` image paths referenced by `cards.csv`
- the discovered folder labels are 53 suit-specific card identities, not the 14 rank targets used by this project
- the metadata schema can still be normalized to the 14 rank classes because CSV rank `xxx` maps to `joker`

## Canonical local dataset source
The repo derives a complete local 14-rank dataset from the images that do exist locally and uses that as the canonical training source:

- dataset root: `data/processed/rank14_from_local_raw/`
- layout: flat class directories from `ace` through `joker`
- total images: `1,866`
- derivation method: existing raw `train/` and `test/` images are grouped by rank, collapsing suit-specific labels into rank labels
- storage mode: hard links back to the original raw image files
- manifest: `data/processed/rank14_from_local_raw/manifest.csv`
- summary: `data/processed/rank14_from_local_raw/summary.json`

This supports strict 5-fold stratified cross-validation because every class has at least five examples. The main caveat is severe class imbalance, especially `joker` with only five images.

## Current result of record
The strongest run currently documented in the repo is:
- run name: `stage8_refresh_resnet50_ft_linear_lr1e4_e3`
- backbone: pretrained `resnet50`
- strategy: full fine-tuning
- folds: `5`
- epochs per fold: `3`
- mean validation accuracy: `0.9619813205 +/- 0.0067484664`

## Repository structure
- `src/`: reusable data, model, training, evaluation, and utility modules
- `configs/`: experiment configuration files
- `scripts/`: notebook and workflow helpers
- `notebooks/`: executed notebook artifacts
- `outputs/`: checkpoints, logs, folds, visualizations, and feature maps
- `reports/`: saved figures, notes, and tables
- `progress/`: lightweight internal status notes

## Environment setup
Use any Python 3.11+ virtual environment for the project. A simple local setup looks like:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are continuing from an existing local environment created for this repo, using that interpreter is also fine.

## Project workflow
The implementation path in this repository is:

1. scaffold the repo
2. implement dataset loading and fold logic
3. build pretrained baselines
4. train and validate on one fold
5. scale to full 5-fold cross-validation
6. generate visualizations and confusion matrices
7. generate feature maps
8. run narrowly scoped improvement passes when needed
9. assemble the final notebook/report artifact
10. package shareable project outputs

## Operating docs
- `AGENTS.md`: internal workflow guide for continuing the project in staged increments
- `handoff.md`: latest project status, result of record, and recommended next step
- `repo_skeleton.txt`: reference layout for the scaffolded repository

## Success criteria
The repo is in a good state when it produces:
- a modular, reproducible training pipeline
- real 5-fold cross-validation results across all 14 classes
- clear visualizations and saved artifacts
- feature-map examples for representative face and number cards
- an executed notebook/report with visible outputs
- a clean, shareable release package

## Notes
The priority is clarity, reproducibility, and verifiable results. A clean, well-documented transfer-learning pipeline is more valuable here than a large, opaque experiment sweep.
