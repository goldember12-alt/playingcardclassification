# OliviaMLAssignment

This repository is a documentation-first working repo for a machine learning homework assignment on transfer learning for image classification.

## Assignment summary
The assignment is to use a pretrained image classification model and either:

- bottleneck it, or
- fine-tune it

to classify a new image dataset of playing cards.

The dataset contains 14 classes, corresponding to traditional card ranks. The work must include 5-fold validation, result visualizations, and feature-map visualizations from the trained network.

## Dataset
Kaggle dataset referenced by the assignment:

`https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification`

## Local dataset audit
The repository has local raw data under `data/raw/`, but the raw split tree itself is incomplete and is no longer the canonical training source.

Observed local state:
- `data/raw/train/` exists
- `data/raw/test/` exists
- `data/raw/valid/` does not exist
- `data/raw/cards.csv` exists and describes a 53-label deck dataset grouped into 14 rank targets
- `data/raw/14card types-14-(200 X 200)-94.61.h5` and `data/raw/53cards-53-(200 X 200)-100.00.h5` are HDF5/Keras model artifacts, not image folders for the PyTorch pipeline

Current reconciliation result:
- the folder tree on disk contains only 1,866 of the 8,155 image paths referenced by `cards.csv`
- the discovered folder labels are 53 suit-specific card identities, not the assignment's 14 rank targets
- the metadata schema can be normalized to the assignment's 14 classes because CSV rank `xxx` corresponds to `joker`

## Canonical local dataset source
The repo now derives a complete local 14-rank dataset from the images that do exist locally and uses that as the canonical Stage 5 source:

- dataset root: `data/processed/rank14_from_local_raw/`
- layout: flat class directories (`ace` through `joker`)
- total images: 1,866
- derivation method: existing raw `train/` and `test/` images are grouped by rank, collapsing suit-specific 53-card labels into 14 assignment targets
- storage mode: hard links back to the original raw image files
- manifest: `data/processed/rank14_from_local_raw/manifest.csv`
- summary: `data/processed/rank14_from_local_raw/summary.json`

Stage 5 can proceed from this derived 14-rank dataset because every rank has at least 5 examples, which supports strict 5-fold stratified CV. The main caveat is severe class imbalance, especially `joker` with only 5 images.

## Deliverable requirements
The final submission must include:

- a Python notebook with all cell outputs visible
- classification across all 14 classes
- 5-fold cross-validation
- clear result visualizations
- feature maps rendered for:
  - at least two face-card types
  - at least two number-card types
- an augmentation attempt if validation accuracy is below 90%
- a final submission folder named `LASTNAME_HW5`
- a zip file named `LASTNAME_HW5.zip`

## Current repository state
This repository is intentionally starting from a blank implementation state.

Before handing the project to Codex, it contains only:

- `AGENTS.md`
- `README.md`
- `handoff.md`
- `repo_skeleton.txt`

The idea is to let Codex build the repo in a staged, controlled way rather than dropping directly into an unstructured notebook.

## Intended implementation plan
The likely implementation path is:

1. scaffold the repo
2. implement dataset loading and fold logic
3. implement a pretrained baseline model
4. build one-fold training/evaluation
5. scale to 5-fold cross-validation
6. generate plots and confusion matrices
7. generate feature maps
8. improve performance if needed
9. assemble the final notebook
10. package final submission files

## Preferred technical direction
The initial working assumption is:

- framework: PyTorch
- baseline pretrained model: ResNet50
- baseline method: bottlenecking
- backup improvement path: selective fine-tuning
- validation method: stratified 5-fold cross-validation

This can change if implementation evidence suggests a better choice, but the default should be to stay simple and defensible.

## Repository guidance
`AGENTS.md` is the main operating manual for Codex.

`handoff.md` is the live project-state document and should be updated after each stage so a new Codex session can continue without confusion.

`repo_skeleton.txt` defines the intended initial folder structure and main file roles once scaffolding begins.

## Python setup for this repo
This repository should use the project virtual-environment interpreter created by `bootstrap.ps1`:

`C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe`

Use that venv interpreter for future Codex steps in this repo. Fall back to the machine-wide base interpreter only if you are explicitly repairing the bootstrap path.

Recommended command pattern:

```powershell
& 'C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe' --version
& 'C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe' -m pip --version
& 'C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe' -m pip install -r requirements.txt
```

For running project code from this repository, use the same prefix:

```powershell
& 'C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe' -c "print('hello')"
```

The venv location is defined by `bootstrap.ps1` as:
- venv root: `$HOME\.venvs`
- project venv: `$HOME\.venvs\OliviaMLAssignment`

## Dependency setup
The project dependency list for the current repo stages lives in `requirements.txt`.

Install command:

```powershell
& 'C:\Users\golde\.venvs\OliviaMLAssignment\Scripts\python.exe' -m pip install -r requirements.txt
```

Current repo coverage in `requirements.txt` includes:
- dataset discovery/loading support
- fold generation support
- pretrained ResNet50 model construction
- upcoming one-fold training support
- plotting and notebook support

Current environment status:
- the repo venv exists and is importable
- the required core packages for Stages 1 through 4 were successfully verified from that venv in the current session

## Success criteria
This repo is successful when it produces:

- a clean modular implementation
- a complete final notebook with outputs visible
- cross-validation evidence
- required visualizations
- required feature maps
- a clear submission package

## Notes for the implementer
Do not optimize prematurely. The first priority is correctness and assignment compliance.

A simple, well-documented ResNet50 pipeline that clearly satisfies the rubric is better than an overcomplicated experiment that becomes hard to finish or explain.
