# AGENTS.md

## Project identity
This repository exists to complete a machine learning homework assignment on image classification using a pretrained CNN and the Kaggle cards image dataset. The assignment must be completed in a way that is clean, modular, reproducible, and notebook-submission-ready.

Working directory:
`C:\Users\golde\OneDrive - University of Virginia\OliviaMLAssignment`

## Core assignment requirements
The final work must satisfy all of the following:

1. Use a pretrained image classification model.
2. Choose and implement one of:
   - bottlenecking
   - fine-tuning
3. Classify all 14 classes in the dataset.
4. Use 5-fold cross-validation (`k=5`).
5. Demonstrate that the model performs well on validation data.
6. Visualize results.
7. Visualize feature maps from the trained network for:
   - at least two face-card classes
   - at least two number-card classes
8. If performance is poor (< 90% validation accuracy), apply some form of data augmentation and document the attempt.
9. The final deliverable must be a Python notebook with visible outputs.
10. The final submission must be packaged into a folder named `LASTNAME_HW5`, zipped as `LASTNAME_HW5.zip`.

## Working philosophy
This project should be built in a staged, doc-first way.

The implementation should prioritize:
- clarity over cleverness
- reproducibility over speed hacks
- modular code over notebook-only code
- explicit saved outputs over hidden intermediate work
- a notebook that tells a coherent story from setup to conclusion

Do not jump directly to a giant notebook. Build reusable components first, then assemble the final notebook once the pipeline works.

## Preferred technical choices
Unless strong evidence suggests otherwise, use the following defaults:

- Framework: PyTorch
- Baseline backbone: ResNet50 pretrained on ImageNet
- Baseline method: bottlenecking first
- Improvement path: selective fine-tuning if needed
- Cross-validation: StratifiedKFold with 5 folds
- Metrics: validation accuracy at minimum, plus fold summaries
- Visualization stack: matplotlib, sklearn confusion matrix tools, and saved PNG outputs
- Reproducibility: fixed random seeds where feasible

These are defaults, not absolute rules. If a different pretrained model is clearly more practical for the assignment, explain the reason before changing course.

## Non-negotiable implementation constraints
- The final notebook must show visible outputs.
- The project must support all 14 classes explicitly.
- Cross-validation must be real 5-fold validation, not a single train/val split relabeled as k-fold.
- Feature-map visualization must be implemented, not merely described.
- At least four example inputs must be used for feature-map rendering:
  - 2 face-card types
  - 2 number-card types
- If accuracy is below 90%, at least one augmentation attempt must be added and documented.
- The code and notebook must save or expose enough outputs for a grader to verify the work.

## Repository state at handoff
At the moment of first handoff, the repository intentionally contains only:
- `AGENTS.md`
- `README.md`
- `handoff.md`
- `repo_skeleton.txt`

This is expected. The repository is starting from a documentation-first blank state.

## Primary mission for Codex
Translate the assignment into a staged implementation that produces:

1. a working image classification pipeline
2. 5-fold cross-validation results
3. visualizations of performance
4. feature-map visualizations
5. a clean final notebook suitable for submission
6. a submission folder and zip-ready structure

## Stage plan

### Stage 0 — Repository scaffold
Create the working folder structure and initial placeholder files based on `repo_skeleton.txt`.

Deliverables:
- folder structure created
- placeholder modules created
- basic config and dependency files created

Do not implement full training yet.

### Stage 1 — Dataset ingestion and validation
Implement dataset discovery and loading logic.

Requirements:
- identify the 14 classes
- map class names to labels
- verify image loading works
- verify train/validation transforms can be applied
- produce a dataset summary suitable for notebook inclusion

Deliverables:
- reusable dataset loader module
- class-to-index mapping
- sanity-check output showing dataset integrity

### Stage 2 — Fold generation
Implement 5-fold cross-validation split generation.

Requirements:
- use stratified folds if possible
- save fold assignments or generate them deterministically
- confirm each fold contains all classes as expected

Deliverables:
- fold-generation utility
- reproducible fold behavior
- fold summary output

### Stage 3 — Baseline model
Implement the baseline pretrained model.

Default:
- pretrained ResNet50
- frozen backbone
- new classification head for 14 classes

Deliverables:
- model construction utility
- configurable freeze/unfreeze behavior
- correct output dimensions

### Stage 4 — One-fold training pipeline
Build a reliable single-fold training and validation loop first.

Requirements:
- training loss tracking
- validation loss tracking if feasible
- validation accuracy tracking
- checkpointing of best model for the fold
- saved plots or metrics logs

Deliverables:
- one-fold train/validate pipeline
- metrics collection
- saved outputs

### Stage 5 — Full 5-fold cross-validation
Run and aggregate all 5 folds.

Requirements:
- per-fold metrics
- aggregate mean and standard deviation
- clear saved summary for notebook insertion

Deliverables:
- 5-fold results table
- aggregate validation summary
- stable reproducible run path

### Stage 6 — Result visualization
Implement standard visualizations.

At minimum:
- training/validation curves
- confusion matrix
- sample predictions or misclassifications
- fold summary chart or table

Deliverables:
- saved figure files
- notebook-ready visual assets

### Stage 7 — Feature maps
Implement feature-map extraction and visualization.

Requirements:
- hook one or more intermediate convolutional layers
- render feature maps for at least:
  - 2 face-card classes
  - 2 number-card classes
- save figures and make the logic reusable

Deliverables:
- feature-map extraction utility
- saved feature-map images
- notebook-ready explanation

### Stage 8 — Performance improvement pass
If validation accuracy is below 90%, attempt at least one improvement pass.

Allowed approaches:
- data augmentation
- selective unfreezing / fine-tuning
- modest optimizer or learning-rate tuning

Requirements:
- document what changed
- document whether it improved performance
- do not perform uncontrolled experimentation without recording decisions

Deliverables:
- updated config or training mode
- comparison against baseline
- concise record of rationale

### Stage 9 — Final notebook assembly
Create the final submission notebook.

The notebook should include:
- assignment overview
- dataset summary
- model choice and rationale
- bottlenecking or fine-tuning explanation
- cross-validation method
- performance results
- visualizations
- feature maps
- concise conclusion

The notebook should be readable by the grader without requiring them to inspect every module.

### Stage 10 — Submission packaging
Prepare the final handoff structure:

- folder named `LASTNAME_HW5`
- all needed files included
- notebook included with outputs visible
- zip-ready layout

## Required work style
For every stage:
1. Make bounded changes only.
2. Do not silently change earlier architectural decisions without documenting why.
3. Prefer writing reusable modules in `src/` over long notebook cells.
4. Keep plots, artifacts, and checkpoints organized.
5. Write concise summaries of what changed.
6. Update `handoff.md` at the end of each stage.

## Required `handoff.md` update protocol
After every meaningful work session, update `handoff.md` with:
- current stage
- what was completed
- key decisions made
- files created or modified
- blockers or open risks
- exact recommended next prompt

The goal is that a fresh Codex session can read `AGENTS.md` and `handoff.md` and continue without ambiguity.

## Decision defaults to use unless contradicted
At the start of implementation, assume:
- model: ResNet50
- initial strategy: bottlenecking
- fallback strategy: fine-tuning upper layers
- augmentation only if needed for sub-90% validation accuracy
- notebook created after pipeline functionality is verified

## Quality bar
A good result is not just high accuracy. A good result must also be:
- well organized
- clearly explained
- reproducible
- easy to grade
- visibly compliant with every assignment requirement

## Things to avoid
- do not hide all logic in the notebook
- do not skip feature maps
- do not hardcode assumptions without explanation
- do not run large uncontrolled experiments without recording them
- do not prioritize cosmetic polish over completing assignment requirements
- do not claim cross-validation unless all 5 folds are actually executed or explicitly staged and documented

## Expected communication style from Codex
At the end of each stage, provide:
- a concise summary of work completed
- the exact files changed
- any blockers
- the next recommended prompt

Keep summaries practical and specific. Avoid vague statements like “set things up” without naming what was created.

## Initial first task
The first task after reading this file is:

Create the repository scaffold from `repo_skeleton.txt`, then update `handoff.md` with the new structure and the recommended next implementation prompt.
