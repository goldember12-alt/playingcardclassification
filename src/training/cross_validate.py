"""Stage 5 cross-validation orchestration utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

from src.data.folds import FoldSummary, build_fold_inventory, save_fold_artifacts
from src.training.train_one_fold import train_one_fold
from src.utils.paths import DERIVED_RANK_DATASET_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR


@dataclass(frozen=True)
class CrossValidationArtifacts:
    """Aggregate artifact locations produced by a Stage 5 run."""

    fold_assignments_csv: str
    fold_overview_csv: str
    validation_counts_csv: str
    per_fold_results_csv: str
    summary_table_csv: str
    aggregate_summary_json: str
    aggregate_summary_md: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for Stage 5 cross-validation aggregation.")
    return pd


def _build_stem(run_name: str, num_folds: int, random_seed: int) -> str:
    safe_run_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in run_name)
    return f"{safe_run_name}_{num_folds}fold_seed{random_seed}"


def _artifact_paths(run_name: str, num_folds: int, random_seed: int) -> tuple[str, CrossValidationArtifacts]:
    stem = _build_stem(run_name=run_name, num_folds=num_folds, random_seed=random_seed)
    logs_dir = OUTPUTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return stem, CrossValidationArtifacts(
        fold_assignments_csv="",
        fold_overview_csv="",
        validation_counts_csv="",
        per_fold_results_csv=str((logs_dir / f"{stem}_per_fold_results.csv").resolve()),
        summary_table_csv=str((logs_dir / f"{stem}_summary_table.csv").resolve()),
        aggregate_summary_json=str((logs_dir / f"{stem}_aggregate_summary.json").resolve()),
        aggregate_summary_md=str((logs_dir / f"{stem}_aggregate_summary.md").resolve()),
    )


def _best_epoch_row(history: Any, best_epoch: int) -> Any:
    pandas = _require_pandas()
    if not isinstance(history, pandas.DataFrame):
        history = pandas.DataFrame(history)
    matching_rows = history.loc[history["epoch"] == best_epoch]
    if matching_rows.empty:
        raise ValueError(f"Best epoch {best_epoch} was not found in the recorded fold history.")
    return matching_rows.iloc[-1]


def _fold_result_row(history: Any, summary: dict[str, Any]) -> dict[str, Any]:
    best_epoch = int(summary["best_epoch"])
    best_row = _best_epoch_row(history, best_epoch=best_epoch)
    artifacts = dict(summary["artifacts"])
    return {
        "fold": int(summary["fold_index"]),
        "best_epoch": best_epoch,
        "train_samples": int(summary["train_samples"]),
        "validation_samples": int(summary["validation_samples"]),
        "best_metric_name": str(summary["best_metric_name"]),
        "best_metric_value": float(summary["best_metric_value"]),
        "train_loss_at_best_epoch": float(best_row["train_loss"]),
        "train_accuracy_at_best_epoch": float(best_row["train_accuracy"]),
        "val_loss_at_best_epoch": float(best_row["val_loss"]),
        "val_accuracy_at_best_epoch": float(best_row["val_accuracy"]),
        "elapsed_seconds_total": float(history["elapsed_seconds"].sum()),
        "checkpoint_path": str(artifacts["checkpoint_path"]),
        "metrics_csv_path": str(artifacts["metrics_csv_path"]),
        "fold_summary_json_path": str(artifacts["summary_json_path"]),
    }


def _dataset_balance_note(fold_summary: FoldSummary) -> str:
    counts = {str(class_name): int(count) for class_name, count in fold_summary.pool_class_counts.items()}
    if not counts:
        return "The dataset remains imbalanced, but class-count details were not available in the saved fold summary."

    smallest_class_name, smallest_class_count = min(counts.items(), key=lambda item: (item[1], item[0]))
    validation_counts = [
        int(class_counts.get(smallest_class_name, 0))
        for class_counts in fold_summary.validation_class_counts.values()
    ]
    if validation_counts:
        min_count = min(validation_counts)
        max_count = max(validation_counts)
        if min_count == max_count:
            fold_detail = f"Each validation fold includes {min_count} `{smallest_class_name}` examples."
        else:
            fold_detail = (
                f"Validation folds include between {min_count} and {max_count} "
                f"`{smallest_class_name}` examples."
            )
    else:
        fold_detail = "Per-fold validation counts were not available in the saved summary."

    return (
        f"The dataset remains imbalanced; `{smallest_class_name}` is the smallest class with "
        f"{smallest_class_count} images. {fold_detail}"
    )


def _aggregate_metrics(per_fold_results: Any) -> tuple[dict[str, dict[str, float]], Any]:
    pandas = _require_pandas()
    if not isinstance(per_fold_results, pandas.DataFrame):
        per_fold_results = pandas.DataFrame(per_fold_results)

    metric_columns = [
        "best_epoch",
        "best_metric_value",
        "train_loss_at_best_epoch",
        "train_accuracy_at_best_epoch",
        "val_loss_at_best_epoch",
        "val_accuracy_at_best_epoch",
        "elapsed_seconds_total",
    ]
    if per_fold_results.empty:
        raise ValueError("Cross-validation aggregation requires at least one completed fold result.")

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for column in metric_columns:
        values = per_fold_results[column].astype(float)
        aggregate_metrics[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary_rows = per_fold_results.copy()
    mean_row = {column: None for column in summary_rows.columns}
    std_row = {column: None for column in summary_rows.columns}
    mean_row["fold"] = "mean"
    std_row["fold"] = "std"
    for column, metrics in aggregate_metrics.items():
        mean_row[column] = metrics["mean"]
        std_row[column] = metrics["std"]
    summary_rows = pandas.concat([summary_rows, pandas.DataFrame([mean_row, std_row])], ignore_index=True)
    return aggregate_metrics, summary_rows


def _write_markdown_summary(
    run_name: str,
    fold_summary: FoldSummary,
    per_fold_results: Any,
    aggregate_metrics: dict[str, dict[str, float]],
    artifact_path: str | Path,
    use_augmentation: bool = False,
    augmentation_profile: str | None = None,
    optimizer_name: str = "adam",
    backbone_learning_rate: float | None = None,
    class_weight_strategy: str = "none",
    sampling_strategy: str = "none",
    scheduler_name: str = "none",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> None:
    pandas = _require_pandas()
    if not isinstance(per_fold_results, pandas.DataFrame):
        per_fold_results = pandas.DataFrame(per_fold_results)

    lines = ["## Cross-Validation Summary", ""]
    lines.append(f"- Run name: `{run_name}`")
    lines.append(f"- Dataset root: `{fold_summary.discovered_root}`")
    lines.append(f"- Fold source strategy: `{fold_summary.fold_source_strategy}`")
    lines.append(f"- Number of folds: {fold_summary.n_splits}")
    lines.append(f"- Random seed: {fold_summary.random_seed}")
    lines.append(f"- CV pool samples: {fold_summary.total_pool_samples}")
    lines.append(f"- Training augmentation enabled: {'yes' if use_augmentation else 'no'}")
    if augmentation_profile:
        lines.append(f"- Augmentation profile: `{augmentation_profile}`")
    lines.append(f"- Optimizer: `{optimizer_name}`")
    if backbone_learning_rate is not None:
        lines.append(f"- Backbone learning rate: `{backbone_learning_rate}`")
    lines.append(f"- Class weighting strategy: `{class_weight_strategy}`")
    lines.append(f"- Sampling strategy: `{sampling_strategy}`")
    lines.append(f"- Scheduler: `{scheduler_name}`")
    if early_stopping_patience is not None:
        lines.append(
            f"- Early stopping: patience `{early_stopping_patience}` | min delta `{early_stopping_min_delta}`"
        )
    lines.append("")
    lines.append(
        "- Validation accuracy (best epoch per fold): "
        f"{aggregate_metrics['val_accuracy_at_best_epoch']['mean']:.4f} +/- "
        f"{aggregate_metrics['val_accuracy_at_best_epoch']['std']:.4f}"
    )
    lines.append(
        "- Validation loss (best epoch per fold): "
        f"{aggregate_metrics['val_loss_at_best_epoch']['mean']:.4f} +/- "
        f"{aggregate_metrics['val_loss_at_best_epoch']['std']:.4f}"
    )
    lines.append(
        "- Total elapsed seconds per fold: "
        f"{aggregate_metrics['elapsed_seconds_total']['mean']:.2f} +/- "
        f"{aggregate_metrics['elapsed_seconds_total']['std']:.2f}"
    )
    lines.append("")
    lines.append("### Per-Fold Results")
    lines.append("")
    lines.append("| Fold | Best Epoch | Val Accuracy | Val Loss | Train Accuracy | Train Loss | Elapsed Seconds |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in per_fold_results.itertuples(index=False):
        lines.append(
            f"| {row.fold} | {row.best_epoch} | {row.val_accuracy_at_best_epoch:.4f} | "
            f"{row.val_loss_at_best_epoch:.4f} | {row.train_accuracy_at_best_epoch:.4f} | "
            f"{row.train_loss_at_best_epoch:.4f} | {row.elapsed_seconds_total:.2f} |"
        )

    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- This run uses the derived local 14-rank dataset under `data/processed/rank14_from_local_raw/`.")
    lines.append(f"- {_dataset_balance_note(fold_summary)}")
    for note in fold_summary.notes:
        lines.append(f"- {note}")

    with Path(artifact_path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_cross_validation(
    run_name: str = "baseline",
    num_folds: int = 5,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    device: str = "auto",
    image_size: int = 224,
    random_seed: int = 42,
    raw_data_dir: str | Path = PROCESSED_DATA_DIR,
    dataset_name: str | None = DERIVED_RANK_DATASET_DIR.name,
    dataset_root: str | Path | None = None,
    model_name: str = "resnet50",
    num_classes: int | None = 14,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,
    classifier_hidden_dim: int | None = None,
    classifier_dropout: float = 0.0,
    optimizer_name: str = "adam",
    backbone_learning_rate: float | None = None,
    loss_name: str = "cross_entropy",
    class_weight_strategy: str = "none",
    class_weight_beta: float = 0.999,
    sampling_strategy: str = "none",
    sampling_weight_power: float = 1.0,
    scheduler_name: str = "none",
    scheduler_min_lr: float = 1e-6,
    scheduler_plateau_factor: float = 0.5,
    scheduler_plateau_patience: int = 2,
    label_smoothing: float = 0.0,
    use_augmentation: bool = False,
    augmentation_profile: str | None = None,
    targeted_augmentation_profile: str | None = None,
    targeted_augmentation_max_class_count: int = 32,
    monitor_metric: str = "val_accuracy",
    monitor_mode: str = "max",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> dict[str, Any]:
    """Run the full Stage 5 k-fold loop and save aggregate artifacts."""
    pandas = _require_pandas()

    assignments, fold_summary = build_fold_inventory(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
        n_splits=num_folds,
        random_seed=random_seed,
    )
    class_mapping = (
        assignments.loc[assignments["fold_role"] == "cv_pool", ["class_name", "class_index"]]
        .drop_duplicates()
        .sort_values(by=["class_index", "class_name"], kind="stable")
    )
    class_names = class_mapping["class_name"].tolist()

    stem, artifacts = _artifact_paths(run_name=run_name, num_folds=num_folds, random_seed=random_seed)
    fold_paths = save_fold_artifacts(assignments=assignments, summary=fold_summary, stem=stem)
    artifacts = CrossValidationArtifacts(
        fold_assignments_csv=fold_paths["assignments_csv"],
        fold_overview_csv=fold_paths["overview_csv"],
        validation_counts_csv=fold_paths["validation_counts_csv"],
        per_fold_results_csv=artifacts.per_fold_results_csv,
        summary_table_csv=artifacts.summary_table_csv,
        aggregate_summary_json=artifacts.aggregate_summary_json,
        aggregate_summary_md=artifacts.aggregate_summary_md,
    )

    per_fold_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_index in range(num_folds):
        history, fold_run_summary = train_one_fold(
            fold_index=fold_index,
            assignments=assignments,
            run_name=run_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=num_workers,
            device=device,
            num_folds=num_folds,
            image_size=image_size,
            random_seed=random_seed,
            raw_data_dir=raw_data_dir,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            model_name=model_name,
            num_classes=num_classes,
            class_names=class_names,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            unfreeze_from=unfreeze_from,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
            optimizer_name=optimizer_name,
            backbone_learning_rate=backbone_learning_rate,
            loss_name=loss_name,
            class_weight_strategy=class_weight_strategy,
            class_weight_beta=class_weight_beta,
            sampling_strategy=sampling_strategy,
            sampling_weight_power=sampling_weight_power,
            scheduler_name=scheduler_name,
            scheduler_min_lr=scheduler_min_lr,
            scheduler_plateau_factor=scheduler_plateau_factor,
            scheduler_plateau_patience=scheduler_plateau_patience,
            label_smoothing=label_smoothing,
            use_augmentation=use_augmentation,
            augmentation_profile=augmentation_profile,
            targeted_augmentation_profile=targeted_augmentation_profile,
            targeted_augmentation_max_class_count=targeted_augmentation_max_class_count,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        per_fold_rows.append(_fold_result_row(history=history, summary=fold_run_summary))
        fold_summaries.append(fold_run_summary)

    per_fold_results = pandas.DataFrame(per_fold_rows).sort_values(by=["fold"], kind="stable").reset_index(drop=True)
    aggregate_metrics, summary_table = _aggregate_metrics(per_fold_results=per_fold_results)

    per_fold_results.to_csv(artifacts.per_fold_results_csv, index=False)
    summary_table.to_csv(artifacts.summary_table_csv, index=False)
    _write_markdown_summary(
        run_name=run_name,
        fold_summary=fold_summary,
        per_fold_results=per_fold_results,
        aggregate_metrics=aggregate_metrics,
        artifact_path=artifacts.aggregate_summary_md,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        optimizer_name=optimizer_name,
        backbone_learning_rate=backbone_learning_rate,
        class_weight_strategy=class_weight_strategy,
        sampling_strategy=sampling_strategy,
        scheduler_name=scheduler_name,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    result = {
        "run_name": run_name,
        "dataset_root": fold_summary.discovered_root,
        "dataset_name": dataset_name,
        "raw_data_dir": str(Path(raw_data_dir).resolve()),
        "num_folds": num_folds,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "device": device,
        "image_size": image_size,
        "random_seed": random_seed,
        "model_name": model_name,
        "pretrained_backbone": pretrained_backbone,
        "freeze_backbone": freeze_backbone,
        "unfreeze_from": unfreeze_from,
        "classifier_hidden_dim": classifier_hidden_dim,
        "classifier_dropout": classifier_dropout,
        "optimizer_name": optimizer_name,
        "backbone_learning_rate": backbone_learning_rate,
        "loss_name": loss_name,
        "class_weight_strategy": class_weight_strategy,
        "class_weight_beta": class_weight_beta,
        "sampling_strategy": sampling_strategy,
        "sampling_weight_power": sampling_weight_power,
        "scheduler_name": scheduler_name,
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_plateau_factor": scheduler_plateau_factor,
        "scheduler_plateau_patience": scheduler_plateau_patience,
        "label_smoothing": label_smoothing,
        "use_augmentation": use_augmentation,
        "augmentation_profile": augmentation_profile,
        "targeted_augmentation_profile": targeted_augmentation_profile,
        "targeted_augmentation_max_class_count": targeted_augmentation_max_class_count,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "class_names": class_names,
        "fold_summary": fold_summary.to_dict(),
        "per_fold_results": per_fold_results.to_dict(orient="records"),
        "aggregate_metrics": aggregate_metrics,
        "artifacts": artifacts.to_dict(),
        "notes": [
            "This run uses deterministic 5-fold stratified cross-validation over the derived local 14-rank dataset.",
            _dataset_balance_note(fold_summary),
            "Training augmentation was enabled." if use_augmentation else "Training augmentation was not enabled.",
            f"Class weighting strategy: `{class_weight_strategy}`.",
            f"Sampling strategy: `{sampling_strategy}`.",
            f"Scheduler: `{scheduler_name}`.",
        ],
        "fold_run_summaries": fold_summaries,
    }
    with Path(artifacts.aggregate_summary_json).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    result = run_cross_validation()
    print(json.dumps({"artifacts": result["artifacts"], "aggregate_metrics": result["aggregate_metrics"]}, indent=2))
