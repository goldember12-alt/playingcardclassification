"""Stage 8 improvement-pass orchestration and baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - surfaced when called
    pd = None

from src.training.cross_validate import run_cross_validation
from src.utils.paths import OUTPUTS_DIR


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("Stage 8 reporting requires pandas.")
    return pd


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_runs(
    baseline_summary_path: str | Path,
    improved_summary_path: str | Path,
    output_stem: str,
) -> dict[str, Any]:
    """Compare two cross-validation runs and save a concise Stage 8 report."""
    pandas = _require_pandas()
    baseline = _load_json(baseline_summary_path)
    improved = _load_json(improved_summary_path)

    baseline_rows = pandas.DataFrame(baseline["per_fold_results"]).sort_values(by=["fold"], kind="stable")
    improved_rows = pandas.DataFrame(improved["per_fold_results"]).sort_values(by=["fold"], kind="stable")
    merged = baseline_rows.merge(
        improved_rows,
        on="fold",
        suffixes=("_baseline", "_improved"),
        how="inner",
    )
    merged["delta_val_accuracy"] = (
        merged["val_accuracy_at_best_epoch_improved"] - merged["val_accuracy_at_best_epoch_baseline"]
    )
    merged["delta_val_loss"] = merged["val_loss_at_best_epoch_improved"] - merged["val_loss_at_best_epoch_baseline"]

    comparison = {
        "baseline_run_name": baseline["run_name"],
        "improved_run_name": improved["run_name"],
        "baseline_summary_path": str(Path(baseline_summary_path).resolve()),
        "improved_summary_path": str(Path(improved_summary_path).resolve()),
        "strategy_summary": {
            "use_augmentation": bool(improved.get("use_augmentation", False)),
            "augmentation_profile": improved.get("augmentation_profile"),
            "freeze_backbone": bool(improved.get("freeze_backbone", True)),
            "unfreeze_from": improved.get("unfreeze_from"),
            "optimizer_name": improved.get("optimizer_name"),
            "learning_rate": improved.get("learning_rate"),
            "backbone_learning_rate": improved.get("backbone_learning_rate"),
            "class_weight_strategy": improved.get("class_weight_strategy"),
            "sampling_strategy": improved.get("sampling_strategy"),
            "scheduler_name": improved.get("scheduler_name"),
            "targeted_augmentation_profile": improved.get("targeted_augmentation_profile"),
            "label_smoothing": improved.get("label_smoothing"),
        },
        "baseline_metrics": baseline["aggregate_metrics"],
        "improved_metrics": improved["aggregate_metrics"],
        "metric_deltas": {
            "val_accuracy_at_best_epoch_mean_delta": float(
                improved["aggregate_metrics"]["val_accuracy_at_best_epoch"]["mean"]
                - baseline["aggregate_metrics"]["val_accuracy_at_best_epoch"]["mean"]
            ),
            "val_loss_at_best_epoch_mean_delta": float(
                improved["aggregate_metrics"]["val_loss_at_best_epoch"]["mean"]
                - baseline["aggregate_metrics"]["val_loss_at_best_epoch"]["mean"]
            ),
            "train_accuracy_at_best_epoch_mean_delta": float(
                improved["aggregate_metrics"]["train_accuracy_at_best_epoch"]["mean"]
                - baseline["aggregate_metrics"]["train_accuracy_at_best_epoch"]["mean"]
            ),
            "elapsed_seconds_total_mean_delta": float(
                improved["aggregate_metrics"]["elapsed_seconds_total"]["mean"]
                - baseline["aggregate_metrics"]["elapsed_seconds_total"]["mean"]
            ),
        },
        "per_fold_comparison": merged[
            [
                "fold",
                "val_accuracy_at_best_epoch_baseline",
                "val_accuracy_at_best_epoch_improved",
                "delta_val_accuracy",
                "val_loss_at_best_epoch_baseline",
                "val_loss_at_best_epoch_improved",
                "delta_val_loss",
            ]
        ].to_dict(orient="records"),
        "notes": [
            "Stage 8 keeps the Stage 5 baseline run as the comparison point of record.",
            "Because mean validation accuracy stayed below 90%, this improvement pass is required to meet the project's target.",
        ],
    }

    logs_dir = OUTPUTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    comparison_json_path = logs_dir / f"{output_stem}_comparison.json"
    comparison_csv_path = logs_dir / f"{output_stem}_per_fold_comparison.csv"
    comparison_md_path = logs_dir / f"{output_stem}_comparison.md"

    merged[
        [
            "fold",
            "val_accuracy_at_best_epoch_baseline",
            "val_accuracy_at_best_epoch_improved",
            "delta_val_accuracy",
            "val_loss_at_best_epoch_baseline",
            "val_loss_at_best_epoch_improved",
            "delta_val_loss",
        ]
    ].to_csv(comparison_csv_path, index=False)
    with comparison_json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    lines = ["## Stage 8 Improvement Comparison", ""]
    lines.append(f"- Baseline run: `{baseline['run_name']}`")
    lines.append(f"- Improved run: `{improved['run_name']}`")
    lines.append(
        "- Improvement strategy: "
        f"augmentation={'yes' if improved.get('use_augmentation') else 'no'}"
        + (
            f" | profile=`{improved.get('augmentation_profile')}`"
            if improved.get("augmentation_profile")
            else ""
        )
    )
    lines.append(
        "- Mean validation accuracy: "
        f"{baseline['aggregate_metrics']['val_accuracy_at_best_epoch']['mean']:.4f} -> "
        f"{improved['aggregate_metrics']['val_accuracy_at_best_epoch']['mean']:.4f}"
    )
    lines.append(
        "- Mean validation accuracy delta: "
        f"{comparison['metric_deltas']['val_accuracy_at_best_epoch_mean_delta']:+.4f}"
    )
    lines.append(
        "- Mean validation loss: "
        f"{baseline['aggregate_metrics']['val_loss_at_best_epoch']['mean']:.4f} -> "
        f"{improved['aggregate_metrics']['val_loss_at_best_epoch']['mean']:.4f}"
    )
    lines.append(
        "- Mean validation loss delta: "
        f"{comparison['metric_deltas']['val_loss_at_best_epoch_mean_delta']:+.4f}"
    )
    lines.append("")
    lines.append("### Per-Fold Validation Accuracy")
    lines.append("")
    lines.append("| Fold | Baseline | Improved | Delta |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in comparison["per_fold_comparison"]:
        lines.append(
            f"| {row['fold']} | {row['val_accuracy_at_best_epoch_baseline']:.4f} | "
            f"{row['val_accuracy_at_best_epoch_improved']:.4f} | {row['delta_val_accuracy']:+.4f} |"
        )
    lines.append("")
    lines.append("### Notes")
    lines.append("")
    for note in comparison["notes"]:
        lines.append(f"- {note}")

    with comparison_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    comparison["artifacts"] = {
        "comparison_json": str(comparison_json_path.resolve()),
        "comparison_csv": str(comparison_csv_path.resolve()),
        "comparison_md": str(comparison_md_path.resolve()),
    }
    return comparison


def run_stage8_improvement(
    baseline_summary_path: str | Path = OUTPUTS_DIR / "logs" / "stage5_baseline_5fold_seed42_aggregate_summary.json",
    run_name: str = "stage8_augmented",
    num_folds: int = 5,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    device: str = "auto",
    image_size: int = 224,
    random_seed: int = 42,
    model_name: str = "resnet50",
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
    label_smoothing: float = 0.0,
    use_augmentation: bool = True,
    augmentation_profile: str | None = "stage8_cards",
    targeted_augmentation_profile: str | None = None,
    targeted_augmentation_max_class_count: int = 32,
) -> dict[str, Any]:
    """Run a bounded Stage 8 improvement pass and compare it against baseline."""
    improved = run_cross_validation(
        run_name=run_name,
        num_folds=num_folds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        device=device,
        image_size=image_size,
        random_seed=random_seed,
        model_name=model_name,
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
        label_smoothing=label_smoothing,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        targeted_augmentation_profile=targeted_augmentation_profile,
        targeted_augmentation_max_class_count=targeted_augmentation_max_class_count,
    )
    comparison = compare_runs(
        baseline_summary_path=baseline_summary_path,
        improved_summary_path=improved["artifacts"]["aggregate_summary_json"],
        output_stem=run_name,
    )
    return {
        "improved_run": improved,
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = run_stage8_improvement()
    print(
        json.dumps(
            {
                "improved_summary": result["improved_run"]["artifacts"]["aggregate_summary_json"],
                "comparison_artifacts": result["comparison"]["artifacts"],
                "accuracy_delta": result["comparison"]["metric_deltas"]["val_accuracy_at_best_epoch_mean_delta"],
            },
            indent=2,
        )
    )
