"""Stage 6 visualization orchestration built on completed Stage 5 outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

from PIL import Image, ImageDraw, ImageFont

from src.data.dataset import EXPECTED_CARD_CLASSES
from src.evaluation.confusion import save_confusion_outputs
from src.evaluation.predict import collect_cross_validation_predictions
from src.utils.paths import OUTPUTS_DIR, VISUALIZATIONS_OUTPUT_DIR


@dataclass(frozen=True)
class Stage6Artifacts:
    """Paths to saved Stage 6 visualization assets."""

    output_root: str
    training_curves_png: str
    fold_summary_chart_png: str
    fold_summary_table_csv: str
    aggregate_predictions_csv: str
    aggregate_misclassifications_csv: str
    prediction_gallery_png: str
    misclassification_gallery_png: str
    aggregate_confusion_counts_csv: str
    aggregate_confusion_normalized_csv: str
    aggregate_confusion_counts_png: str
    aggregate_confusion_normalized_png: str
    stage6_summary_json: str
    stage6_summary_md: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COLOR_PALETTE = [
    "#2E86AB",
    "#F18F01",
    "#C73E1D",
    "#6C5B7B",
    "#355C7D",
    "#43AA8B",
    "#577590",
    "#B56576",
    "#90BE6D",
    "#4D908E",
]


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("Stage 6 visualization helpers require pandas.")
    return pd


def _font() -> Any:
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _discover_stage5_summary_json() -> Path:
    candidate_paths = sorted((OUTPUTS_DIR / "logs").glob("*_aggregate_summary.json"))
    if not candidate_paths:
        raise FileNotFoundError("No aggregate summary JSON was found under outputs/logs/.")

    canonical_candidates: list[tuple[float, float, str, Path]] = []
    for path in candidate_paths:
        if "smoke" in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset_root = str(payload.get("dataset_root", ""))
        if "rank14_from_local_raw" not in dataset_root:
            continue
        canonical_candidates.append(
            (
                float(payload["aggregate_metrics"]["val_accuracy_at_best_epoch"]["mean"]),
                float(payload["aggregate_metrics"]["val_loss_at_best_epoch"]["mean"]),
                str(payload.get("run_name", path.stem)),
                path,
            )
        )
    if canonical_candidates:
        canonical_candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        return canonical_candidates[0][3]

    non_smoke_candidates = [path for path in candidate_paths if "smoke" not in path.name]
    refresh_candidates = [path for path in non_smoke_candidates if "refresh" in path.name]
    if refresh_candidates:
        return refresh_candidates[-1]
    preferred_candidates = [path for path in non_smoke_candidates if "baseline" in path.name]
    if preferred_candidates:
        return preferred_candidates[-1]
    if non_smoke_candidates:
        return non_smoke_candidates[-1]
    return candidate_paths[-1]


def _load_stage5_summary(stage5_summary_path: str | Path | None = None) -> dict[str, Any]:
    summary_path = Path(stage5_summary_path) if stage5_summary_path else _discover_stage5_summary_json()
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_output_dirs(run_name: str) -> dict[str, Path]:
    root = VISUALIZATIONS_OUTPUT_DIR / run_name
    directories = {
        "root": root,
        "confusion": root / "confusion",
        "predictions": root / "predictions",
        "curves": root / "curves",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _blank_canvas(width: int, height: int, title: str | None = None) -> tuple[Any, Any]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if title:
        draw.text((20, 15), title, fill="black", font=_font())
    return image, draw


def _draw_axes(draw: Any, bounds: tuple[int, int, int, int], y_ticks: list[float], x_ticks: list[int]) -> None:
    x0, y0, x1, y1 = bounds
    font = _font()
    draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
    for tick in y_ticks:
        y = y1 - int((tick - y_ticks[0]) / (y_ticks[-1] - y_ticks[0] or 1.0) * (y1 - y0))
        draw.line([x0, y, x1, y], fill=(220, 220, 220), width=1)
        label = f"{tick:.2f}"
        draw.text((x0 - 38, y - 6), label, fill="black", font=font)

    for tick in x_ticks:
        if len(x_ticks) == 1:
            x = x0 + (x1 - x0) // 2
        else:
            x = x0 + int((tick - x_ticks[0]) / (x_ticks[-1] - x_ticks[0]) * (x1 - x0))
        draw.line([x, y0, x, y1], fill=(235, 235, 235), width=1)
        draw.text((x - 4, y1 + 8), str(tick), fill="black", font=font)


def _plot_series(
    draw: Any,
    bounds: tuple[int, int, int, int],
    x_values: list[int],
    y_values: list[float],
    y_min: float,
    y_max: float,
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = bounds
    if not x_values:
        return

    points: list[tuple[int, int]] = []
    for x_value, y_value in zip(x_values, y_values):
        x = x0 + (
            (x1 - x0) // 2
            if len(x_values) == 1
            else int((x_value - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0))
        )
        y = y1 - int((y_value - y_min) / (y_max - y_min or 1.0) * (y1 - y0))
        points.append((x, y))

    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    for point in points:
        draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=color, outline=color)


def _draw_legend(draw: Any, items: list[tuple[str, tuple[int, int, int]]], start_xy: tuple[int, int]) -> None:
    x, y = start_xy
    font = _font()
    for label, color in items:
        draw.rectangle([x, y + 2, x + 12, y + 14], fill=color, outline=color)
        draw.text((x + 18, y), label, fill="black", font=font)
        y += 18


def save_training_curves(
    metrics_by_fold: dict[int, Any],
    output_path: str | Path,
    run_name: str,
) -> str:
    """Save a combined training/validation-curve figure across all folds."""
    output_path = Path(output_path)
    image, draw = _blank_canvas(1600, 760, title=f"{run_name} Training and Validation Curves")
    font = _font()

    loss_bounds = (80, 80, 760, 620)
    accuracy_bounds = (840, 80, 1520, 620)
    draw.text((80, 55), "Loss", fill="black", font=font)
    draw.text((840, 55), "Accuracy", fill="black", font=font)

    all_epochs = sorted({int(epoch) for frame in metrics_by_fold.values() for epoch in frame["epoch"].tolist()})
    all_losses = [
        float(value)
        for frame in metrics_by_fold.values()
        for column in ("train_loss", "val_loss")
        for value in frame[column].tolist()
    ]
    all_accuracies = [
        float(value)
        for frame in metrics_by_fold.values()
        for column in ("train_accuracy", "val_accuracy")
        for value in frame[column].tolist()
    ]
    loss_min = min(all_losses) - 0.05
    loss_max = max(all_losses) + 0.05
    accuracy_min = max(0.0, min(all_accuracies) - 0.05)
    accuracy_max = min(1.0, max(all_accuracies) + 0.05)

    loss_ticks = [round(loss_min + index * (loss_max - loss_min) / 4, 2) for index in range(5)]
    accuracy_ticks = [round(accuracy_min + index * (accuracy_max - accuracy_min) / 4, 2) for index in range(5)]
    _draw_axes(draw, loss_bounds, loss_ticks, all_epochs)
    _draw_axes(draw, accuracy_bounds, accuracy_ticks, all_epochs)

    legend_items: list[tuple[str, tuple[int, int, int]]] = []
    for palette_index, fold_index in enumerate(sorted(metrics_by_fold)):
        frame = metrics_by_fold[fold_index]
        epochs = [int(value) for value in frame["epoch"].tolist()]
        train_color = _hex_to_rgb(COLOR_PALETTE[palette_index % len(COLOR_PALETTE)])
        val_color = tuple(max(component - 40, 0) for component in train_color)
        _plot_series(draw, loss_bounds, epochs, frame["train_loss"].astype(float).tolist(), loss_min, loss_max, train_color)
        _plot_series(draw, loss_bounds, epochs, frame["val_loss"].astype(float).tolist(), loss_min, loss_max, val_color)
        _plot_series(
            draw,
            accuracy_bounds,
            epochs,
            frame["train_accuracy"].astype(float).tolist(),
            accuracy_min,
            accuracy_max,
            train_color,
        )
        _plot_series(
            draw,
            accuracy_bounds,
            epochs,
            frame["val_accuracy"].astype(float).tolist(),
            accuracy_min,
            accuracy_max,
            val_color,
        )
        legend_items.append((f"Fold {fold_index} train", train_color))
        legend_items.append((f"Fold {fold_index} val", val_color))

    _draw_legend(draw, legend_items, (80, 650))
    image.save(output_path)
    return str(output_path.resolve())


def _draw_bar_chart(
    draw: Any,
    bounds: tuple[int, int, int, int],
    title: str,
    values: list[float],
    labels: list[str],
    mean_value: float,
    color: tuple[int, int, int],
    y_max: float,
) -> None:
    x0, y0, x1, y1 = bounds
    font = _font()
    draw.text((x0, y0 - 22), title, fill="black", font=font)
    draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
    width = x1 - x0
    height = y1 - y0
    bar_width = max(18, width // max(len(values) * 2, 2))

    for index, value in enumerate(values):
        center_x = x0 + int((index + 0.5) / len(values) * width)
        bar_height = int((value / (y_max or 1.0)) * height)
        draw.rectangle(
            [center_x - bar_width // 2, y1 - bar_height, center_x + bar_width // 2, y1],
            fill=color,
            outline=color,
        )
        draw.text((center_x - 10, y1 + 8), labels[index], fill="black", font=font)
        draw.text((center_x - 16, y1 - bar_height - 14), f"{value:.3f}", fill="black", font=font)

    mean_y = y1 - int((mean_value / (y_max or 1.0)) * height)
    draw.line([x0, mean_y, x1, mean_y], fill=(180, 0, 0), width=2)
    draw.text((x1 - 110, mean_y - 16), f"mean={mean_value:.3f}", fill=(180, 0, 0), font=font)


def _save_fold_summary_chart(
    per_fold_results: Any,
    output_path: str | Path,
    table_output_path: str | Path,
) -> tuple[str, str]:
    """Save a fold-summary chart and a compact CSV table."""
    pandas = _require_pandas()
    if not isinstance(per_fold_results, pandas.DataFrame):
        per_fold_results = pandas.DataFrame(per_fold_results)

    summary_table = per_fold_results[
        [
            "fold",
            "best_epoch",
            "val_accuracy_at_best_epoch",
            "val_loss_at_best_epoch",
            "train_accuracy_at_best_epoch",
            "train_loss_at_best_epoch",
            "elapsed_seconds_total",
        ]
    ].copy()
    summary_table.to_csv(table_output_path, index=False)

    folds = [str(int(value)) for value in summary_table["fold"].tolist()]
    accuracies = summary_table["val_accuracy_at_best_epoch"].astype(float).tolist()
    losses = summary_table["val_loss_at_best_epoch"].astype(float).tolist()

    image, draw = _blank_canvas(1200, 520, title="Stage 5 Fold Summary")
    _draw_bar_chart(
        draw=draw,
        bounds=(70, 90, 560, 400),
        title="Validation Accuracy By Fold",
        values=accuracies,
        labels=folds,
        mean_value=sum(accuracies) / len(accuracies),
        color=_hex_to_rgb("#2E86AB"),
        y_max=1.0,
    )
    _draw_bar_chart(
        draw=draw,
        bounds=(640, 90, 1130, 400),
        title="Validation Loss By Fold",
        values=losses,
        labels=folds,
        mean_value=sum(losses) / len(losses),
        color=_hex_to_rgb("#F6C85F"),
        y_max=max(losses) * 1.15,
    )
    image.save(output_path)
    return str(Path(output_path).resolve()), str(Path(table_output_path).resolve())


def _select_gallery_rows(predictions: Any, max_items: int, misclassifications_only: bool) -> Any:
    pandas = _require_pandas()
    if not isinstance(predictions, pandas.DataFrame):
        predictions = pandas.DataFrame(predictions)

    if misclassifications_only:
        filtered = predictions.loc[~predictions["is_correct"]].copy()
        if filtered.empty:
            filtered = predictions.copy()
    else:
        filtered = predictions.copy()

    return filtered.sort_values(
        by=["fold", "is_correct", "confidence", "true_class_name", "predicted_class_name", "path"],
        ascending=[True, True, False, True, True, True],
        kind="stable",
    ).head(max_items).reset_index(drop=True)


def _fit_image(path: str | Path, size: tuple[int, int]) -> Any:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _save_prediction_gallery(
    predictions: Any,
    output_path: str | Path,
    title: str,
    max_items: int = 12,
    misclassifications_only: bool = False,
) -> str:
    selected = _select_gallery_rows(
        predictions=predictions,
        max_items=max_items,
        misclassifications_only=misclassifications_only,
    )
    output_path = Path(output_path)
    font = _font()

    tile_width = 240
    tile_height = 280
    ncols = 4
    nrows = max(1, (len(selected) + ncols - 1) // ncols)
    image = Image.new("RGB", (tile_width * ncols, 40 + tile_height * nrows), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), title, fill="black", font=font)

    if selected.empty:
        draw.text((20, 60), "No prediction rows available.", fill="black", font=font)
        image.save(output_path)
        return str(output_path.resolve())

    for index, row in enumerate(selected.itertuples(index=False)):
        row_index = index // ncols
        column_index = index % ncols
        x0 = column_index * tile_width
        y0 = 40 + row_index * tile_height
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile_draw = ImageDraw.Draw(tile)
        fitted = _fit_image(row.path, (200, 180))
        tile.paste(fitted, (20, 10))
        tile_draw.rectangle([0, 0, tile_width - 1, tile_height - 1], outline=(210, 210, 210), width=1)
        text_lines = [
            f"Fold {row.fold}",
            f"true={row.true_class_name}",
            f"pred={row.predicted_class_name}",
            f"conf={row.confidence:.2f}",
        ]
        text_y = 200
        for line in text_lines:
            tile_draw.text((12, text_y), line, fill="black", font=font)
            text_y += 16
        image.paste(tile, (x0, y0))

    image.save(output_path)
    return str(output_path.resolve())


def _write_stage6_markdown(
    run_name: str,
    stage5_summary: dict[str, Any],
    predictions: Any,
    artifacts: Stage6Artifacts,
) -> None:
    pandas = _require_pandas()
    if not isinstance(predictions, pandas.DataFrame):
        predictions = pandas.DataFrame(predictions)

    overall_accuracy = float(predictions["is_correct"].mean()) if not predictions.empty else 0.0
    misclassification_count = int((~predictions["is_correct"]).sum()) if not predictions.empty else 0
    lines = ["## Stage 6 Visualization Summary", ""]
    lines.append(f"- Stage 5 run: `{run_name}`")
    lines.append(f"- Dataset root: `{stage5_summary['dataset_root']}`")
    lines.append(f"- Validation predictions aggregated across folds: {len(predictions)}")
    lines.append(f"- Aggregated validation accuracy from saved predictions: {overall_accuracy:.4f}")
    lines.append(f"- Misclassified validation examples: {misclassification_count}")
    lines.append("")
    lines.append("### Key Artifacts")
    lines.append("")
    lines.append(f"- Training curves: `{artifacts.training_curves_png}`")
    lines.append(f"- Fold summary chart: `{artifacts.fold_summary_chart_png}`")
    lines.append(f"- Aggregate confusion matrix: `{artifacts.aggregate_confusion_normalized_png}`")
    lines.append(f"- Misclassification gallery: `{artifacts.misclassification_gallery_png}`")
    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- Stage 6 reuses the completed cross-validation checkpoints and logs without retraining.")
    lines.append("- The derived local dataset remains highly imbalanced, especially `joker` with only 5 total images.")
    if overall_accuracy < 0.90:
        lines.append("- Mean validation performance remains below the project's 90% threshold, so Stage 8 will require an augmentation or improvement pass.")
    else:
        lines.append("- Mean validation performance now clears the project's 90% threshold on the refreshed full-dataset baseline.")

    with Path(artifacts.stage6_summary_md).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_stage6_visualizations(
    stage5_summary_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Build the complete Stage 6 visualization/reporting bundle from saved Stage 5 outputs."""
    pandas = _require_pandas()
    stage5_summary = _load_stage5_summary(stage5_summary_path=stage5_summary_path)
    run_name = str(stage5_summary["run_name"])
    directories = _resolve_output_dirs(run_name=run_name)
    class_names = list(EXPECTED_CARD_CLASSES)

    per_fold_results = pandas.DataFrame(stage5_summary["per_fold_results"]).sort_values(by=["fold"], kind="stable")
    assignments = pandas.read_csv(stage5_summary["artifacts"]["fold_assignments_csv"])
    metrics_by_fold = {
        int(row.fold): pandas.read_csv(row.metrics_csv_path)
        for row in per_fold_results.itertuples(index=False)
    }

    training_curves_png = save_training_curves(
        metrics_by_fold=metrics_by_fold,
        output_path=directories["curves"] / "training_validation_curves.png",
        run_name=run_name,
    )
    fold_summary_chart_png, fold_summary_table_csv = _save_fold_summary_chart(
        per_fold_results=per_fold_results,
        output_path=directories["curves"] / "fold_summary_chart.png",
        table_output_path=directories["curves"] / "fold_summary_table.csv",
    )

    predictions = collect_cross_validation_predictions(
        assignments=assignments,
        per_fold_results=per_fold_results,
        class_names=list(stage5_summary["class_names"]),
        image_size=int(stage5_summary["image_size"]),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    aggregate_predictions_csv = directories["predictions"] / "aggregate_validation_predictions.csv"
    aggregate_misclassifications_csv = directories["predictions"] / "aggregate_misclassifications.csv"
    predictions.to_csv(aggregate_predictions_csv, index=False)
    predictions.loc[~predictions["is_correct"]].to_csv(aggregate_misclassifications_csv, index=False)

    prediction_gallery_png = _save_prediction_gallery(
        predictions=predictions,
        output_path=directories["predictions"] / "prediction_gallery.png",
        title="Validation Prediction Samples",
        max_items=12,
        misclassifications_only=False,
    )
    misclassification_gallery_png = _save_prediction_gallery(
        predictions=predictions,
        output_path=directories["predictions"] / "misclassification_gallery.png",
        title="Validation Misclassifications",
        max_items=12,
        misclassifications_only=True,
    )

    aggregate_confusion_paths = save_confusion_outputs(
        predictions=predictions,
        class_names=class_names,
        output_dir=directories["confusion"],
        stem="aggregate_confusion_matrix",
        title_prefix="Aggregate Validation",
    )
    per_fold_confusion_paths: dict[str, dict[str, str]] = {}
    for fold_index in sorted(predictions["fold"].dropna().astype(int).unique().tolist()):
        fold_predictions = predictions.loc[predictions["fold"] == fold_index].copy()
        per_fold_confusion_paths[str(fold_index)] = save_confusion_outputs(
            predictions=fold_predictions,
            class_names=class_names,
            output_dir=directories["confusion"],
            stem=f"fold_{fold_index:02d}_confusion_matrix",
            title_prefix=f"Fold {fold_index} Validation",
        )

    artifacts = Stage6Artifacts(
        output_root=str(directories["root"].resolve()),
        training_curves_png=training_curves_png,
        fold_summary_chart_png=fold_summary_chart_png,
        fold_summary_table_csv=fold_summary_table_csv,
        aggregate_predictions_csv=str(aggregate_predictions_csv.resolve()),
        aggregate_misclassifications_csv=str(aggregate_misclassifications_csv.resolve()),
        prediction_gallery_png=prediction_gallery_png,
        misclassification_gallery_png=misclassification_gallery_png,
        aggregate_confusion_counts_csv=aggregate_confusion_paths["counts_csv"],
        aggregate_confusion_normalized_csv=aggregate_confusion_paths["normalized_csv"],
        aggregate_confusion_counts_png=aggregate_confusion_paths["counts_png"],
        aggregate_confusion_normalized_png=aggregate_confusion_paths["normalized_png"],
        stage6_summary_json=str((directories["root"] / "stage6_summary.json").resolve()),
        stage6_summary_md=str((directories["root"] / "stage6_summary.md").resolve()),
    )

    overall_accuracy = float(predictions["is_correct"].mean()) if not predictions.empty else 0.0
    notes = [
        "Stage 6 visualizations were built from the completed cross-validation run without retraining.",
        "The derived local dataset is highly imbalanced, especially `joker` with only 5 total images.",
    ]
    if overall_accuracy < 0.90:
        notes.append("Because validation accuracy remains below 90%, Stage 8 will need an augmentation or improvement pass.")
    else:
        notes.append("The refreshed full-dataset baseline already clears the project's 90% validation-accuracy threshold.")

    result = {
        "stage": "Stage 6",
        "run_name": run_name,
        "stage5_summary_path": str(_discover_stage5_summary_json().resolve())
        if stage5_summary_path is None
        else str(Path(stage5_summary_path).resolve()),
        "dataset_root": stage5_summary["dataset_root"],
        "class_names": class_names,
        "num_validation_predictions": int(len(predictions)),
        "overall_validation_accuracy": overall_accuracy,
        "aggregate_metrics_from_stage5": stage5_summary["aggregate_metrics"],
        "artifacts": artifacts.to_dict(),
        "per_fold_confusion_artifacts": per_fold_confusion_paths,
        "notes": notes,
    }
    with Path(artifacts.stage6_summary_json).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _write_stage6_markdown(run_name=run_name, stage5_summary=stage5_summary, predictions=predictions, artifacts=artifacts)
    return result


if __name__ == "__main__":
    result = build_stage6_visualizations()
    print(json.dumps({"artifacts": result["artifacts"], "overall_validation_accuracy": result["overall_validation_accuracy"]}, indent=2))
