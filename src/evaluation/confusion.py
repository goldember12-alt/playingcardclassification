"""Confusion-matrix helpers for Stage 6 evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

from PIL import Image, ImageDraw, ImageFont

try:
    from sklearn.metrics import confusion_matrix
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    confusion_matrix = None


def _require_runtime() -> tuple[Any, Any]:
    if pd is None or confusion_matrix is None:
        raise ImportError("Stage 6 confusion-matrix helpers require pandas and scikit-learn.")
    return pd, confusion_matrix


def _font() -> Any:
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def compute_confusion_table(
    predictions: Any,
    class_names: list[str],
    normalize: str | None = None,
) -> Any:
    """Compute a labeled confusion-matrix table from prediction rows."""
    pandas, confusion_matrix_fn = _require_runtime()
    if not isinstance(predictions, pandas.DataFrame):
        predictions = pandas.DataFrame(predictions)

    matrix = confusion_matrix_fn(
        predictions["true_class_name"],
        predictions["predicted_class_name"],
        labels=class_names,
        normalize=normalize,
    )
    return pandas.DataFrame(matrix, index=class_names, columns=class_names)


def _cell_fill(value: float, max_value: float) -> tuple[int, int, int]:
    ratio = 0.0 if max_value <= 0 else min(max(value / max_value, 0.0), 1.0)
    base = 235
    blue = int(base - ratio * 150)
    return (blue, blue + 10 if blue + 10 <= 255 else 255, 255)


def plot_confusion_matrix(
    matrix: Any,
    class_names: list[str],
    title: str,
    normalize: bool = False,
    cell_size: int = 54,
) -> Any:
    """Render a confusion matrix to a PIL image with value annotations."""
    if hasattr(matrix, "to_numpy"):
        matrix_values = matrix.to_numpy()
    else:
        matrix_values = matrix

    left_margin = 145
    top_margin = 70
    right_margin = 30
    bottom_margin = 95
    width = left_margin + len(class_names) * cell_size + right_margin
    height = top_margin + len(class_names) * cell_size + bottom_margin
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()
    title_font = _font()

    draw.text((20, 20), title, fill="black", font=title_font)
    draw.text((left_margin + 120, height - 30), "Predicted label", fill="black", font=font)
    draw.text((20, top_margin - 25), "True label", fill="black", font=font)

    max_value = float(matrix_values.max()) if getattr(matrix_values, "size", 0) else 0.0
    value_format = "{:.2f}" if normalize else "{:.0f}"
    for row_index, true_name in enumerate(class_names):
        y0 = top_margin + row_index * cell_size
        y1 = y0 + cell_size
        label_y = y0 + (cell_size - _text_size(draw, true_name, font)[1]) // 2
        draw.text((10, label_y), true_name, fill="black", font=font)
        for column_index, predicted_name in enumerate(class_names):
            x0 = left_margin + column_index * cell_size
            x1 = x0 + cell_size
            value = float(matrix_values[row_index, column_index])
            draw.rectangle([x0, y0, x1, y1], fill=_cell_fill(value, max_value), outline="gray", width=1)
            text = value_format.format(value)
            text_width, text_height = _text_size(draw, text, font)
            draw.text(
                (x0 + (cell_size - text_width) // 2, y0 + (cell_size - text_height) // 2),
                text,
                fill="black",
                font=font,
            )

            if row_index == len(class_names) - 1:
                label_width, label_height = _text_size(draw, predicted_name, font)
                draw.text(
                    (x0 + (cell_size - label_width) // 2, y1 + 8),
                    predicted_name,
                    fill="black",
                    font=font,
                )

    return image


def save_confusion_outputs(
    predictions: Any,
    class_names: list[str],
    output_dir: str | Path,
    stem: str,
    title_prefix: str,
) -> dict[str, str]:
    """Save raw and normalized confusion tables plus PNG renderings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_table = compute_confusion_table(predictions=predictions, class_names=class_names, normalize=None)
    normalized_table = compute_confusion_table(predictions=predictions, class_names=class_names, normalize="true")

    raw_csv_path = output_dir / f"{stem}_counts.csv"
    normalized_csv_path = output_dir / f"{stem}_normalized.csv"
    raw_png_path = output_dir / f"{stem}_counts.png"
    normalized_png_path = output_dir / f"{stem}_normalized.png"

    raw_table.to_csv(raw_csv_path)
    normalized_table.to_csv(normalized_csv_path)

    raw_image = plot_confusion_matrix(
        matrix=raw_table,
        class_names=class_names,
        title=f"{title_prefix} Confusion Matrix (Counts)",
        normalize=False,
    )
    raw_image.save(raw_png_path)

    normalized_image = plot_confusion_matrix(
        matrix=normalized_table,
        class_names=class_names,
        title=f"{title_prefix} Confusion Matrix (Normalized)",
        normalize=True,
    )
    normalized_image.save(normalized_png_path)

    return {
        "counts_csv": str(raw_csv_path.resolve()),
        "normalized_csv": str(normalized_csv_path.resolve()),
        "counts_png": str(raw_png_path.resolve()),
        "normalized_png": str(normalized_png_path.resolve()),
    }
