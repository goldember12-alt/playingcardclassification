"""Prediction helpers for Stage 6 evaluation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    torch = None
    DataLoader = None

from src.models.classifier import build_model
from src.training.train_one_fold import build_fold_datasets


def _require_runtime() -> tuple[Any, Any, Any]:
    if pd is None or torch is None or DataLoader is None:
        raise ImportError("Stage 6 prediction helpers require pandas, torch, and torch DataLoader support.")
    return pd, torch, DataLoader


def _resolve_device(device: str | Any = "cpu") -> Any:
    _, torch_module, _ = _require_runtime()
    if hasattr(device, "type"):
        return device
    normalized = str(device).lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(normalized)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | Any = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a trained model checkpoint for inference without re-downloading pretrained weights."""
    _, torch_module, _ = _require_runtime()
    resolved_device = _resolve_device(device)
    checkpoint = torch_module.load(Path(checkpoint_path), map_location=resolved_device)
    model_spec = checkpoint["model_spec"]
    model, _ = build_model(
        model_name=model_spec["backbone"]["model_name"],
        num_classes=model_spec["classifier"]["num_classes"],
        class_names=checkpoint["class_names"],
        pretrained=False,
        freeze_backbone=model_spec["backbone"]["freeze_backbone"],
        unfreeze_from=model_spec["backbone"]["unfreeze_from"],
        classifier_hidden_dim=model_spec["classifier"]["hidden_dim"],
        classifier_dropout=model_spec["classifier"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()
    return model, checkpoint


def predict_batch(model: Any, inputs: Any, device: str | Any = "cpu") -> tuple[Any, Any]:
    """Run inference on one batch and return predicted indices and confidences."""
    _, torch_module, _ = _require_runtime()
    resolved_device = _resolve_device(device)
    with torch_module.inference_mode():
        logits = model(inputs.to(resolved_device))
        probabilities = torch_module.softmax(logits, dim=1)
        confidences, predicted_indices = probabilities.max(dim=1)
    return predicted_indices.detach().cpu(), confidences.detach().cpu()


def predict_dataset(
    model: Any,
    dataset: Any,
    fold_index: int,
    device: str | Any = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
) -> Any:
    """Predict every item in a dataset and return a notebook-friendly DataFrame."""
    pandas, _, dataloader_cls = _require_runtime()
    class_names = list(dataset.class_names)
    dataloader = dataloader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    rows: list[dict[str, Any]] = []
    offset = 0
    for inputs, targets in dataloader:
        batch_size_actual = int(targets.size(0))
        predicted_indices, confidences = predict_batch(model=model, inputs=inputs, device=device)
        batch_records = dataset.records[offset : offset + batch_size_actual]
        for batch_index, record in enumerate(batch_records):
            true_index = int(targets[batch_index].item())
            predicted_index = int(predicted_indices[batch_index].item())
            rows.append(
                {
                    "fold": int(fold_index),
                    "path": str(record.path.resolve()),
                    "true_class_index": true_index,
                    "true_class_name": class_names[true_index],
                    "predicted_class_index": predicted_index,
                    "predicted_class_name": class_names[predicted_index],
                    "confidence": float(confidences[batch_index].item()),
                    "is_correct": bool(predicted_index == true_index),
                }
            )
        offset += batch_size_actual

    return pandas.DataFrame(rows)


def collect_cross_validation_predictions(
    assignments: Any,
    per_fold_results: Any,
    class_names: list[str],
    image_size: int = 224,
    device: str | Any = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
) -> Any:
    """Collect validation predictions from every saved best-fold checkpoint."""
    pandas, _, _ = _require_runtime()
    if not isinstance(assignments, pandas.DataFrame):
        assignments = pandas.DataFrame(assignments)
    if not isinstance(per_fold_results, pandas.DataFrame):
        per_fold_results = pandas.DataFrame(per_fold_results)

    prediction_frames: list[Any] = []
    for row in per_fold_results.sort_values(by=["fold"], kind="stable").itertuples(index=False):
        model, checkpoint = load_checkpoint_model(checkpoint_path=row.checkpoint_path, device=device)
        fold_datasets = build_fold_datasets(
            assignments=assignments,
            class_names=class_names,
            fold_index=int(row.fold),
            image_size=image_size,
        )
        predictions = predict_dataset(
            model=model,
            dataset=fold_datasets["valid"],
            fold_index=int(row.fold),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        predictions["checkpoint_path"] = str(row.checkpoint_path)
        predictions["best_epoch"] = int(row.best_epoch)
        predictions["best_metric_value"] = float(row.best_metric_value)
        predictions["class_names"] = ",".join(checkpoint["class_names"])
        prediction_frames.append(predictions)

    if not prediction_frames:
        return pandas.DataFrame(
            columns=[
                "fold",
                "path",
                "true_class_index",
                "true_class_name",
                "predicted_class_index",
                "predicted_class_name",
                "confidence",
                "is_correct",
                "checkpoint_path",
                "best_epoch",
                "best_metric_value",
                "class_names",
            ]
        )

    return pandas.concat(prediction_frames, ignore_index=True).sort_values(
        by=["fold", "true_class_name", "path"],
        kind="stable",
    ).reset_index(drop=True)
