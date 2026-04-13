"""Single-fold training utilities."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

try:
    import torch
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    torch = None
    Adam = None
    AdamW = None
    CosineAnnealingLR = None
    ReduceLROnPlateau = None
    DataLoader = None

from src.data.dataset import CardImageDataset, ImageRecord
from src.data.folds import build_fold_inventory
from src.data.transforms import build_targeted_minority_transform, build_transforms
from src.models.classifier import build_model, summarize_model
from src.training.class_balance import build_class_weights, build_weighted_sampler, compute_class_counts
from src.training.losses import build_loss
from src.training.metrics import (
    RunningClassificationMetrics,
    finalize_running_metrics,
    update_running_metrics,
)
from src.utils.paths import DERIVED_RANK_DATASET_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR


@dataclass(frozen=True)
class FoldTrainingConfig:
    """Configuration for a single-fold training run."""

    fold_index: int = 0
    run_name: str = "baseline"
    num_folds: int = 5
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    device: str = "auto"
    image_size: int = 224
    num_classes: int | None = 14
    model_name: str = "resnet50"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    unfreeze_from: str | None = None
    classifier_hidden_dim: int | None = None
    classifier_dropout: float = 0.0
    optimizer_name: str = "adam"
    backbone_learning_rate: float | None = None
    loss_name: str = "cross_entropy"
    class_weight_strategy: str = "none"
    class_weight_beta: float = 0.999
    sampling_strategy: str = "none"
    sampling_weight_power: float = 1.0
    scheduler_name: str = "none"
    scheduler_min_lr: float = 1e-6
    scheduler_plateau_factor: float = 0.5
    scheduler_plateau_patience: int = 2
    label_smoothing: float = 0.0
    use_augmentation: bool = False
    augmentation_profile: str | None = None
    targeted_augmentation_profile: str | None = None
    targeted_augmentation_max_class_count: int = 32
    monitor_metric: str = "val_accuracy"
    monitor_mode: str = "max"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    random_seed: int = 42
    raw_data_dir: str = str(PROCESSED_DATA_DIR)
    dataset_name: str | None = DERIVED_RANK_DATASET_DIR.name
    dataset_root: str | None = None
    strategy: str = "bottleneck"


@dataclass(frozen=True)
class FoldTrainingArtifacts:
    """Artifact locations produced by one-fold training."""

    checkpoint_path: str
    metrics_csv_path: str
    summary_json_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact paths into a plain dictionary."""
        return asdict(self)


def _require_runtime() -> tuple[Any, Any, Any]:
    if (
        torch is None
        or Adam is None
        or AdamW is None
        or CosineAnnealingLR is None
        or ReduceLROnPlateau is None
        or DataLoader is None
        or pd is None
    ):
        raise ImportError(
            "Stage 4 training requires torch, pandas, and related runtime dependencies from requirements.txt."
        )
    return torch, DataLoader, pd


def resolve_device(device_preference: str = "auto") -> Any:
    """Resolve the execution device, preferring CUDA only when available and requested."""
    torch_module, _, _ = _require_runtime()
    normalized = device_preference.lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        return torch_module.device("cuda")
    if normalized == "cpu":
        return torch_module.device("cpu")
    return torch_module.device(device_preference)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible fold training."""
    torch_module, _, _ = _require_runtime()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _records_from_assignments(assignments: Any) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for row in assignments.itertuples(index=False):
        records.append(
            ImageRecord(
                path=Path(row.path),
                class_name=str(row.class_name),
                class_index=int(row.class_index),
                split_name=str(row.split_name),
            )
        )
    return records


def build_fold_datasets(
    assignments: Any,
    class_names: list[str],
    fold_index: int,
    image_size: int = 224,
    use_augmentation: bool = False,
    augmentation_profile: str | None = None,
    targeted_augmentation_profile: str | None = None,
    targeted_augmentation_max_class_count: int = 32,
) -> dict[str, CardImageDataset]:
    """Build train/validation datasets from Stage 2 fold assignments."""
    _, _, pandas = _require_runtime()
    if not isinstance(assignments, pandas.DataFrame):
        assignments = pandas.DataFrame(assignments)

    required_columns = {"path", "class_name", "class_index", "split_name", "fold", "fold_role"}
    missing = sorted(required_columns - set(assignments.columns))
    if missing:
        raise ValueError(f"Fold assignments are missing required columns: {missing}")

    cv_pool = assignments.loc[assignments["fold_role"] == "cv_pool"].copy()
    if cv_pool.empty:
        raise ValueError("No cross-validation pool records are available for training.")

    train_rows = cv_pool.loc[cv_pool["fold"] != fold_index].copy()
    val_rows = cv_pool.loc[cv_pool["fold"] == fold_index].copy()
    if train_rows.empty or val_rows.empty:
        raise ValueError(
            f"Fold {fold_index} is not usable: train rows={len(train_rows)}, validation rows={len(val_rows)}."
        )

    transform_map = build_transforms(
        image_size=image_size,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
    )
    train_class_counts = train_rows["class_name"].value_counts().to_dict()
    minority_class_transforms: dict[str, Any] = {}
    if targeted_augmentation_profile:
        minority_transform = build_targeted_minority_transform(
            image_size=image_size,
            augmentation_profile=targeted_augmentation_profile,
        )
        minority_class_transforms = {
            class_name: minority_transform
            for class_name, count in train_class_counts.items()
            if int(count) <= targeted_augmentation_max_class_count
        }
    return {
        "train": CardImageDataset(
            records=_records_from_assignments(train_rows),
            class_names=class_names,
            transform=transform_map["train"],
            transform_overrides_by_class=minority_class_transforms,
        ),
        "valid": CardImageDataset(
            records=_records_from_assignments(val_rows),
            class_names=class_names,
            transform=transform_map["valid"],
        ),
    }


def _build_dataloaders(
    datasets: dict[str, CardImageDataset],
    batch_size: int,
    num_workers: int,
    device: Any,
    random_seed: int,
    sampling_strategy: str = "none",
    sampling_weight_power: float = 1.0,
) -> dict[str, Any]:
    torch_module, dataloader_cls, _ = _require_runtime()
    generator = torch_module.Generator()
    generator.manual_seed(random_seed)
    pin_memory = device.type == "cuda"
    train_sampler = build_weighted_sampler(
        dataset=datasets["train"],
        strategy=sampling_strategy,
        weight_power=sampling_weight_power,
        generator=generator,
    )
    return {
        "train": dataloader_cls(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        "valid": dataloader_cls(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def _build_optimizer(
    model: Any,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    backbone_learning_rate: float | None = None,
) -> Any:
    normalized = optimizer_name.lower()
    optimizer_cls = {"adam": Adam, "adamw": AdamW}.get(normalized)
    if optimizer_cls is None:
        raise ValueError("Unsupported optimizer_name. Expected one of: adam, adamw.")

    classifier_parameters = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    if not classifier_parameters and not backbone_parameters:
        raise ValueError("No trainable parameters were found for optimizer construction.")

    if backbone_learning_rate is None or not backbone_parameters:
        trainable_parameters = classifier_parameters + backbone_parameters
        return optimizer_cls(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)

    parameter_groups: list[dict[str, Any]] = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})
    if classifier_parameters:
        parameter_groups.append({"params": classifier_parameters, "lr": learning_rate})
    return optimizer_cls(parameter_groups, weight_decay=weight_decay)


def _build_scheduler(
    optimizer: Any,
    scheduler_name: str,
    num_epochs: int,
    scheduler_min_lr: float,
    scheduler_plateau_factor: float = 0.5,
    scheduler_plateau_patience: int = 2,
) -> Any | None:
    normalized = scheduler_name.lower()
    if normalized == "none":
        return None
    if normalized == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=scheduler_min_lr)
    if normalized == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_plateau_factor,
            patience=scheduler_plateau_patience,
            min_lr=scheduler_min_lr,
        )
    raise ValueError("Unsupported scheduler_name. Expected one of: none, cosine, plateau.")


def run_training_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""
    torch_module, _, _ = _require_runtime()
    model.train()
    running = RunningClassificationMetrics()
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        update_running_metrics(running, float(loss.item()), logits.detach(), targets.detach())

    metrics = finalize_running_metrics(running)
    torch_module.cuda.empty_cache() if device.type == "cuda" else None
    return metrics


def run_validation_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Run one validation epoch and return aggregate metrics."""
    torch_module, _, _ = _require_runtime()
    model.eval()
    running = RunningClassificationMetrics()
    with torch_module.inference_mode():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            update_running_metrics(running, float(loss.item()), logits, targets)
    return finalize_running_metrics(running)


def _resolve_monitor_value(epoch_metrics: dict[str, Any], monitor_metric: str) -> float:
    if monitor_metric not in epoch_metrics:
        raise ValueError(f"Monitor metric '{monitor_metric}' was not found in epoch metrics: {sorted(epoch_metrics)}")
    return float(epoch_metrics[monitor_metric])


def _has_improved(candidate: float, best: float | None, mode: str = "max", min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    normalized_mode = mode.lower()
    if normalized_mode == "max":
        return candidate > (best + min_delta)
    if normalized_mode == "min":
        return candidate < (best - min_delta)
    raise ValueError("Metric comparison mode must be either `max` or `min`.")


def _artifact_paths(run_name: str, fold_index: int) -> FoldTrainingArtifacts:
    checkpoint_dir = OUTPUTS_DIR / "checkpoints"
    log_dir = OUTPUTS_DIR / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_name}_fold_{fold_index:02d}"
    return FoldTrainingArtifacts(
        checkpoint_path=str((checkpoint_dir / f"{stem}_best.pt").resolve()),
        metrics_csv_path=str((log_dir / f"{stem}_metrics.csv").resolve()),
        summary_json_path=str((log_dir / f"{stem}_summary.json").resolve()),
    )


def _save_training_outputs(
    history: Any,
    summary: dict[str, Any],
    artifacts: FoldTrainingArtifacts,
) -> None:
    _, _, pandas = _require_runtime()
    if not isinstance(history, pandas.DataFrame):
        history = pandas.DataFrame(history)
    history.to_csv(artifacts.metrics_csv_path, index=False)
    with open(artifacts.summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def train_one_fold(
    fold_index: int = 0,
    assignments: Any | None = None,
    run_name: str = "baseline",
    num_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    device: str = "auto",
    num_folds: int = 5,
    image_size: int = 224,
    random_seed: int = 42,
    raw_data_dir: str | Path = PROCESSED_DATA_DIR,
    dataset_name: str | None = DERIVED_RANK_DATASET_DIR.name,
    dataset_root: str | Path | None = None,
    model_name: str = "resnet50",
    num_classes: int | None = 14,
    class_names: list[str] | tuple[str, ...] | None = None,
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
) -> tuple[Any, dict[str, Any]]:
    """Train and validate one fold, saving a best checkpoint and per-epoch metrics."""
    torch_module, _, pandas = _require_runtime()
    set_global_seed(random_seed)
    resolved_device = resolve_device(device)

    fold_summary = None
    if assignments is None:
        assignments, fold_summary = build_fold_inventory(
            dataset_root=dataset_root,
            raw_data_dir=raw_data_dir,
            dataset_name=dataset_name,
            n_splits=num_folds,
            random_seed=random_seed,
        )
    elif not isinstance(assignments, pandas.DataFrame):
        assignments = pandas.DataFrame(assignments)

    if class_names is None:
        class_mapping = (
            assignments[["class_name", "class_index"]]
            .drop_duplicates()
            .sort_values(by=["class_index", "class_name"], kind="stable")
        )
        class_names = class_mapping["class_name"].tolist()

    datasets = build_fold_datasets(
        assignments=assignments,
        class_names=list(class_names),
        fold_index=fold_index,
        image_size=image_size,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        targeted_augmentation_profile=targeted_augmentation_profile,
        targeted_augmentation_max_class_count=targeted_augmentation_max_class_count,
    )
    dataloaders = _build_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
        random_seed=random_seed,
        sampling_strategy=sampling_strategy,
        sampling_weight_power=sampling_weight_power,
    )
    train_class_counts = compute_class_counts(datasets["train"], num_classes=len(class_names))
    loss_class_weights = build_class_weights(
        class_counts=train_class_counts,
        strategy=class_weight_strategy,
        beta=class_weight_beta,
    )

    model, model_spec = build_model(
        model_name=model_name,
        num_classes=num_classes,
        class_names=class_names,
        pretrained=pretrained_backbone,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
        classifier_hidden_dim=classifier_hidden_dim,
        classifier_dropout=classifier_dropout,
    )
    model = model.to(resolved_device)
    criterion = build_loss(
        loss_name=loss_name,
        class_weights=loss_class_weights,
        label_smoothing=label_smoothing,
        device=resolved_device,
    )
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        backbone_learning_rate=backbone_learning_rate,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        num_epochs=num_epochs,
        scheduler_min_lr=scheduler_min_lr,
        scheduler_plateau_factor=scheduler_plateau_factor,
        scheduler_plateau_patience=scheduler_plateau_patience,
    )

    artifacts = _artifact_paths(run_name=run_name, fold_index=fold_index)
    history_rows: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_metrics = run_training_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=resolved_device,
        )
        val_metrics = run_validation_epoch(
            model=model,
            dataloader=dataloaders["valid"],
            criterion=criterion,
            device=resolved_device,
        )
        elapsed_seconds = time.time() - epoch_start

        epoch_metrics = {
            "epoch": epoch,
            "classifier_lr": float(optimizer.param_groups[-1]["lr"]),
            "backbone_lr": float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 1 else None,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": elapsed_seconds,
        }
        history_rows.append(epoch_metrics)

        monitor_value = _resolve_monitor_value(epoch_metrics, monitor_metric)
        has_improved = _has_improved(
            monitor_value,
            best_metric,
            mode=monitor_mode,
            min_delta=early_stopping_min_delta,
        )
        if has_improved:
            best_metric = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = {
                "epoch": epoch,
                "fold_index": fold_index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_spec": asdict(model_spec),
                "train_config": asdict(
                    FoldTrainingConfig(
                        fold_index=fold_index,
                        run_name=run_name,
                        num_folds=num_folds,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_workers=num_workers,
                        device=device,
                        image_size=image_size,
                        num_classes=num_classes,
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
                        random_seed=random_seed,
                        raw_data_dir=str(raw_data_dir),
                        dataset_name=dataset_name,
                        dataset_root=str(dataset_root) if dataset_root is not None else None,
                        strategy="bottleneck" if freeze_backbone else "fine_tune",
                    )
                ),
                "best_metric_name": monitor_metric,
                "best_metric_value": best_metric,
                "class_names": list(class_names),
            }
            torch_module.save(checkpoint_payload, artifacts.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if scheduler_name.lower() == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            break

    history = pandas.DataFrame(history_rows)
    model_summary = summarize_model(model, model_spec)
    summary = {
        "fold_index": fold_index,
        "run_name": run_name,
        "device": str(resolved_device),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "backbone_learning_rate": backbone_learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "image_size": image_size,
        "class_weight_strategy": class_weight_strategy,
        "class_weight_beta": class_weight_beta,
        "class_weights": loss_class_weights,
        "sampling_strategy": sampling_strategy,
        "sampling_weight_power": sampling_weight_power,
        "scheduler_name": scheduler_name,
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_plateau_factor": scheduler_plateau_factor,
        "scheduler_plateau_patience": scheduler_plateau_patience,
        "train_class_counts": train_class_counts,
        "use_augmentation": use_augmentation,
        "augmentation_profile": augmentation_profile,
        "targeted_augmentation_profile": targeted_augmentation_profile,
        "targeted_augmentation_max_class_count": targeted_augmentation_max_class_count,
        "epochs_completed": int(len(history_rows)),
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["valid"]),
        "best_epoch": best_epoch,
        "best_metric_name": monitor_metric,
        "best_metric_value": best_metric,
        "class_names": list(class_names),
        "model_summary": model_summary,
        "artifacts": artifacts.to_dict(),
        "fold_summary": fold_summary.to_dict() if fold_summary is not None else None,
    }
    _save_training_outputs(history=history, summary=summary, artifacts=artifacts)
    return history, summary
