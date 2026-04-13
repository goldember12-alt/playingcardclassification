"""Loss builders for single-fold training."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    torch = None
    nn = None


def _require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise ImportError("torch is required for training losses. Install the project dependencies first.")
    return torch, nn


def build_loss(
    loss_name: str = "cross_entropy",
    class_weights: list[float] | tuple[float, ...] | None = None,
    label_smoothing: float = 0.0,
    device: str | Any | None = None,
) -> Any:
    """Build the project loss function for classification."""
    torch_module, nn_module = _require_torch()
    normalized_name = loss_name.lower()
    if normalized_name != "cross_entropy":
        raise ValueError("Stage 4 currently supports only `cross_entropy` loss.")

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch_module.tensor(class_weights, dtype=torch_module.float32)
        if device is not None:
            weight_tensor = weight_tensor.to(device)

    return nn_module.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
