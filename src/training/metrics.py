"""Metric helpers for one-fold classification training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    torch = None


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("torch is required for training metrics. Install the project dependencies first.")
    return torch


@dataclass
class RunningClassificationMetrics:
    """Mutable running totals for an epoch."""

    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the running metrics into a plain dictionary."""
        return asdict(self)


def update_running_metrics(
    running: RunningClassificationMetrics,
    batch_loss: float,
    logits: Any,
    targets: Any,
) -> RunningClassificationMetrics:
    """Update running loss and accuracy totals for one batch."""
    torch_module = _require_torch()
    batch_size = int(targets.size(0))
    predictions = torch_module.argmax(logits, dim=1)
    running.total_loss += float(batch_loss) * batch_size
    running.total_correct += int((predictions == targets).sum().item())
    running.total_samples += batch_size
    return running


def finalize_running_metrics(running: RunningClassificationMetrics) -> dict[str, float]:
    """Convert running totals into scalar epoch metrics."""
    if running.total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": running.total_loss / running.total_samples,
        "accuracy": running.total_correct / running.total_samples,
    }


def is_better_metric(candidate: float, best: float | None, mode: str = "max") -> bool:
    """Return whether a candidate metric improves upon the previous best value."""
    if best is None:
        return True
    normalized_mode = mode.lower()
    if normalized_mode == "max":
        return candidate > best
    if normalized_mode == "min":
        return candidate < best
    raise ValueError("Metric comparison mode must be either `max` or `min`.")
