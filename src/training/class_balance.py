"""Class-balancing helpers for imbalanced card-rank training."""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except ImportError:  # pragma: no cover - surfaced when called
    torch = None
    WeightedRandomSampler = None


def _require_torch() -> tuple[Any, Any]:
    if torch is None or WeightedRandomSampler is None:
        raise ImportError("torch is required for class balancing utilities.")
    return torch, WeightedRandomSampler


def compute_class_counts(dataset: Any, num_classes: int | None = None) -> list[int]:
    """Count per-class examples from a dataset backed by ImageRecord objects."""
    counts = Counter(int(record.class_index) for record in dataset.records)
    resolved_num_classes = int(num_classes if num_classes is not None else len(dataset.class_names))
    return [int(counts.get(class_index, 0)) for class_index in range(resolved_num_classes)]


def normalize_weights(weights: list[float]) -> list[float]:
    """Scale weights so their nonzero mean is 1.0 when possible."""
    nonzero = [value for value in weights if value > 0]
    if not nonzero:
        return weights
    mean_value = sum(nonzero) / len(nonzero)
    return [float(value / mean_value) if value > 0 else 0.0 for value in weights]


def build_class_weights(
    class_counts: list[int],
    strategy: str = "none",
    beta: float = 0.999,
) -> list[float] | None:
    """Build class-loss weights for the configured balancing strategy."""
    normalized = (strategy or "none").lower()
    if normalized == "none":
        return None

    weights: list[float] = []
    for count in class_counts:
        if count <= 0:
            weights.append(0.0)
            continue
        if normalized == "inverse_frequency":
            weights.append(1.0 / float(count))
        elif normalized == "sqrt_inverse_frequency":
            weights.append(1.0 / float(count) ** 0.5)
        elif normalized == "effective_num":
            effective = (1.0 - beta**count) / (1.0 - beta)
            weights.append(1.0 / effective)
        else:
            raise ValueError(
                f"Unsupported class weight strategy '{strategy}'. "
                "Expected one of: none, inverse_frequency, sqrt_inverse_frequency, effective_num."
            )
    return normalize_weights(weights)


def build_weighted_sampler(
    dataset: Any,
    strategy: str = "none",
    weight_power: float = 1.0,
    replacement: bool = True,
    generator: Any | None = None,
) -> Any | None:
    """Build a weighted sampler for the training set."""
    torch_module, sampler_cls = _require_torch()
    normalized = (strategy or "none").lower()
    if normalized == "none":
        return None
    if normalized != "balanced":
        raise ValueError("Unsupported sampling strategy. Expected one of: none, balanced.")

    class_counts = compute_class_counts(dataset)
    sample_weights: list[float] = []
    for record in dataset.records:
        count = max(class_counts[int(record.class_index)], 1)
        sample_weights.append(float((1.0 / count) ** weight_power))

    return sampler_cls(
        weights=torch_module.tensor(sample_weights, dtype=torch_module.double),
        num_samples=len(dataset),
        replacement=replacement,
        generator=generator,
    )
