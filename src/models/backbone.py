"""Pretrained backbone construction utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    nn = None

try:
    from torchvision import models
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        EfficientNet_B0_Weights,
        ResNet18_Weights,
        ResNet50_Weights,
    )
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    models = None
    ResNet50_Weights = None
    ResNet18_Weights = None
    ConvNeXt_Tiny_Weights = None
    EfficientNet_B0_Weights = None


SUPPORTED_BACKBONES = ("resnet18", "resnet50", "convnext_tiny", "efficientnet_b0")
BACKBONE_STAGE_ORDERS = {
    "resnet18": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "resnet50": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "convnext_tiny": ("features.0", "features.1", "features.2", "features.3", "features.4", "features.5", "features.6", "features.7"),
    "efficientnet_b0": (
        "features.0",
        "features.1",
        "features.2",
        "features.3",
        "features.4",
        "features.5",
        "features.6",
        "features.7",
        "features.8",
    ),
}
DEFAULT_TORCH_HOME = Path(__file__).resolve().parents[2] / ".tmp" / "torch"


@dataclass(frozen=True)
class BackboneSpec:
    """Description of a constructed pretrained backbone."""

    model_name: str
    feature_dim: int
    pretrained: bool
    freeze_backbone: bool
    unfreeze_from: str | None = None


def _require_torchvision() -> tuple[Any, Any]:
    if nn is None or models is None:
        raise ImportError(
            "torch and torchvision are required for backbone construction. "
            "Install the dependencies from requirements.txt before running model code."
        )
    return nn, models


def _resolve_weights(model_name: str, pretrained: bool) -> Any:
    if not pretrained:
        return None
    weight_map = {
        "resnet18": ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else "IMAGENET1K_V1",
        "resnet50": ResNet50_Weights.DEFAULT if ResNet50_Weights is not None else "IMAGENET1K_V2",
        "convnext_tiny": ConvNeXt_Tiny_Weights.DEFAULT if ConvNeXt_Tiny_Weights is not None else "IMAGENET1K_V1",
        "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT if EfficientNet_B0_Weights is not None else "IMAGENET1K_V1",
    }
    return weight_map[model_name]


def ensure_torch_home() -> None:
    """Point torchvision model downloads at a writable cache directory."""
    torch_home = Path(os.environ.get("TORCH_HOME", DEFAULT_TORCH_HOME)).resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


def set_requires_grad(module: Any, requires_grad: bool) -> None:
    """Set ``requires_grad`` for every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _resolve_stage_order(model_name: str) -> tuple[str, ...]:
    if model_name not in BACKBONE_STAGE_ORDERS:
        raise ValueError(f"Unsupported backbone '{model_name}'. Supported backbones: {list(SUPPORTED_BACKBONES)}")
    return BACKBONE_STAGE_ORDERS[model_name]


def configure_backbone_trainability(
    backbone: Any,
    model_name: str,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,
) -> None:
    """Freeze the backbone by default and optionally unfreeze upper stages."""
    _require_torchvision()
    if not freeze_backbone:
        set_requires_grad(backbone, True)
        return

    set_requires_grad(backbone, False)
    if unfreeze_from is None:
        return

    stage_order = _resolve_stage_order(model_name)
    if unfreeze_from not in stage_order:
        raise ValueError(
            f"Unsupported `unfreeze_from` value '{unfreeze_from}' for backbone '{model_name}'. "
            f"Expected one of: {list(stage_order)}"
        )

    should_unfreeze = False
    for stage_name in stage_order:
        if stage_name == unfreeze_from:
            should_unfreeze = True
        if should_unfreeze:
            set_requires_grad(backbone.get_submodule(stage_name), True)


def _build_resnet18(pretrained: bool) -> tuple[Any, int]:
    _, torchvision_models = _require_torchvision()
    if ResNet18_Weights is None:
        backbone = torchvision_models.resnet18(pretrained=pretrained)
    else:
        backbone = torchvision_models.resnet18(weights=_resolve_weights("resnet18", pretrained))
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_resnet50(pretrained: bool) -> tuple[Any, int]:
    _, torchvision_models = _require_torchvision()
    if ResNet50_Weights is None:
        backbone = torchvision_models.resnet50(pretrained=pretrained)
    else:
        backbone = torchvision_models.resnet50(weights=_resolve_weights("resnet50", pretrained))
    feature_dim = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _build_convnext_tiny(pretrained: bool) -> tuple[Any, int]:
    _, torchvision_models = _require_torchvision()
    weights = _resolve_weights("convnext_tiny", pretrained)
    backbone = torchvision_models.convnext_tiny(weights=weights)
    feature_dim = int(backbone.classifier[-1].in_features)
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


def _build_efficientnet_b0(pretrained: bool) -> tuple[Any, int]:
    _, torchvision_models = _require_torchvision()
    weights = _resolve_weights("efficientnet_b0", pretrained)
    backbone = torchvision_models.efficientnet_b0(weights=weights)
    feature_dim = int(backbone.classifier[-1].in_features)
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


def build_backbone(
    model_name: str = "resnet50",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,
) -> tuple[Any, BackboneSpec]:
    """Build a pretrained backbone with its classification head removed."""
    _require_torchvision()
    ensure_torch_home()
    normalized_name = model_name.lower()
    if normalized_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{normalized_name}'. Supported backbones: {list(SUPPORTED_BACKBONES)}"
        )

    builders = {
        "resnet18": _build_resnet18,
        "resnet50": _build_resnet50,
        "convnext_tiny": _build_convnext_tiny,
        "efficientnet_b0": _build_efficientnet_b0,
    }
    backbone, feature_dim = builders[normalized_name](pretrained=pretrained)
    configure_backbone_trainability(
        backbone=backbone,
        model_name=normalized_name,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    )

    spec = BackboneSpec(
        model_name=normalized_name,
        feature_dim=feature_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    )
    return backbone, spec


def count_trainable_parameters(module: Any) -> int:
    """Count the number of trainable parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def count_total_parameters(module: Any) -> int:
    """Count the total number of parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def summarize_backbone(module: Any, spec: BackboneSpec) -> dict[str, Any]:
    """Return a compact summary of backbone construction and trainability."""
    return {
        "model_name": spec.model_name,
        "feature_dim": spec.feature_dim,
        "pretrained": spec.pretrained,
        "freeze_backbone": spec.freeze_backbone,
        "unfreeze_from": spec.unfreeze_from,
        "trainable_parameters": count_trainable_parameters(module),
        "total_parameters": count_total_parameters(module),
    }
