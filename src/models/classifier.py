"""Classifier-head and full transfer-learning model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    nn = None

from src.models.backbone import BackboneSpec, build_backbone, summarize_backbone


@dataclass(frozen=True)
class ClassifierSpec:
    """Description of a classification head."""

    input_dim: int
    num_classes: int
    hidden_dim: int | None = None
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    """Description of the full transfer-learning model."""

    backbone: BackboneSpec
    classifier: ClassifierSpec


def _require_torch() -> Any:
    if nn is None:
        raise ImportError(
            "torch is required for classifier construction. Install the dependencies from requirements.txt first."
        )
    return nn


BaseModule = nn.Module if nn is not None else object


def resolve_num_classes(
    num_classes: int | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Resolve the model output dimension from explicit class names or a numeric count."""
    if class_names is not None:
        resolved = len(class_names)
        if num_classes is not None and num_classes != resolved:
            raise ValueError(
                f"Requested num_classes={num_classes}, but received {resolved} class names. "
                "These must match."
            )
        return resolved
    if num_classes is None:
        raise ValueError("Provide either `num_classes` or `class_names` when building a classifier.")
    return int(num_classes)


def build_classifier(
    input_dim: int,
    num_classes: int | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
) -> tuple[Any, ClassifierSpec]:
    """Build a classification head for transfer learning."""
    nn_module = _require_torch()
    resolved_num_classes = resolve_num_classes(num_classes=num_classes, class_names=class_names)

    if hidden_dim is None:
        layers: list[Any] = []
        if dropout > 0:
            layers.append(nn_module.Dropout(p=dropout))
        layers.append(nn_module.Linear(input_dim, resolved_num_classes))
        classifier = nn_module.Sequential(*layers)
    else:
        layers = [
            nn_module.Linear(input_dim, hidden_dim),
            nn_module.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn_module.Dropout(p=dropout))
        layers.append(nn_module.Linear(hidden_dim, resolved_num_classes))
        classifier = nn_module.Sequential(*layers)

    spec = ClassifierSpec(
        input_dim=input_dim,
        num_classes=resolved_num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    return classifier, spec


class TransferLearningModel(BaseModule):
    """Simple backbone-plus-head image classifier."""

    def __init__(self, backbone: Any, classifier: Any) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, inputs: Any) -> Any:
        features = self.backbone(inputs)
        if hasattr(features, "ndim") and features.ndim > 2:
            features = features.flatten(start_dim=1)
        return self.classifier(features)


def build_model(
    model_name: str = "resnet50",
    num_classes: int | None = 14,
    class_names: list[str] | tuple[str, ...] | None = None,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    unfreeze_from: str | None = None,
    classifier_hidden_dim: int | None = None,
    classifier_dropout: float = 0.0,
) -> tuple[Any, ModelSpec]:
    """Build the full baseline transfer-learning model."""
    _require_torch()
    backbone, backbone_spec = build_backbone(
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_from=unfreeze_from,
    )
    classifier, classifier_spec = build_classifier(
        input_dim=backbone_spec.feature_dim,
        num_classes=num_classes,
        class_names=class_names,
        hidden_dim=classifier_hidden_dim,
        dropout=classifier_dropout,
    )
    model = TransferLearningModel(backbone=backbone, classifier=classifier)
    return model, ModelSpec(backbone=backbone_spec, classifier=classifier_spec)


def summarize_model(model: Any, spec: ModelSpec) -> dict[str, Any]:
    """Return a compact summary of the full model."""
    backbone_summary = summarize_backbone(model.backbone, spec.backbone)
    classifier_parameters = sum(parameter.numel() for parameter in model.classifier.parameters())
    classifier_trainable = sum(
        parameter.numel() for parameter in model.classifier.parameters() if parameter.requires_grad
    )
    return {
        "model_name": spec.backbone.model_name,
        "num_classes": spec.classifier.num_classes,
        "classifier_input_dim": spec.classifier.input_dim,
        "classifier_hidden_dim": spec.classifier.hidden_dim,
        "classifier_dropout": spec.classifier.dropout,
        "backbone": backbone_summary,
        "classifier_parameters": classifier_parameters,
        "classifier_trainable_parameters": classifier_trainable,
    }
