"""Reusable image transforms for training, validation, and inference."""

from __future__ import annotations

from typing import Any

try:
    from torchvision import transforms as T
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    T = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torchvision() -> Any:
    if T is None:
        raise ImportError(
            "torchvision is required to build image transforms. Install dependencies from requirements.txt first."
        )
    return T


def build_eval_transform(image_size: int = 224, normalize: bool = True) -> Any:
    """Build the deterministic evaluation transform."""
    transforms = _require_torchvision()
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_train_transform(
    image_size: int = 224,
    normalize: bool = True,
    use_augmentation: bool = False,
    augmentation_profile: str | None = None,
) -> Any:
    """Build the training transform, optionally enabling a controlled augmentation profile."""
    transforms = _require_torchvision()
    if not use_augmentation:
        return build_eval_transform(image_size=image_size, normalize=normalize)

    profile_name = (augmentation_profile or "stage8_cards").lower()
    if profile_name == "stage8_cards":
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.20),
            transforms.ToTensor(),
        ]
    elif profile_name == "notebook_cards":
        # Keep the whole card visible and use only the light augmentation patterns
        # that matched the strongest local ResNet-18 reference notebook.
        steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"Unsupported augmentation profile '{augmentation_profile}'.")

    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_targeted_minority_transform(
    image_size: int = 224,
    normalize: bool = True,
    augmentation_profile: str | None = None,
) -> Any:
    """Build a stronger augmentation path intended only for minority classes."""
    transforms = _require_torchvision()
    profile_name = (augmentation_profile or "minority_cards").lower()
    if profile_name != "minority_cards":
        raise ValueError(f"Unsupported targeted augmentation profile '{augmentation_profile}'.")

    steps = [
        transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomRotation(degrees=14),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.03),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.30),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_inference_transform(image_size: int = 224, normalize: bool = True) -> Any:
    """Build the inference transform."""
    return build_eval_transform(image_size=image_size, normalize=normalize)


def build_transforms(
    image_size: int = 224,
    normalize: bool = True,
    include_inference: bool = True,
    use_augmentation: bool = False,
    augmentation_profile: str | None = None,
) -> dict[str, Any]:
    """Build the standard project transform dictionary."""
    transform_map = {
        "train": build_train_transform(
            image_size=image_size,
            normalize=normalize,
            use_augmentation=use_augmentation,
            augmentation_profile=augmentation_profile,
        ),
        "valid": build_eval_transform(image_size=image_size, normalize=normalize),
        "test": build_eval_transform(image_size=image_size, normalize=normalize),
        "all": build_eval_transform(image_size=image_size, normalize=normalize),
    }
    if include_inference:
        transform_map["inference"] = build_inference_transform(image_size=image_size, normalize=normalize)
    transform_map["minority_targeted"] = build_targeted_minority_transform(
        image_size=image_size,
        normalize=normalize,
    )
    return transform_map
