"""Stage 7 feature-map extraction and rendering utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - surfaced at runtime
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - surfaced at runtime
    torch = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - surfaced at runtime
    matplotlib = None
    plt = None

from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_inference_transform
from src.models.classifier import build_model
from src.utils.paths import FEATURE_MAPS_OUTPUT_DIR, OUTPUTS_DIR


FACE_CARD_CLASSES = ("jack", "queen", "king")
NUMBER_CARD_CLASSES = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class FeatureMapExample:
    """One selected image example for feature-map visualization."""

    group_name: str
    class_name: str
    path: str
    fold: int
    confidence: float
    predicted_class_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage7Artifacts:
    """Paths produced by the Stage 7 feature-map workflow."""

    output_root: str
    selected_examples_csv: str
    summary_json: str
    summary_md: str
    overview_png: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_runtime() -> tuple[Any, Any, Any]:
    if pd is None or torch is None or plt is None:
        raise ImportError("Stage 7 requires pandas, torch, and matplotlib.")
    return pd, torch, plt


def _discover_stage5_summary_json() -> Path:
    candidates = sorted((OUTPUTS_DIR / "logs").glob("*_aggregate_summary.json"))
    if not candidates:
        raise FileNotFoundError("No aggregate summary JSON was found under outputs/logs/.")

    canonical_candidates: list[tuple[float, float, str, Path]] = []
    for path in candidates:
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

    non_smoke = [path for path in candidates if "smoke" not in path.name]
    refresh = [path for path in non_smoke if "refresh" in path.name]
    if refresh:
        return refresh[-1]
    preferred = [path for path in non_smoke if "baseline" in path.name]
    if preferred:
        return preferred[-1]
    if non_smoke:
        return non_smoke[-1]
    return candidates[-1]


def _discover_stage6_predictions_csv(run_name: str) -> Path:
    path = OUTPUTS_DIR / "visualizations" / run_name / "predictions" / "aggregate_validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage 6 prediction CSV was not found: {path}")
    return path


def _load_stage5_summary(stage5_summary_path: str | Path | None = None) -> dict[str, Any]:
    summary_path = Path(stage5_summary_path) if stage5_summary_path else _discover_stage5_summary_json()
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_balance_note(stage5_summary: dict[str, Any]) -> str:
    fold_summary = stage5_summary.get("fold_summary", {})
    counts = {
        str(class_name): int(count)
        for class_name, count in dict(fold_summary.get("pool_class_counts", {})).items()
    }
    if not counts:
        return "The dataset remains imbalanced, but class-count details were not available in the saved run summary."

    smallest_class_name, smallest_class_count = min(counts.items(), key=lambda item: (item[1], item[0]))
    validation_counts = [
        int(class_counts.get(smallest_class_name, 0))
        for class_counts in dict(fold_summary.get("validation_class_counts", {})).values()
    ]
    if validation_counts:
        min_count = min(validation_counts)
        max_count = max(validation_counts)
        if min_count == max_count:
            fold_detail = f"Each validation fold includes {min_count} `{smallest_class_name}` examples."
        else:
            fold_detail = (
                f"Validation folds include between {min_count} and {max_count} "
                f"`{smallest_class_name}` examples."
            )
    else:
        fold_detail = "Per-fold validation counts were not available in the saved summary."

    return (
        f"The dataset remains imbalanced; `{smallest_class_name}` is the smallest class with "
        f"{smallest_class_count} images. {fold_detail}"
    )


def _best_checkpoint_row(stage5_summary: dict[str, Any]) -> dict[str, Any]:
    pandas, _, _ = _require_runtime()
    per_fold = pandas.DataFrame(stage5_summary["per_fold_results"])
    best_row = per_fold.sort_values(
        by=["best_metric_value", "fold"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return best_row.to_dict()


def _resolve_device(device: str | Any = "cpu") -> Any:
    _, torch_module, _ = _require_runtime()
    if hasattr(device, "type"):
        return device
    normalized = str(device).lower()
    if normalized == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(normalized)


def load_feature_map_model(
    checkpoint_path: str | Path,
    device: str | Any = "cpu",
) -> tuple[Any, dict[str, Any], Any]:
    """Load a checkpointed model for feature-map extraction."""
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
    return model, checkpoint, resolved_device


def resolve_module(root_module: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path on a model."""
    module = root_module
    for part in dotted_path.split("."):
        if not hasattr(module, part):
            raise AttributeError(f"Model path '{dotted_path}' could not be resolved at '{part}'.")
        module = getattr(module, part)
    return module


def load_image_tensor(image_path: str | Path, image_size: int) -> tuple[Any, Any]:
    """Load one image and convert it into the standard inference tensor."""
    transform = build_inference_transform(image_size=image_size, normalize=True)
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        display_image = rgb_image.copy()
        tensor = transform(rgb_image).unsqueeze(0)
    return tensor, display_image


def _denormalize_tensor(image_tensor: Any) -> Any:
    """Convert a normalized tensor back to displayable RGB numpy format."""
    _, torch_module, _ = _require_runtime()
    tensor = image_tensor.detach().cpu().clone().squeeze(0)
    mean = torch_module.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch_module.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def extract_feature_maps(
    model: Any,
    image_tensor: Any,
    layer_names: list[str] | tuple[str, ...],
    device: str | Any = "cpu",
) -> tuple[dict[str, Any], Any]:
    """Run one forward pass and capture activations from the requested layers."""
    _, torch_module, _ = _require_runtime()
    resolved_device = _resolve_device(device)
    activations: dict[str, Any] = {}
    handles: list[Any] = []

    def make_hook(layer_name: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations[layer_name] = output.detach().cpu()

        return _hook

    for layer_name in layer_names:
        handles.append(resolve_module(model, layer_name).register_forward_hook(make_hook(layer_name)))

    with torch_module.inference_mode():
        logits = model(image_tensor.to(resolved_device))
        probabilities = torch_module.softmax(logits, dim=1).detach().cpu()

    for handle in handles:
        handle.remove()

    return activations, probabilities


def select_top_feature_channels(feature_map_tensor: Any, top_k: int = 8) -> Any:
    """Select the top activated channels from a single feature-map tensor."""
    _, torch_module, _ = _require_runtime()
    if feature_map_tensor.ndim != 4 or feature_map_tensor.size(0) != 1:
        raise ValueError("Expected a feature-map tensor of shape [1, C, H, W].")
    activation_strength = feature_map_tensor.abs().mean(dim=(2, 3)).squeeze(0)
    top_indices = torch_module.topk(activation_strength, k=min(top_k, activation_strength.numel())).indices.tolist()
    selected = feature_map_tensor.squeeze(0)[top_indices]
    return selected, top_indices


def _normalize_feature_map(feature_map: Any) -> Any:
    _, torch_module, _ = _require_runtime()
    feature_map = feature_map.detach().cpu()
    minimum = feature_map.min()
    maximum = feature_map.max()
    if float(maximum - minimum) <= 1e-8:
        return torch_module.zeros_like(feature_map)
    return (feature_map - minimum) / (maximum - minimum)


def render_feature_map_figure(
    image_tensor: Any,
    display_image: Any,
    feature_maps_by_layer: dict[str, Any],
    top_indices_by_layer: dict[str, list[int]],
    class_name: str,
    predicted_class_name: str,
    confidence: float,
    layer_names: list[str],
    output_path: str | Path,
    model_name: str = "model",
) -> str:
    """Render the input image and top feature maps into one notebook-ready PNG."""
    _, _, pyplot = _require_runtime()
    output_path = Path(output_path)
    num_layers = len(layer_names)
    channels_per_layer = len(next(iter(feature_maps_by_layer.values())))
    ncols = max(channels_per_layer, 4)
    nrows = 1 + num_layers
    figure, axes = pyplot.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.5 * ncols, 2.4 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]
    elif nrows > 1 and ncols == 1:
        axes = [[axis] for axis in axes]

    input_axis = axes[0][0] if ncols > 1 else axes[0]
    input_axis.imshow(display_image)
    input_axis.set_title(
        f"Input\ntrue={class_name}\npred={predicted_class_name}\nconf={confidence:.3f}",
        fontsize=10,
    )
    input_axis.axis("off")

    denormalized = _denormalize_tensor(image_tensor)
    if ncols > 1:
        axes[0][1].imshow(denormalized)
        axes[0][1].set_title("Model Input Tensor", fontsize=10)
        axes[0][1].axis("off")
        for axis in axes[0][2:]:
            axis.axis("off")
    else:
        axes[0].imshow(denormalized)

    for layer_row_index, layer_name in enumerate(layer_names, start=1):
        row_axes = axes[layer_row_index]
        top_maps = feature_maps_by_layer[layer_name]
        top_indices = top_indices_by_layer[layer_name]
        for column_index, axis in enumerate(row_axes):
            if column_index >= len(top_maps):
                axis.axis("off")
                continue
            normalized_map = _normalize_feature_map(top_maps[column_index]).numpy()
            axis.imshow(normalized_map, cmap="viridis")
            axis.set_title(f"{layer_name}\nch {top_indices[column_index]}", fontsize=9)
            axis.axis("off")

    figure.suptitle(f"{model_name} Feature Maps", fontsize=14)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    pyplot.close(figure)
    return str(output_path.resolve())


def select_stage7_examples(
    predictions_csv_path: str | Path,
    target_fold: int,
    face_classes: tuple[str, ...] = FACE_CARD_CLASSES,
    number_classes: tuple[str, ...] = NUMBER_CARD_CLASSES,
    num_face_examples: int = 2,
    num_number_examples: int = 2,
) -> list[FeatureMapExample]:
    """Choose reproducible example images from the best fold's correct predictions."""
    pandas, _, _ = _require_runtime()
    predictions = pandas.read_csv(predictions_csv_path)
    fold_predictions = predictions.loc[(predictions["fold"] == target_fold) & (predictions["is_correct"])].copy()
    if fold_predictions.empty:
        raise ValueError(f"No correct predictions were found for fold {target_fold}.")

    examples: list[FeatureMapExample] = []
    selection_specs = [
        ("face_cards", face_classes, num_face_examples),
        ("number_cards", number_classes, num_number_examples),
    ]
    for group_name, allowed_classes, max_examples in selection_specs:
        candidates = (
            fold_predictions.loc[fold_predictions["true_class_name"].isin(allowed_classes)]
            .sort_values(by=["true_class_name", "confidence", "path"], ascending=[True, False, True], kind="stable")
            .groupby("true_class_name", as_index=False)
            .head(1)
            .sort_values(by=["confidence", "true_class_name"], ascending=[False, True], kind="stable")
            .head(max_examples)
        )
        if len(candidates) < max_examples:
            raise ValueError(
                f"Could not select {max_examples} distinct examples for group '{group_name}' from fold {target_fold}."
            )
        for row in candidates.itertuples(index=False):
            examples.append(
                FeatureMapExample(
                    group_name=group_name,
                    class_name=str(row.true_class_name),
                    path=str(row.path),
                    fold=int(row.fold),
                    confidence=float(row.confidence),
                    predicted_class_name=str(row.predicted_class_name),
                )
            )

    return examples


def _safe_stem(image_path: str | Path) -> str:
    path = Path(image_path)
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in path.stem)


def _write_stage7_markdown(
    run_name: str,
    stage5_summary: dict[str, Any],
    checkpoint_row: dict[str, Any],
    layer_names: list[str],
    examples: list[FeatureMapExample],
    artifacts: Stage7Artifacts,
) -> None:
    lines = ["## Stage 7 Feature-Map Summary", ""]
    lines.append(f"- Stage 5 run: `{run_name}`")
    lines.append(f"- Selected checkpoint fold: `{int(checkpoint_row['fold'])}`")
    lines.append(f"- Checkpoint path: `{checkpoint_row['checkpoint_path']}`")
    lines.append(f"- Best fold validation accuracy: `{float(checkpoint_row['best_metric_value']):.6f}`")
    lines.append(f"- Hooked layers: `{', '.join(layer_names)}`")
    lines.append("")
    lines.append("### Selected Examples")
    lines.append("")
    for example in examples:
        lines.append(
            f"- `{example.group_name}` | `{example.class_name}` | fold `{example.fold}` | "
            f"confidence `{example.confidence:.3f}` | `{example.path}`"
        )
    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- Stage 7 uses saved cross-validation checkpoints and Stage 6 prediction outputs; no retraining was performed.")
    lines.append(f"- {_dataset_balance_note(stage5_summary)}")
    lines.append("- The repo venv matplotlib import was repaired via a local overlay so pyplot-based PNG rendering can run again.")

    with Path(artifacts.summary_md).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_stage7_feature_maps(
    stage5_summary_path: str | Path | None = None,
    predictions_csv_path: str | Path | None = None,
    layer_names: list[str] | None = None,
    device: str = "cpu",
    top_k: int = 8,
) -> dict[str, Any]:
    """Build Stage 7 feature-map figures and summary artifacts."""
    pandas, _, pyplot = _require_runtime()
    del pyplot  # imported as part of runtime requirement only
    stage5_summary = _load_stage5_summary(stage5_summary_path=stage5_summary_path)
    run_name = str(stage5_summary["run_name"])
    predictions_csv = Path(predictions_csv_path) if predictions_csv_path else _discover_stage6_predictions_csv(run_name)
    checkpoint_row = _best_checkpoint_row(stage5_summary)
    selected_fold = int(checkpoint_row["fold"])
    selected_examples = select_stage7_examples(predictions_csv_path=predictions_csv, target_fold=selected_fold)
    layer_names = layer_names or ["backbone.layer3", "backbone.layer4"]

    output_root = FEATURE_MAPS_OUTPUT_DIR / run_name / f"best_fold_{selected_fold:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    selected_examples_csv = output_root / "selected_examples.csv"
    examples_frame = pandas.DataFrame([example.to_dict() for example in selected_examples])
    examples_frame.to_csv(selected_examples_csv, index=False)

    model, checkpoint, resolved_device = load_feature_map_model(
        checkpoint_path=checkpoint_row["checkpoint_path"],
        device=device,
    )
    image_size = int(checkpoint["train_config"]["image_size"])

    generated_figure_paths: list[str] = []
    example_summaries: list[dict[str, Any]] = []
    for example in selected_examples:
        image_tensor, display_image = load_image_tensor(example.path, image_size=image_size)
        activations, probabilities = extract_feature_maps(
            model=model,
            image_tensor=image_tensor,
            layer_names=layer_names,
            device=resolved_device,
        )
        feature_maps_by_layer: dict[str, Any] = {}
        top_indices_by_layer: dict[str, list[int]] = {}
        for layer_name in layer_names:
            top_maps, top_indices = select_top_feature_channels(activations[layer_name], top_k=top_k)
            feature_maps_by_layer[layer_name] = top_maps
            top_indices_by_layer[layer_name] = top_indices

        predicted_index = int(probabilities.argmax(dim=1).item())
        predicted_class_name = checkpoint["class_names"][predicted_index]
        predicted_confidence = float(probabilities[0, predicted_index].item())
        output_path = output_root / f"{example.group_name}__{example.class_name}__{_safe_stem(example.path)}.png"
        saved_path = render_feature_map_figure(
            image_tensor=image_tensor,
            display_image=display_image,
            feature_maps_by_layer=feature_maps_by_layer,
            top_indices_by_layer=top_indices_by_layer,
            class_name=example.class_name,
            predicted_class_name=predicted_class_name,
            confidence=predicted_confidence,
            layer_names=layer_names,
            output_path=output_path,
            model_name=str(checkpoint["model_spec"]["backbone"]["model_name"]),
        )
        generated_figure_paths.append(saved_path)
        example_summaries.append(
            {
                "group_name": example.group_name,
                "class_name": example.class_name,
                "path": example.path,
                "fold": example.fold,
                "selection_confidence": example.confidence,
                "render_predicted_class_name": predicted_class_name,
                "render_predicted_confidence": predicted_confidence,
                "output_png": saved_path,
                "top_channel_indices": top_indices_by_layer,
            }
        )

    # Build a lightweight overview figure by tiling the saved per-example PNGs.
    overview_path = output_root / "feature_map_overview.png"
    with Image.open(generated_figure_paths[0]) as first_image:
        tile_width, tile_height = first_image.size
    overview = Image.new("RGB", (tile_width * 2, tile_height * 2), "white")
    for index, image_path in enumerate(generated_figure_paths):
        with Image.open(image_path) as image:
            row = index // 2
            column = index % 2
            overview.paste(image, (column * tile_width, row * tile_height))
    overview.save(overview_path)

    artifacts = Stage7Artifacts(
        output_root=str(output_root.resolve()),
        selected_examples_csv=str(selected_examples_csv.resolve()),
        summary_json=str((output_root / "stage7_summary.json").resolve()),
        summary_md=str((output_root / "stage7_summary.md").resolve()),
        overview_png=str(overview_path.resolve()),
    )
    result = {
        "stage": "Stage 7",
        "run_name": run_name,
        "selected_checkpoint": checkpoint_row,
        "layer_names": layer_names,
        "device": str(resolved_device),
        "matplotlib_version": matplotlib.__version__ if matplotlib is not None else None,
        "artifacts": artifacts.to_dict(),
        "selected_examples": [example.to_dict() for example in selected_examples],
        "generated_feature_maps": example_summaries,
        "notes": [
            "Stage 7 uses the best saved cross-validation checkpoint rather than retraining.",
            "Example images were selected from the Stage 6 aggregated correct predictions for the chosen fold.",
            "The repo venv matplotlib import was repaired via a local overlay so pyplot-based rendering works again.",
            _dataset_balance_note(stage5_summary),
        ],
    }
    with Path(artifacts.summary_json).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    _write_stage7_markdown(
        run_name=run_name,
        stage5_summary=stage5_summary,
        checkpoint_row=checkpoint_row,
        layer_names=layer_names,
        examples=selected_examples,
        artifacts=artifacts,
    )
    return result


if __name__ == "__main__":
    result = build_stage7_feature_maps()
    print(
        json.dumps(
            {
                "artifacts": result["artifacts"],
                "selected_checkpoint": result["selected_checkpoint"],
                "selected_examples": result["selected_examples"],
            },
            indent=2,
        )
    )
