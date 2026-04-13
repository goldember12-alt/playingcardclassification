"""Bounded probe-screen utilities for refreshed full-dataset strategy selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - surfaced at runtime
    pd = None

from src.training.train_one_fold import train_one_fold
from src.utils.paths import DERIVED_RANK_DATASET_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR


@dataclass(frozen=True)
class ProbeScreenArtifacts:
    """Saved outputs for a probe-screen pass."""

    probe_table_csv: str
    probe_table_md: str
    probe_summary_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("Probe screening requires pandas.")
    return pd


def _freeze_strategy_row(probe: dict[str, Any]) -> str:
    if not probe.get("freeze_backbone", True):
        return "full_finetune"
    unfreeze_from = probe.get("unfreeze_from")
    if unfreeze_from:
        return f"partial_from_{unfreeze_from}"
    return "frozen_backbone"


def _classifier_head_row(probe: dict[str, Any]) -> str:
    hidden_dim = probe.get("classifier_hidden_dim")
    if hidden_dim is None:
        return "linear"
    return f"mlp_{int(hidden_dim)}"


def _failure_mode(history: Any) -> str:
    pandas = _require_pandas()
    if not isinstance(history, pandas.DataFrame):
        history = pandas.DataFrame(history)
    if history.empty:
        return "no_history"

    best_val_accuracy = float(history["val_accuracy"].max())
    best_train_accuracy = float(history["train_accuracy"].max())
    final_val_accuracy = float(history["val_accuracy"].iloc[-1])

    if best_val_accuracy < 0.60:
        return "underfitting_or_collapse"
    if best_train_accuracy - best_val_accuracy > 0.10:
        return "overfitting_gap"
    if best_val_accuracy - final_val_accuracy > 0.05:
        return "late_epoch_instability"
    return ""


def _probe_result_row(probe: dict[str, Any], history: Any, summary: dict[str, Any]) -> dict[str, Any]:
    pandas = _require_pandas()
    if not isinstance(history, pandas.DataFrame):
        history = pandas.DataFrame(history)

    best_accuracy_index = history["val_accuracy"].astype(float).idxmax()
    best_accuracy_row = history.loc[best_accuracy_index]
    return {
        "probe_name": str(probe["run_name"]),
        "rationale": str(probe.get("rationale", "")),
        "model_name": str(probe.get("model_name", "resnet50")),
        "freeze_strategy": _freeze_strategy_row(probe),
        "classifier_head": _classifier_head_row(probe),
        "optimizer": str(probe.get("optimizer_name", "adam")),
        "learning_rate": float(probe.get("learning_rate", 1e-3)),
        "backbone_learning_rate": (
            float(probe["backbone_learning_rate"]) if probe.get("backbone_learning_rate") is not None else None
        ),
        "batch_size": int(probe.get("batch_size", summary["batch_size"])),
        "scheduler": str(probe.get("scheduler_name", "none")),
        "augmentation": (
            str(probe.get("augmentation_profile"))
            if probe.get("use_augmentation")
            else "none"
        ),
        "class_weight_strategy": str(probe.get("class_weight_strategy", "none")),
        "sampling_strategy": str(probe.get("sampling_strategy", "none")),
        "epochs_completed": int(summary.get("epochs_completed", len(history))),
        "best_epoch_by_accuracy": int(best_accuracy_row["epoch"]),
        "best_val_accuracy": float(best_accuracy_row["val_accuracy"]),
        "val_loss_at_best_accuracy": float(best_accuracy_row["val_loss"]),
        "best_val_loss_any_epoch": float(history["val_loss"].astype(float).min()),
        "train_accuracy_at_best_accuracy": float(best_accuracy_row["train_accuracy"]),
        "elapsed_seconds_total": float(history["elapsed_seconds"].astype(float).sum()),
        "stopped_early": bool(summary.get("stopped_early", False)),
        "failure_mode": _failure_mode(history),
        "summary_json_path": str(summary["artifacts"]["summary_json_path"]),
        "metrics_csv_path": str(summary["artifacts"]["metrics_csv_path"]),
        "checkpoint_path": str(summary["artifacts"]["checkpoint_path"]),
    }


def _write_markdown_summary(
    rows: Any,
    selected_probe_name: str,
    output_path: str | Path,
    fold_index: int,
) -> None:
    pandas = _require_pandas()
    if not isinstance(rows, pandas.DataFrame):
        rows = pandas.DataFrame(rows)

    lines = ["## Probe Screen Summary", ""]
    lines.append(f"- Screening fold: `{fold_index}`")
    lines.append(f"- Selected strategy for full 5-fold run: `{selected_probe_name}`")
    lines.append("")
    lines.append("| Probe | Backbone | Freeze Strategy | Head | Optimizer | LR | Scheduler | Augmentation | Best Val Acc | Failure Mode |")
    lines.append("| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- |")
    for row in rows.itertuples(index=False):
        lr_display = f"{row.learning_rate:.0e}" if row.learning_rate < 0.001 else f"{row.learning_rate:.5f}"
        lines.append(
            f"| {row.probe_name} | {row.model_name} | {row.freeze_strategy} | {row.classifier_head} | "
            f"{row.optimizer} | {lr_display} | {row.scheduler} | {row.augmentation} | "
            f"{row.best_val_accuracy:.4f} | {row.failure_mode or 'none observed'} |"
        )

    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_probe_screen(
    probes: list[dict[str, Any]],
    output_stem: str = "full_dataset_probe_screen",
    fold_index: int = 0,
    num_folds: int = 5,
    random_seed: int = 42,
    raw_data_dir: str | Path = PROCESSED_DATA_DIR,
    dataset_name: str | None = DERIVED_RANK_DATASET_DIR.name,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a compact one-fold probe table and rank the candidates by validation accuracy."""
    pandas = _require_pandas()
    logs_dir = OUTPUTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    probe_summaries: list[dict[str, Any]] = []
    for probe in probes:
        train_kwargs = dict(probe)
        train_kwargs.pop("rationale", None)
        history, summary = train_one_fold(
            fold_index=fold_index,
            num_folds=num_folds,
            random_seed=random_seed,
            raw_data_dir=raw_data_dir,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            **train_kwargs,
        )
        rows.append(_probe_result_row(probe=probe, history=history, summary=summary))
        probe_summaries.append(summary)

    results = (
        pandas.DataFrame(rows)
        .sort_values(
            by=["best_val_accuracy", "val_loss_at_best_accuracy", "elapsed_seconds_total"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    selected_probe_name = str(results.iloc[0]["probe_name"])

    artifacts = ProbeScreenArtifacts(
        probe_table_csv=str((logs_dir / f"{output_stem}_probe_table.csv").resolve()),
        probe_table_md=str((logs_dir / f"{output_stem}_probe_table.md").resolve()),
        probe_summary_json=str((logs_dir / f"{output_stem}_probe_summary.json").resolve()),
    )
    results.to_csv(artifacts.probe_table_csv, index=False)
    _write_markdown_summary(
        rows=results,
        selected_probe_name=selected_probe_name,
        output_path=artifacts.probe_table_md,
        fold_index=fold_index,
    )

    payload = {
        "fold_index": fold_index,
        "num_folds": num_folds,
        "random_seed": random_seed,
        "selected_probe_name": selected_probe_name,
        "artifacts": artifacts.to_dict(),
        "probe_results": results.to_dict(orient="records"),
        "probe_run_summaries": probe_summaries,
    }
    with Path(artifacts.probe_summary_json).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
