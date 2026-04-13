"""Deterministic stratified fold generation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    StratifiedKFold = None

from src.data.dataset import (
    DEFAULT_PIPELINE_DATASET_NAME,
    DEFAULT_PIPELINE_DATA_DIR,
    DatasetSummary,
    build_sample_inventory,
)
from src.utils.paths import FOLDS_OUTPUT_DIR


@dataclass
class FoldSummary:
    """Notebook-friendly summary of generated cross-validation folds."""

    dataset_found: bool
    discovered_root: str | None
    n_splits: int
    random_seed: int
    fold_source_strategy: str | None = None
    fold_source_splits: list[str] = field(default_factory=list)
    held_out_splits: list[str] = field(default_factory=list)
    total_pool_samples: int = 0
    total_held_out_samples: int = 0
    class_names: list[str] = field(default_factory=list)
    class_to_index: dict[str, int] = field(default_factory=dict)
    pool_class_counts: dict[str, int] = field(default_factory=dict)
    fold_overview: list[dict[str, Any]] = field(default_factory=list)
    validation_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary into a plain dictionary."""
        return asdict(self)


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for fold-generation utilities.")
    return pd


def _require_sklearn() -> Any:
    if StratifiedKFold is None:
        raise ImportError("scikit-learn is required for stratified fold generation.")
    return StratifiedKFold


def _validate_dataset_runtime_ready(dataset_summary: DatasetSummary) -> None:
    """Block fold generation when the discovered raw data is known to be inconsistent."""
    if dataset_summary.runtime_ready_for_stage5:
        return

    recommendation = dataset_summary.recommended_data_action or "inspect_local_data"
    recent_notes = " ".join(dataset_summary.notes[-3:]) if dataset_summary.notes else "No additional notes available."
    raise ValueError(
        "The discovered dataset is not ready for Stage 5 fold generation. "
        f"Recommended action: {recommendation}. {recent_notes}"
    )


def _normalize_inventory(inventory: Any) -> Any:
    pandas = _require_pandas()
    required_columns = {"item_id", "path", "class_name", "class_index", "split_name"}
    if not isinstance(inventory, pandas.DataFrame):
        inventory = pandas.DataFrame(inventory)

    missing_columns = sorted(required_columns - set(inventory.columns))
    if missing_columns:
        raise ValueError(f"Inventory is missing required columns: {missing_columns}")

    inventory = inventory.copy()
    inventory["item_id"] = inventory["item_id"].astype(str)
    inventory["path"] = inventory["path"].astype(str)
    inventory["class_name"] = inventory["class_name"].astype(str)
    inventory["class_index"] = inventory["class_index"].astype(int)
    inventory["split_name"] = inventory["split_name"].astype(str)
    if "relative_path" not in inventory.columns:
        inventory["relative_path"] = inventory["path"]

    return (
        inventory.sort_values(by=["split_name", "class_index", "relative_path", "path"], kind="stable")
        .reset_index(drop=True)
    )


def _choose_fold_source(inventory: Any) -> tuple[Any, Any, str, list[str]]:
    """Select which discovered items participate in cross-validation."""
    pandas = _require_pandas()
    inventory = _normalize_inventory(inventory)
    split_names = sorted(inventory["split_name"].dropna().unique().tolist())
    named_splits = [split_name for split_name in split_names if split_name != "all"]

    notes: list[str] = []
    if named_splits:
        pool_mask = inventory["split_name"] != "test"
        held_out_mask = inventory["split_name"] == "test"
        pool_inventory = inventory.loc[pool_mask].copy()
        held_out_inventory = inventory.loc[held_out_mask].copy()
        source_strategy = "named_non_test_pool"
        notes.append(
            "Detected named dataset splits. Cross-validation folds are generated from the non-test pool, "
            "and any `test` split is kept untouched as an external holdout."
        )
        if pool_inventory.empty:
            raise ValueError("No non-test samples are available for fold generation.")
    else:
        pool_inventory = inventory.copy()
        held_out_inventory = pandas.DataFrame(columns=inventory.columns)
        source_strategy = "all_split_pool"
        notes.append(
            "Detected a flat class-directory dataset. Cross-validation folds are generated from the combined `all` pool."
        )

    return pool_inventory.reset_index(drop=True), held_out_inventory.reset_index(drop=True), source_strategy, notes


def validate_stratification_support(inventory: Any, n_splits: int = 5) -> dict[str, int]:
    """Validate that every class has enough samples for strict stratified k-fold."""
    inventory = _normalize_inventory(inventory)
    if inventory.empty:
        raise ValueError("Fold generation requires at least one sample in the cross-validation pool.")

    class_counts = inventory["class_name"].value_counts().sort_index()
    insufficient = class_counts[class_counts < n_splits]
    if not insufficient.empty:
        details = ", ".join(f"{class_name}={count}" for class_name, count in insufficient.items())
        raise ValueError(
            f"Strict stratified {n_splits}-fold CV is impossible because some classes have fewer than {n_splits} "
            f"samples in the fold pool: {details}"
        )
    return {class_name: int(count) for class_name, count in class_counts.items()}


def make_folds(
    inventory: Any,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Any:
    """Assign each pool sample to one validation fold using deterministic stratified k-fold."""
    splitter_cls = _require_sklearn()
    inventory = _normalize_inventory(inventory)
    class_counts = validate_stratification_support(inventory, n_splits=n_splits)

    assignments = inventory.copy()
    assignments["fold"] = -1
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(assignments["path"], assignments["class_index"])
    ):
        assignments.loc[validation_indices, "fold"] = fold_index

    if (assignments["fold"] < 0).any():
        raise RuntimeError("Fold assignment failed: at least one sample was not assigned to a validation fold.")

    assignments["fold"] = assignments["fold"].astype(int)
    assignments["fold_role"] = "cv_pool"
    assignments["n_splits"] = n_splits
    assignments["random_seed"] = random_seed
    assignments["pool_class_count"] = assignments["class_name"].map(class_counts).astype(int)
    return assignments


def build_fold_inventory(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[Any, FoldSummary]:
    """Build reusable fold assignments from the discovered Stage 1 dataset inventory."""
    pandas = _require_pandas()
    inventory, dataset_summary = build_sample_inventory(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )
    _validate_dataset_runtime_ready(dataset_summary)

    pool_inventory, held_out_inventory, source_strategy, notes = _choose_fold_source(inventory)
    fold_assignments = make_folds(pool_inventory, n_splits=n_splits, random_seed=random_seed)

    if not held_out_inventory.empty:
        held_out_inventory = held_out_inventory.copy()
        held_out_inventory["fold"] = pandas.Series([pandas.NA] * len(held_out_inventory), dtype="Int64")
        held_out_inventory["fold_role"] = "held_out_test"
        held_out_inventory["n_splits"] = n_splits
        held_out_inventory["random_seed"] = random_seed
        held_out_inventory["pool_class_count"] = pandas.NA
        assignments = pandas.concat([fold_assignments, held_out_inventory], ignore_index=True, sort=False)
    else:
        assignments = fold_assignments.copy()

    assignments["is_cv_pool"] = assignments["fold_role"].eq("cv_pool")
    assignments["is_held_out"] = assignments["fold_role"].eq("held_out_test")

    summary = summarize_folds(
        assignments=assignments,
        dataset_summary=dataset_summary,
        n_splits=n_splits,
        random_seed=random_seed,
        fold_source_strategy=source_strategy,
        notes=notes,
    )
    return assignments, summary


def summarize_folds(
    assignments: Any,
    dataset_summary: DatasetSummary,
    n_splits: int,
    random_seed: int,
    fold_source_strategy: str,
    notes: list[str] | None = None,
) -> FoldSummary:
    """Create a concise summary of fold coverage and counts."""
    assignments = _normalize_inventory(assignments)
    cv_pool = assignments.loc[assignments["fold_role"] == "cv_pool"].copy()
    held_out = assignments.loc[assignments["fold_role"] == "held_out_test"].copy()

    pool_class_counts = (
        cv_pool["class_name"].value_counts().sort_index().astype(int).to_dict()
        if not cv_pool.empty
        else {}
    )

    fold_overview: list[dict[str, Any]] = []
    validation_class_counts: dict[str, dict[str, int]] = {}
    for fold_index in sorted(cv_pool["fold"].dropna().astype(int).unique().tolist()):
        validation_slice = cv_pool.loc[cv_pool["fold"] == fold_index]
        training_slice = cv_pool.loc[cv_pool["fold"] != fold_index]
        class_counts = (
            validation_slice["class_name"].value_counts().reindex(dataset_summary.class_names, fill_value=0).astype(int)
        )
        validation_class_counts[str(fold_index)] = {
            class_name: int(count) for class_name, count in class_counts.items()
        }
        fold_overview.append(
            {
                "fold": int(fold_index),
                "train_samples": int(len(training_slice)),
                "validation_samples": int(len(validation_slice)),
                "validation_classes_present": int((class_counts > 0).sum()),
                "validation_min_class_count": int(class_counts.min()) if not class_counts.empty else 0,
                "validation_max_class_count": int(class_counts.max()) if not class_counts.empty else 0,
            }
        )

    summary_notes = list(notes or [])
    summary_notes.append("Fold assignments are deterministic for a given seed and sample inventory.")

    return FoldSummary(
        dataset_found=dataset_summary.dataset_found,
        discovered_root=dataset_summary.discovered_root,
        n_splits=n_splits,
        random_seed=random_seed,
        fold_source_strategy=fold_source_strategy,
        fold_source_splits=sorted(cv_pool["split_name"].dropna().unique().tolist()),
        held_out_splits=sorted(held_out["split_name"].dropna().unique().tolist()),
        total_pool_samples=int(len(cv_pool)),
        total_held_out_samples=int(len(held_out)),
        class_names=list(dataset_summary.class_names),
        class_to_index=dict(dataset_summary.class_to_index),
        pool_class_counts={class_name: int(count) for class_name, count in pool_class_counts.items()},
        fold_overview=fold_overview,
        validation_class_counts=validation_class_counts,
        notes=summary_notes,
    )


def fold_overview_to_dataframe(summary: FoldSummary) -> Any:
    """Convert the fold overview section of the summary to a DataFrame."""
    pandas = _require_pandas()
    return pandas.DataFrame(summary.fold_overview)


def validation_counts_to_dataframe(summary: FoldSummary) -> Any:
    """Convert validation class counts by fold into a long-form DataFrame."""
    pandas = _require_pandas()
    rows: list[dict[str, Any]] = []
    for fold_name, class_counts in summary.validation_class_counts.items():
        for class_name, count in class_counts.items():
            rows.append(
                {
                    "fold": int(fold_name),
                    "class_name": class_name,
                    "validation_count": int(count),
                }
            )
    return pandas.DataFrame(rows)


def save_fold_artifacts(
    assignments: Any,
    summary: FoldSummary,
    output_dir: str | Path = FOLDS_OUTPUT_DIR,
    stem: str = "stratified_5fold",
) -> dict[str, str]:
    """Save reusable fold assignments and summaries as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / f"{stem}_assignments.csv"
    overview_path = output_dir / f"{stem}_overview.csv"
    validation_counts_path = output_dir / f"{stem}_validation_class_counts.csv"

    _normalize_inventory(assignments).to_csv(assignments_path, index=False)
    fold_overview_to_dataframe(summary).to_csv(overview_path, index=False)
    validation_counts_to_dataframe(summary).to_csv(validation_counts_path, index=False)

    return {
        "assignments_csv": str(assignments_path.resolve()),
        "overview_csv": str(overview_path.resolve()),
        "validation_counts_csv": str(validation_counts_path.resolve()),
    }


def build_and_save_folds(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    n_splits: int = 5,
    random_seed: int = 42,
    output_dir: str | Path = FOLDS_OUTPUT_DIR,
    stem: str = "stratified_5fold",
) -> tuple[Any, FoldSummary, dict[str, str]]:
    """Discover the dataset, generate folds, and persist the reusable CSV artifacts."""
    assignments, summary = build_fold_inventory(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
        n_splits=n_splits,
        random_seed=random_seed,
    )
    saved_paths = save_fold_artifacts(assignments=assignments, summary=summary, output_dir=output_dir, stem=stem)
    return assignments, summary, saved_paths


def fold_summary_to_markdown(summary: FoldSummary) -> str:
    """Render a concise markdown summary suitable for notebook inclusion."""
    lines = ["## Fold Summary", ""]
    lines.append(f"- Dataset found: {'yes' if summary.dataset_found else 'no'}")
    lines.append(f"- Discovered root: `{summary.discovered_root}`" if summary.discovered_root else "- Discovered root: none")
    lines.append(f"- Fold strategy: `{summary.fold_source_strategy}`")
    lines.append(f"- Number of folds: {summary.n_splits}")
    lines.append(f"- Random seed: {summary.random_seed}")
    lines.append(
        f"- Fold source splits: {', '.join(summary.fold_source_splits) if summary.fold_source_splits else 'none'}"
    )
    lines.append(f"- Held-out splits: {', '.join(summary.held_out_splits) if summary.held_out_splits else 'none'}")
    lines.append(f"- CV pool samples: {summary.total_pool_samples}")
    lines.append(f"- Held-out samples: {summary.total_held_out_samples}")
    lines.append("")
    lines.append("### Per-Fold Overview")
    lines.append("")
    lines.append("| Fold | Train Samples | Validation Samples | Validation Classes Present | Min Class Count | Max Class Count |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary.fold_overview:
        lines.append(
            f"| {row['fold']} | {row['train_samples']} | {row['validation_samples']} | "
            f"{row['validation_classes_present']} | {row['validation_min_class_count']} | "
            f"{row['validation_max_class_count']} |"
        )

    if summary.notes:
        lines.append("")
        lines.append("### Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
