"""Dataset discovery, summary, and loading utilities for playing-card images."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image
try:
    import pandas as pd
except ImportError:  # pragma: no cover - import error is surfaced when functions are called
    pd = None

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - allows discovery utilities without torch installed
    class Dataset:  # type: ignore[override]
        """Fallback Dataset base class when torch is unavailable."""


from src.utils.paths import DERIVED_RANK_DATASET_NAME, PROCESSED_DATA_DIR, RAW_DATA_DIR


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METADATA_CSV_NAME = "cards.csv"
KNOWN_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "valid": ("valid", "val", "validation"),
    "test": ("test", "testing", "eval", "evaluation"),
}
SPLIT_DISPLAY_ORDER = {"train": 0, "valid": 1, "test": 2, "all": 3}
EXPECTED_CARD_CLASSES = (
    "ace",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "jack",
    "queen",
    "king",
    "joker",
)
RANK_NAME_ALIASES = {
    "joker": "joker",
    "xxx": "joker",
}
DEFAULT_PIPELINE_DATA_DIR = PROCESSED_DATA_DIR
DEFAULT_PIPELINE_DATASET_NAME = DERIVED_RANK_DATASET_NAME


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for tabular dataset inventory utilities.")
    return pd


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for a single discovered image file."""

    path: Path
    class_name: str
    class_index: int
    split_name: str


@dataclass(frozen=True)
class SampleImageInfo:
    """Basic information about one representative image."""

    path: str
    size: tuple[int, int]
    mode: str


@dataclass
class DatasetSummary:
    """Notebook-friendly summary of the discovered dataset layout."""

    dataset_found: bool
    expected_raw_data_dir: str
    discovered_root: str | None = None
    layout: str | None = None
    split_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    class_to_index: dict[str, int] = field(default_factory=dict)
    counts_by_split: dict[str, dict[str, int]] = field(default_factory=dict)
    total_images: int = 0
    sample_image: SampleImageInfo | None = None
    expected_class_count: int = 14
    class_count_matches_expectation: bool = False
    missing_expected_classes: list[str] = field(default_factory=list)
    unexpected_classes: list[str] = field(default_factory=list)
    metadata_csv_found: bool = False
    metadata_csv_path: str | None = None
    metadata_total_rows: int = 0
    metadata_split_counts: dict[str, int] = field(default_factory=dict)
    metadata_label_count: int = 0
    metadata_rank_names: list[str] = field(default_factory=list)
    metadata_existing_rows: int = 0
    metadata_missing_rows: int = 0
    metadata_existing_rows_by_split: dict[str, int] = field(default_factory=dict)
    metadata_missing_rows_by_split: dict[str, int] = field(default_factory=dict)
    metadata_missing_examples: dict[str, list[str]] = field(default_factory=dict)
    assignment_target_classes: list[str] = field(default_factory=list)
    assignment_target_schema_supported: bool = False
    runtime_ready_for_stage5: bool = False
    recommended_data_action: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary into a plain dictionary."""
        return asdict(self)


class CardImageDataset(Dataset):
    """Simple image dataset backed by discovered image records."""

    def __init__(
        self,
        records: list[ImageRecord],
        class_names: list[str],
        transform: Callable[[Image.Image], Any] | None = None,
        transform_overrides_by_class: dict[str, Callable[[Image.Image], Any]] | None = None,
        image_mode: str = "RGB",
    ) -> None:
        self.records = records
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = transform
        self.transform_overrides_by_class = transform_overrides_by_class or {}
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert(self.image_mode)
            transform = self.transform_overrides_by_class.get(record.class_name, self.transform)
            if transform is not None:
                image = transform(image)
        return image, record.class_index


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _normalize_label(label: str) -> str:
    return "".join(character for character in label.lower() if character.isalnum())


def _normalize_rank_name(rank_name: str) -> str:
    normalized = _normalize_label(rank_name)
    return RANK_NAME_ALIASES.get(normalized, normalized)


def _sorted_subdirectories(path: Path) -> list[Path]:
    return sorted((item for item in path.iterdir() if item.is_dir()), key=lambda item: item.name.casefold())


def _directory_has_images(path: Path) -> bool:
    return any(_is_image_file(child) for child in path.iterdir())


def _directory_has_class_subdirectories(path: Path) -> bool:
    return any(child.is_dir() and _directory_has_images(child) for child in path.iterdir())


def _split_sort_key(split_name: str) -> tuple[int, str]:
    return (SPLIT_DISPLAY_ORDER.get(split_name, len(SPLIT_DISPLAY_ORDER)), split_name)


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: _split_sort_key(item[0])))


def _find_split_directories(path: Path) -> dict[str, Path]:
    split_directories: dict[str, Path] = {}
    for child in _sorted_subdirectories(path):
        child_name = child.name.casefold()
        for canonical_name, aliases in KNOWN_SPLIT_ALIASES.items():
            if child_name in aliases and _directory_has_class_subdirectories(child):
                split_directories[canonical_name] = child
                break
    return split_directories


def _looks_like_dataset_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if _find_split_directories(path):
        return True
    return _directory_has_class_subdirectories(path)


def find_candidate_dataset_roots(
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    max_depth: int = 2,
) -> list[Path]:
    """Return dataset-like directories under the raw-data folder."""
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        return []

    candidates: list[Path] = []
    queue: list[tuple[Path, int]] = [(raw_data_dir, 0)]
    seen: set[Path] = set()
    while queue:
        current_path, depth = queue.pop(0)
        resolved = current_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        if _looks_like_dataset_root(current_path):
            candidates.append(current_path)
            continue

        if depth >= max_depth:
            continue

        for child in _sorted_subdirectories(current_path):
            queue.append((child, depth + 1))

    return candidates


def discover_dataset_root(
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
) -> Path | None:
    """Discover the most likely dataset root under ``data/raw``."""
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        return None

    if dataset_name:
        preferred = raw_data_dir / dataset_name
        return preferred if _looks_like_dataset_root(preferred) else None

    candidates = find_candidate_dataset_roots(raw_data_dir=raw_data_dir)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return max(
        candidates,
        key=lambda candidate: sum(1 for path in candidate.rglob("*") if _is_image_file(path)),
    )


def _collect_flat_records(root: Path) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for class_dir in _sorted_subdirectories(root):
        image_paths = [path for path in sorted(class_dir.iterdir()) if _is_image_file(path)]
        if not image_paths:
            continue
        for image_path in image_paths:
            records.append(("all", class_dir.name, image_path))
    return records


def _collect_split_records(root: Path, split_directories: dict[str, Path]) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for split_name, split_dir in split_directories.items():
        for class_dir in _sorted_subdirectories(split_dir):
            image_paths = [path for path in sorted(class_dir.iterdir()) if _is_image_file(path)]
            if not image_paths:
                continue
            for image_path in image_paths:
                records.append((split_name, class_dir.name, image_path))
    return records


def _inspect_metadata_csv(raw_data_dir: Path) -> dict[str, Any]:
    metadata_path = raw_data_dir / METADATA_CSV_NAME
    summary: dict[str, Any] = {
        "metadata_csv_found": metadata_path.exists(),
        "metadata_csv_path": str(metadata_path.resolve()) if metadata_path.exists() else None,
        "metadata_total_rows": 0,
        "metadata_split_counts": {},
        "metadata_label_count": 0,
        "metadata_rank_names": [],
        "metadata_existing_rows": 0,
        "metadata_missing_rows": 0,
        "metadata_existing_rows_by_split": {},
        "metadata_missing_rows_by_split": {},
        "metadata_missing_examples": {},
    }
    if not metadata_path.exists():
        return summary

    split_counts: defaultdict[str, int] = defaultdict(int)
    labels: set[str] = set()
    rank_names: set[str] = set()
    existing_rows_by_split: defaultdict[str, int] = defaultdict(int)
    missing_rows_by_split: defaultdict[str, int] = defaultdict(int)
    missing_examples: defaultdict[str, list[str]] = defaultdict(list)
    total_rows = 0
    existing_rows = 0
    missing_rows = 0

    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            split_name = row.get("data set", "").strip().lower()
            label_name = row.get("labels", "").strip()
            rank_name = _normalize_rank_name(row.get("card type", "").strip())
            relative_path = row.get("filepaths", "").strip()
            absolute_path = raw_data_dir / relative_path

            split_counts[split_name] += 1
            if label_name:
                labels.add(label_name)
            if rank_name:
                rank_names.add(rank_name)

            if absolute_path.exists():
                existing_rows += 1
                existing_rows_by_split[split_name] += 1
            else:
                missing_rows += 1
                missing_rows_by_split[split_name] += 1
                if relative_path and len(missing_examples[split_name]) < 5:
                    missing_examples[split_name].append(relative_path)

    summary.update(
        metadata_total_rows=total_rows,
        metadata_split_counts=_sorted_counts(dict(split_counts)),
        metadata_label_count=len(labels),
        metadata_rank_names=sorted(rank_names, key=str.casefold),
        metadata_existing_rows=existing_rows,
        metadata_missing_rows=missing_rows,
        metadata_existing_rows_by_split=_sorted_counts(dict(existing_rows_by_split)),
        metadata_missing_rows_by_split=_sorted_counts(dict(missing_rows_by_split)),
        metadata_missing_examples=dict(sorted(missing_examples.items(), key=lambda item: _split_sort_key(item[0]))),
    )
    return summary


def collect_image_records(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
) -> tuple[list[ImageRecord], DatasetSummary]:
    """Collect image records and the corresponding dataset summary."""
    summary = summarize_dataset(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )
    if not summary.dataset_found or not summary.discovered_root:
        expected_location = Path(raw_data_dir).resolve()
        raise FileNotFoundError(
            "No dataset was found under "
            f"{expected_location}. Place the extracted cards dataset there, for example "
            f"{expected_location / 'cards-image-datasetclassification'}."
        )

    discovered_root = Path(summary.discovered_root)
    split_directories = _find_split_directories(discovered_root)
    raw_records = (
        _collect_split_records(discovered_root, split_directories)
        if split_directories
        else _collect_flat_records(discovered_root)
    )

    records = [
        ImageRecord(
            path=image_path,
            class_name=class_name,
            class_index=summary.class_to_index[class_name],
            split_name=split_name,
        )
        for split_name, class_name, image_path in raw_records
    ]
    return records, summary


def records_to_dataframe(records: list[ImageRecord], dataset_root: str | Path | None = None) -> Any:
    """Convert discovered image records into a stable item-level metadata table."""
    pandas = _require_pandas()
    resolved_root = Path(dataset_root).resolve() if dataset_root else None
    rows: list[dict[str, Any]] = []
    for record in records:
        resolved_path = record.path.resolve()
        try:
            relative_path = str(resolved_path.relative_to(resolved_root)) if resolved_root else resolved_path.name
        except ValueError:
            relative_path = resolved_path.name

        rows.append(
            {
                "item_id": str(resolved_path),
                "path": str(resolved_path),
                "relative_path": relative_path,
                "class_name": record.class_name,
                "class_index": record.class_index,
                "split_name": record.split_name,
            }
        )

    dataframe = pandas.DataFrame(rows)
    if dataframe.empty:
        return dataframe

    return (
        dataframe.sort_values(by=["split_name", "class_index", "relative_path", "path"], kind="stable")
        .reset_index(drop=True)
    )


def build_sample_inventory(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
) -> tuple[Any, DatasetSummary]:
    """Build an item-level metadata table from the discovered dataset records."""
    records, summary = collect_image_records(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )
    return records_to_dataframe(records, dataset_root=summary.discovered_root), summary


def summarize_dataset(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    expected_classes: tuple[str, ...] = EXPECTED_CARD_CLASSES,
) -> DatasetSummary:
    """Inspect the dataset layout and return a structured summary."""
    raw_data_dir = Path(raw_data_dir)
    resolved_root = Path(dataset_root) if dataset_root else discover_dataset_root(raw_data_dir, dataset_name)
    summary = DatasetSummary(
        dataset_found=False,
        expected_raw_data_dir=str(raw_data_dir.resolve()),
        expected_class_count=len(expected_classes),
        assignment_target_classes=list(expected_classes),
    )

    if resolved_root is None:
        summary.notes.append(
            "Dataset not found locally. Populate data/raw/ with the extracted Kaggle dataset or a compatible image tree."
        )
        return summary
    if not resolved_root.exists() or not resolved_root.is_dir():
        summary.notes.append(f"The requested dataset root does not exist: {resolved_root}")
        return summary

    split_directories = _find_split_directories(resolved_root)
    raw_records = (
        _collect_split_records(resolved_root, split_directories)
        if split_directories
        else _collect_flat_records(resolved_root)
    )
    if not raw_records:
        summary.notes.append(
            f"No image files were discovered under {resolved_root}. Populate class folders with supported image files."
        )
        return summary

    class_names = sorted({class_name for _, class_name, _ in raw_records}, key=str.casefold)
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_image: SampleImageInfo | None = None

    for split_name, class_name, image_path in raw_records:
        counts_by_split[split_name][class_name] += 1
        if sample_image is None:
            with Image.open(image_path) as image:
                sample_image = SampleImageInfo(
                    path=str(image_path.resolve()),
                    size=image.size,
                    mode=image.mode,
                )

    normalized_expected = {_normalize_label(name): name for name in expected_classes}
    normalized_discovered = {_normalize_label(name): name for name in class_names}
    missing_expected = sorted(
        [name for key, name in normalized_expected.items() if key not in normalized_discovered],
        key=str.casefold,
    )
    unexpected_classes = sorted(
        [name for key, name in normalized_discovered.items() if key not in normalized_expected],
        key=str.casefold,
    )

    summary.dataset_found = True
    summary.discovered_root = str(resolved_root.resolve())
    summary.layout = "split_directories" if split_directories else "flat_class_directories"
    summary.split_names = list(sorted(counts_by_split.keys(), key=_split_sort_key))
    summary.class_names = class_names
    summary.class_to_index = class_to_index
    summary.counts_by_split = {
        split_name: dict(sorted(class_counts.items(), key=lambda item: item[0].casefold()))
        for split_name, class_counts in sorted(counts_by_split.items(), key=lambda item: _split_sort_key(item[0]))
    }
    summary.total_images = sum(sum(class_counts.values()) for class_counts in summary.counts_by_split.values())
    summary.sample_image = sample_image
    summary.class_count_matches_expectation = not missing_expected and not unexpected_classes
    summary.missing_expected_classes = missing_expected
    summary.unexpected_classes = unexpected_classes
    metadata_summary = _inspect_metadata_csv(raw_data_dir)
    summary.metadata_csv_found = bool(metadata_summary["metadata_csv_found"])
    summary.metadata_csv_path = metadata_summary["metadata_csv_path"]
    summary.metadata_total_rows = int(metadata_summary["metadata_total_rows"])
    summary.metadata_split_counts = dict(metadata_summary["metadata_split_counts"])
    summary.metadata_label_count = int(metadata_summary["metadata_label_count"])
    summary.metadata_rank_names = list(metadata_summary["metadata_rank_names"])
    summary.metadata_existing_rows = int(metadata_summary["metadata_existing_rows"])
    summary.metadata_missing_rows = int(metadata_summary["metadata_missing_rows"])
    summary.metadata_existing_rows_by_split = dict(metadata_summary["metadata_existing_rows_by_split"])
    summary.metadata_missing_rows_by_split = dict(metadata_summary["metadata_missing_rows_by_split"])
    summary.metadata_missing_examples = dict(metadata_summary["metadata_missing_examples"])

    normalized_expected_rank_names = {_normalize_rank_name(name) for name in expected_classes}
    normalized_metadata_rank_names = {_normalize_rank_name(name) for name in summary.metadata_rank_names}
    summary.assignment_target_schema_supported = bool(summary.metadata_rank_names) and (
        normalized_expected_rank_names == normalized_metadata_rank_names
    )

    if split_directories:
        summary.notes.append(
            "Detected split-based layout. Supported split aliases are train/training, "
            "valid/val/validation, and test/testing/eval/evaluation."
        )
    else:
        summary.notes.append(
            "Detected a flat class-directory layout. Later stages can derive folds from the combined `all` split."
        )

    if not summary.class_count_matches_expectation:
        summary.notes.append(
            f"Discovered {len(class_names)} classes, which does not match the assignment expectation of "
            f"{len(expected_classes)} classes."
        )

    if summary.metadata_csv_found:
        summary.notes.append(
            f"`{METADATA_CSV_NAME}` describes {summary.metadata_label_count} suit-specific labels grouped into "
            f"{len(summary.metadata_rank_names)} rank targets across splits: "
            f"{', '.join(f'{name}={count}' for name, count in summary.metadata_split_counts.items())}."
        )
        if "valid" in summary.metadata_split_counts and "valid" not in summary.split_names:
            summary.notes.append(
                "The metadata expects a `valid/` split, but no local `valid/` or `val/` directory was found under "
                "the discovered dataset root."
            )
        if summary.metadata_missing_rows > 0:
            summary.notes.append(
                f"Only {summary.metadata_existing_rows} of {summary.metadata_total_rows} metadata-referenced files "
                f"exist locally; {summary.metadata_missing_rows} referenced files are missing."
            )
            for split_name, missing_count in summary.metadata_missing_rows_by_split.items():
                example_paths = summary.metadata_missing_examples.get(split_name, [])
                if example_paths:
                    summary.notes.append(
                        f"Missing metadata rows in `{split_name}`: {missing_count}. Example paths: "
                        f"{', '.join(example_paths)}"
                    )
                else:
                    summary.notes.append(f"Missing metadata rows in `{split_name}`: {missing_count}.")
    else:
        summary.notes.append(
            f"No `{METADATA_CSV_NAME}` file was found, so rank-level reconciliation can only rely on folder names."
        )

    if summary.assignment_target_schema_supported:
        summary.notes.append(
            "The metadata rank schema matches the assignment's 14 target classes after normalizing the CSV `xxx` "
            "joker label to `joker`."
        )
    elif summary.metadata_csv_found:
        summary.notes.append(
            "The metadata rank schema does not cleanly match the assignment target classes and needs manual review."
        )

    if (
        summary.metadata_csv_found
        and summary.assignment_target_schema_supported
        and summary.metadata_missing_rows == 0
        and "valid" in summary.metadata_split_counts
        and "valid" in summary.split_names
        and len(summary.class_names) == len(expected_classes)
    ):
        summary.runtime_ready_for_stage5 = True
        summary.recommended_data_action = "use_existing_14_class_local_dataset"
    elif summary.metadata_csv_found and summary.assignment_target_schema_supported:
        summary.runtime_ready_for_stage5 = False
        summary.recommended_data_action = "stop_and_correct_local_data"
        summary.notes.append(
            "Stage 5 should stay blocked for now: the local data is not runtime-consistent enough to derive a "
            "trustworthy 14-class training pool."
        )
    elif summary.class_count_matches_expectation:
        summary.runtime_ready_for_stage5 = True
        summary.recommended_data_action = "use_folder_dataset_as_is"
    else:
        summary.runtime_ready_for_stage5 = False
        summary.recommended_data_action = "stop_and_correct_local_data"

    return summary


def build_dataset(
    split_name: str,
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    transform: Callable[[Image.Image], Any] | None = None,
    image_mode: str = "RGB",
) -> CardImageDataset:
    """Build a dataset for a discovered split."""
    records, summary = collect_image_records(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )

    available_splits = {record.split_name for record in records}
    if split_name not in available_splits:
        raise ValueError(f"Requested split '{split_name}' is not available. Found: {sorted(available_splits)}")

    split_records = [record for record in records if record.split_name == split_name]
    return CardImageDataset(
        records=split_records,
        class_names=summary.class_names,
        transform=transform,
        image_mode=image_mode,
    )


def build_datasets(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    transform_map: dict[str, Callable[[Image.Image], Any] | None] | None = None,
    image_mode: str = "RGB",
) -> tuple[dict[str, CardImageDataset], DatasetSummary]:
    """Build datasets for every discovered split."""
    records, summary = collect_image_records(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )
    transform_map = transform_map or {}

    datasets: dict[str, CardImageDataset] = {}
    for split_name in sorted({record.split_name for record in records}):
        split_records = [record for record in records if record.split_name == split_name]
        datasets[split_name] = CardImageDataset(
            records=split_records,
            class_names=summary.class_names,
            transform=transform_map.get(split_name),
            image_mode=image_mode,
        )
    return datasets, summary


def run_dataset_sanity_check(
    dataset_root: str | Path | None = None,
    raw_data_dir: str | Path = DEFAULT_PIPELINE_DATA_DIR,
    dataset_name: str | None = DEFAULT_PIPELINE_DATASET_NAME,
    transform_map: dict[str, Callable[[Image.Image], Any] | None] | None = None,
) -> dict[str, Any]:
    """Load one sample per split and report whether discovery and transforms work."""
    summary = summarize_dataset(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
    )
    results: dict[str, Any] = {
        "dataset_found": summary.dataset_found,
        "discovered_root": summary.discovered_root,
        "split_names": summary.split_names,
        "class_count": len(summary.class_names),
        "sample_checks": {},
        "notes": list(summary.notes),
    }
    if not summary.dataset_found:
        return results

    datasets, _ = build_datasets(
        dataset_root=dataset_root,
        raw_data_dir=raw_data_dir,
        dataset_name=dataset_name,
        transform_map=transform_map,
    )

    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            results["sample_checks"][split_name] = {"status": "empty_split"}
            continue

        image_or_tensor, target = dataset[0]
        transformed_shape = list(image_or_tensor.shape) if hasattr(image_or_tensor, "shape") else None
        original_path = str(dataset.records[0].path.resolve())
        results["sample_checks"][split_name] = {
            "status": "ok",
            "sample_path": original_path,
            "target_index": target,
            "transformed_shape": transformed_shape,
            "python_type": type(image_or_tensor).__name__,
        }

    return results


def dataset_summary_to_markdown(summary: DatasetSummary) -> str:
    """Render a compact markdown summary for notebook inclusion."""
    lines = ["## Dataset Summary", ""]
    lines.append(f"- Dataset found: {'yes' if summary.dataset_found else 'no'}")
    lines.append(f"- Expected raw-data location: `{summary.expected_raw_data_dir}`")

    if not summary.dataset_found:
        lines.append("- Notes:")
        for note in summary.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)

    lines.append(f"- Discovered root: `{summary.discovered_root}`")
    lines.append(f"- Layout: `{summary.layout}`")
    lines.append(f"- Splits found: {', '.join(summary.split_names) if summary.split_names else 'none'}")
    lines.append(f"- Classes discovered ({len(summary.class_names)}): {', '.join(summary.class_names)}")
    lines.append(f"- Total images: {summary.total_images}")
    lines.append(f"- Runtime ready for Stage 5: {'yes' if summary.runtime_ready_for_stage5 else 'no'}")
    lines.append(
        f"- Recommended data action: `{summary.recommended_data_action}`"
        if summary.recommended_data_action
        else "- Recommended data action: not set"
    )

    if summary.sample_image is not None:
        lines.append(
            "- Sample image: "
            f"`{summary.sample_image.path}` | size={summary.sample_image.size} | mode={summary.sample_image.mode}"
        )

    if summary.metadata_csv_found:
        lines.append(
            f"- Metadata CSV: `{summary.metadata_csv_path}` | rows={summary.metadata_total_rows} | "
            f"rank targets ({len(summary.metadata_rank_names)}): {', '.join(summary.metadata_rank_names)}"
        )
        lines.append(
            "- Metadata file coverage: "
            f"{summary.metadata_existing_rows} existing / {summary.metadata_missing_rows} missing"
        )
    else:
        lines.append(f"- Metadata CSV: `{METADATA_CSV_NAME}` not found")

    lines.append("")
    lines.append("### Images Per Class Per Split")
    lines.append("")
    lines.append("| Split | Class | Count |")
    lines.append("| --- | --- | ---: |")
    for split_name, class_counts in summary.counts_by_split.items():
        for class_name, count in class_counts.items():
            lines.append(f"| {split_name} | {class_name} | {count} |")

    if summary.notes:
        lines.append("")
        lines.append("### Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
