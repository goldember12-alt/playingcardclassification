"""Build a reproducible 14-rank dataset from the locally available card images."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.dataset import EXPECTED_CARD_CLASSES, IMAGE_EXTENSIONS
from src.utils.paths import DERIVED_RANK_DATASET_DIR, RAW_DATA_DIR


@dataclass(frozen=True)
class DerivedRankDatasetSummary:
    """Summary of the locally derived 14-rank dataset."""

    output_root: str
    manifest_csv: str
    summary_json: str
    link_mode: str
    total_images: int
    rank_counts: dict[str, int]
    source_split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def derive_rank_label(source_label: str) -> str:
    normalized = source_label.strip().casefold()
    if normalized == "joker":
        return "joker"
    if " of " in normalized:
        return normalized.split(" of ", maxsplit=1)[0].strip()
    raise ValueError(f"Could not derive a rank label from source class '{source_label}'.")


def _safe_filename_part(value: str) -> str:
    collapsed = "_".join(value.strip().casefold().split())
    return "".join(character for character in collapsed if character.isalnum() or character in {"_", "-"})


def _create_link_or_copy(source_path: Path, destination_path: Path) -> str:
    try:
        os.link(source_path, destination_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "copy"


def build_derived_rank_dataset(
    raw_data_dir: str | Path = RAW_DATA_DIR,
    output_root: str | Path = DERIVED_RANK_DATASET_DIR,
) -> DerivedRankDatasetSummary:
    """Create a flat 14-class dataset from the existing raw image folders."""
    raw_data_dir = Path(raw_data_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for class_name in EXPECTED_CARD_CLASSES:
        (output_root / class_name).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    rank_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()

    for split_dir in sorted((path for path in raw_data_dir.iterdir() if path.is_dir()), key=lambda item: item.name.casefold()):
        split_name = split_dir.name.casefold()
        for class_dir in sorted((path for path in split_dir.iterdir() if path.is_dir()), key=lambda item: item.name.casefold()):
            source_label = class_dir.name
            derived_rank = derive_rank_label(source_label)
            destination_dir = output_root / derived_rank
            for image_path in sorted((path for path in class_dir.iterdir() if _is_image_file(path)), key=lambda item: item.name.casefold()):
                destination_name = (
                    f"{split_name}__{_safe_filename_part(source_label)}__{image_path.stem.casefold()}{image_path.suffix.lower()}"
                )
                destination_path = destination_dir / destination_name
                if not destination_path.exists():
                    link_mode = _create_link_or_copy(image_path, destination_path)
                    link_modes[link_mode] += 1

                manifest_rows.append(
                    {
                        "item_id": str(destination_path.resolve()),
                        "path": str(destination_path.resolve()),
                        "relative_path": str(destination_path.relative_to(output_root)),
                        "class_name": derived_rank,
                        "source_label": source_label,
                        "source_split": split_name,
                        "source_path": str(image_path.resolve()),
                    }
                )
                rank_counts[derived_rank] += 1
                source_split_counts[split_name] += 1

    manifest_rows.sort(key=lambda row: (str(row["class_name"]), str(row["relative_path"])))

    manifest_csv = output_root / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "path", "relative_path", "class_name", "source_label", "source_split", "source_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = DerivedRankDatasetSummary(
        output_root=str(output_root),
        manifest_csv=str(manifest_csv),
        summary_json=str((output_root / "summary.json")),
        link_mode="hardlink" if link_modes.get("copy", 0) == 0 else "mixed",
        total_images=sum(rank_counts.values()),
        rank_counts={class_name: int(rank_counts.get(class_name, 0)) for class_name in EXPECTED_CARD_CLASSES},
        source_split_counts=dict(sorted(source_split_counts.items())),
    )

    with Path(summary.summary_json).open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)

    return summary


if __name__ == "__main__":
    result = build_derived_rank_dataset()
    print(json.dumps(result.to_dict(), indent=2))
