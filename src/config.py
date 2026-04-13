"""Shared configuration scaffolding for staged project development."""

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import DERIVED_RANK_DATASET_NAME


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass(frozen=True)
class RuntimeConfig:
    """Minimal runtime configuration shared by the early pipeline stages."""

    random_seed: int = 42
    num_folds: int = 5
    num_classes: int = 14
    image_size: int = 224
    model_name: str = "resnet50"
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    unfreeze_from: str | None = None
    classifier_hidden_dim: int | None = None
    classifier_dropout: float = 0.0
    train_num_epochs: int = 5
    train_batch_size: int = 16
    train_learning_rate: float = 1e-3
    train_weight_decay: float = 0.0
    train_num_workers: int = 0
    train_device: str = "auto"
    raw_data_dir: str = "data/processed"
    dataset_name: str | None = DERIVED_RANK_DATASET_NAME
    dataset_label_granularity: str = "rank"
    metadata_csv_name: str = "cards.csv"
    expected_class_names: tuple[str, ...] = EXPECTED_CARD_CLASSES
