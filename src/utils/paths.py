"""Common project path definitions."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DERIVED_RANK_DATASET_NAME = "rank14_from_local_raw"
DERIVED_RANK_DATASET_DIR = PROCESSED_DATA_DIR / DERIVED_RANK_DATASET_NAME
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_OUTPUT_DIR = OUTPUTS_DIR / "visualizations"
FEATURE_MAPS_OUTPUT_DIR = OUTPUTS_DIR / "feature_maps"
FOLDS_OUTPUT_DIR = OUTPUTS_DIR / "folds"
REPORTS_DIR = PROJECT_ROOT / "reports"
