"""Central configuration, paths, and random seed definitions."""
from pathlib import Path

RANDOM_SEED: int = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

TASTE_DIMENSIONS = ["sweet", "sour", "bitter", "salty", "umami"]

# Minimum threshold of matched taste-bearing ingredients required for a recipe
# to be included in downstream taste profile and cross-cuisine analyses.
MIN_MATCHED_INGREDIENTS: int = 2
MIN_MATCH_COVERAGE_RATIO: float = 0.25


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for folder in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DOCS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)