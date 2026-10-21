"""Central configuration, directory paths, and deterministic constants."""
from pathlib import Path
from typing import List

RANDOM_SEED: int = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data paths
DEFAULT_RAW_RECIPES_PATH = DATA_RAW_DIR / "train.json"
DEFAULT_LEXICON_PATH = DATA_RAW_DIR / "ingredient_taste_lexicon.csv"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"

# The 5 primary gustatory modalities (non-volatile soluble tastants activating taste buds)
TASTE_DIMENSIONS: List[str] = ["sweet", "sour", "bitter", "salty", "umami"]

# Trigeminal / chemesthetic sensations (somatosensory, mediated via TRPV1/TRPA1, not basic tastes)
CHEMESTHETIC_TAGS: List[str] = [
    "none",
    "capsaicin_heat",
    "piperine_pungency",
    "gingerol_pungency",
    "allicin_pungency",
    "isothiocyanate_pungency",
]

# Annotation confidence categories
CONFIDENCE_LEVELS: List[str] = ["high", "medium", "low_proxy"]

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
