"""Central configuration, directory paths, and deterministic constants."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

RANDOM_SEED: int = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_RAW_RECIPES_PATH = DATA_RAW_DIR / "train.json"
DEFAULT_LEXICON_PATH = DATA_RAW_DIR / "ingredient_taste_lexicon.csv"
DEFAULT_ALIASES_PATH = DATA_RAW_DIR / "ingredient_aliases.csv"
DATASET_MANIFEST_PATH = PROJECT_ROOT / "data" / "dataset_manifest.json"

# Five primary gustatory modalities (non-volatile soluble tastants activating lingual taste receptors)
TASTE_DIMENSIONS: List[str] = ["sweet", "sour", "bitter", "salty", "umami"]

# Trigeminal / chemesthetic sensations (somatosensory TRP channels, NOT basic taste modalities)
CHEMESTHETIC_TAGS: List[str] = [
    "none",
    "capsaicin_heat",
    "piperine_pungency",
    "gingerol_pungency",
    "allicin_pungency",
    "isothiocyanate_pungency",
]

CONFIDENCE_LEVELS: List[str] = ["high", "medium", "low_proxy", "unknown"]
ANNOTATION_STATUSES: List[str] = ["reference_standard", "curated_proxy", "taste_unknown"]

# Minimum threshold of taste-bearing matched ingredients for downstream profile & cuisine analyses
MIN_MATCHED_INGREDIENTS: int = 2
MIN_MATCH_COVERAGE_RATIO: float = 0.25

# Ingredients whose max basic-taste proxy intensity meets this threshold are masked in Setting 2
ANCHOR_INTENSITY_THRESHOLD: float = 0.75


@dataclass(frozen=True)
class RunContext:
    """Filesystem targets for one pipeline run. Tests inject a temporary directory tree."""

    processed_dir: Path
    reports_dir: Path
    results_dir: Path
    models_dir: Path
    manifest_path: Path
    lexicon_path: Path
    aliases_path: Path

    def ensure(self) -> None:
        for folder in [
            self.processed_dir,
            self.reports_dir,
            self.results_dir,
            self.models_dir,
            self.manifest_path.parent,
            self.lexicon_path.parent,
            self.aliases_path.parent,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


def default_context() -> RunContext:
    """Return the default repository filesystem RunContext."""
    return RunContext(
        processed_dir=DATA_PROCESSED_DIR,
        reports_dir=REPORTS_DIR,
        results_dir=RESULTS_DIR,
        models_dir=MODELS_DIR,
        manifest_path=DATASET_MANIFEST_PATH,
        lexicon_path=DEFAULT_LEXICON_PATH,
        aliases_path=DEFAULT_ALIASES_PATH,
    )


def ensure_directories(ctx: Optional[RunContext] = None) -> None:
    """Create required project directories if they do not exist."""
    (ctx or default_context()).ensure()
    if ctx is None:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    """Return a repo-relative POSIX path for reports (never leak machine-local absolute paths)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def setup_logging(verbose: bool = False) -> None:
    """Configure deterministic console logging without noisy third-party debug spam."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
