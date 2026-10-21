"""End-to-end integration and smoke tests for the 4-deliverable pipeline."""
import json
import pytest

from src.config import (
    DATA_PROCESSED_DIR,
    DATASET_MANIFEST_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
)
from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import run_deliverable_1
from src.taste_prediction import run_deliverable_4
from src.taste_profile import run_deliverable_2


def test_synthetic_smoke_pipeline():
    """
    Smoke test: run all 4 deliverables end-to-end using synthetic data
    to verify pipeline contracts and artifact generation.
    """
    # 1. Deliverable 1
    d1_stats = run_deliverable_1(use_sample_data=True)
    assert d1_stats["is_synthetic"] is True
    assert d1_stats["n_recipes_clean"] > 0
    assert d1_stats["n_sufficient"] > 0

    assert (DATA_PROCESSED_DIR / "recipes_clean.csv").exists()
    assert (DATA_PROCESSED_DIR / "recipe_ingredients.csv").exists()
    assert (DATA_PROCESSED_DIR / "ingredient_flavor_mapping.csv").exists()
    assert DATASET_MANIFEST_PATH.exists()

    with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["is_synthetic"] is True

    # 2. Deliverable 2
    d2_df = run_deliverable_2()
    assert len(d2_df) > 0
    assert (DATA_PROCESSED_DIR / "recipe_taste_features.csv").exists()
    assert (RESULTS_DIR / "taste_profile_distributions.png").exists()
    assert (RESULTS_DIR / "example_recipe_profiles.png").exists()
    assert (REPORTS_DIR / "taste_profile_methodology.md").exists()

    # 3. Deliverable 3
    d3_df = run_deliverable_3()
    assert len(d3_df) > 0
    assert (RESULTS_DIR / "cuisine_comparison.csv").exists()
    assert (RESULTS_DIR / "cuisine_taste_heatmap.png").exists()
    assert (RESULTS_DIR / "cuisine_taste_boxplots.png").exists()
    assert (RESULTS_DIR / "cuisine_distinctive_tastes.png").exists()
    assert (RESULTS_DIR / "cuisine_pca_projection.png").exists()
    assert (REPORTS_DIR / "cross_cuisine_analysis.md").exists()

    # 4. Deliverable 4
    d4_df = run_deliverable_4()
    assert len(d4_df) > 0
    assert (MODELS_DIR / "taste_model.joblib").exists()
    assert (RESULTS_DIR / "model_comparison.csv").exists()
    assert (RESULTS_DIR / "confusion_matrix.png").exists()
    assert (RESULTS_DIR / "feature_importance.png").exists()
    assert (REPORTS_DIR / "prediction_evaluation.md").exists()
