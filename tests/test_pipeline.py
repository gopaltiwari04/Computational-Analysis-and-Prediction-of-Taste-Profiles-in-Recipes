"""Isolated end-to-end pipeline integration tests using tmp_path + RunContext."""
import json
import pytest

from src.config import DEFAULT_ALIASES_PATH, DEFAULT_LEXICON_PATH, RunContext
from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import run_deliverable_1
from src.taste_prediction import predict_recipe_taste, run_deliverable_4
from src.taste_profile import run_deliverable_2


def test_isolated_synthetic_smoke_pipeline(tmp_path):
    """
    Execute all 4 deliverables end-to-end inside an isolated tmp_path RunContext,
    verifying every artifact, synthetic provenance tagging, and standalone inference
    without modifying real repository data/results/models/reports.
    """
    ctx = RunContext(
        processed_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports",
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
        manifest_path=tmp_path / "data" / "dataset_manifest.json",
        lexicon_path=DEFAULT_LEXICON_PATH,
        aliases_path=DEFAULT_ALIASES_PATH,
    )

    d1 = run_deliverable_1(use_sample_data=True, ctx=ctx)
    assert d1["is_synthetic"] is True
    assert d1["n_recipes_clean"] > 0
    assert d1["n_sufficient"] > 0
    assert (ctx.processed_dir / "recipes_clean.csv").exists()
    assert (ctx.processed_dir / "recipe_ingredients.csv").exists()
    assert (ctx.processed_dir / "ingredient_flavor_mapping.csv").exists()
    assert (ctx.reports_dir / "dataset_summary.md").exists()
    assert "SYNTHETIC" in (ctx.reports_dir / "dataset_summary.md").read_text(encoding="utf-8")

    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] is True

    d2 = run_deliverable_2(ctx=ctx)
    assert len(d2) > 0
    assert (ctx.processed_dir / "recipe_taste_features.csv").exists()
    assert (ctx.results_dir / "taste_profile_distributions.png").exists()
    assert (ctx.results_dir / "example_recipe_profiles.png").exists()
    assert "SYNTHETIC" in (ctx.reports_dir / "taste_profile_methodology.md").read_text(encoding="utf-8")

    d3 = run_deliverable_3(ctx=ctx)
    assert len(d3) > 0
    assert (ctx.results_dir / "cuisine_comparison.csv").exists()
    assert (ctx.results_dir / "cuisine_taste_heatmap.png").exists()
    assert (ctx.results_dir / "cuisine_taste_boxplots.png").exists()
    assert (ctx.results_dir / "cuisine_distinctive_tastes.png").exists()
    assert (ctx.results_dir / "cuisine_pca_projection.png").exists()
    assert "SYNTHETIC" in (ctx.reports_dir / "cross_cuisine_analysis.md").read_text(encoding="utf-8")

    d4 = run_deliverable_4(ctx=ctx)
    assert len(d4) == 6
    model_path = ctx.models_dir / "taste_model.joblib"
    assert model_path.exists()
    assert (ctx.results_dir / "model_comparison.csv").exists()
    assert (ctx.results_dir / "confusion_matrix.png").exists()
    assert (ctx.results_dir / "feature_importance.png").exists()
    assert "SYNTHETIC" in (ctx.reports_dir / "prediction_evaluation.md").read_text(encoding="utf-8")

    pred_full = predict_recipe_taste(["tomato", "garlic", "basil", "olive oil"], model_path=model_path, setting="full")
    pred_masked = predict_recipe_taste(["tomato", "garlic", "basil", "olive oil"], model_path=model_path, setting="masked")
    assert pred_full["status"] == "success"
    assert pred_masked["status"] == "success"
    assert pred_full["is_synthetic_model"] is True
