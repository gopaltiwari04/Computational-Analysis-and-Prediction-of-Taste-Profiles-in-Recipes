"""Integration tests for the deployable FastAPI web application and Vercel pure-Python engine."""
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.config import DEFAULT_ALIASES_PATH, DEFAULT_LEXICON_PATH, RunContext
from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import run_deliverable_1
from src.taste_prediction import predict_recipe_taste, run_deliverable_4
from src.taste_profile import run_deliverable_2


@pytest.fixture(scope="module")
def web_env(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("webapp_smoke")
    ctx = RunContext(
        processed_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports",
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
        manifest_path=tmp_path / "data" / "dataset_manifest.json",
        lexicon_path=DEFAULT_LEXICON_PATH,
        aliases_path=DEFAULT_ALIASES_PATH,
    )
    run_deliverable_1(use_sample_data=True, ctx=ctx)
    run_deliverable_2(ctx=ctx)
    run_deliverable_3(ctx=ctx)
    run_deliverable_4(ctx=ctx)
    return TestClient(create_app(ctx=ctx)), ctx


def test_webapp_health_and_homepage(web_env):
    client, _ = web_env
    home = client.get("/")
    assert home.status_code == 200
    assert "Computational Gastronomy" in home.text

    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["corpus"]["is_synthetic"] is True
    assert health["model"]["available"] is True


def test_webapp_analyze_and_sklearn_parity(web_env):
    client, ctx = web_env
    ingredients = ["Roma tomatoes", "fresh garlic", "extra-virgin olive oil", "fresh basil"]
    resp = client.post("/api/analyze", json={"ingredients": ingredients})
    assert resp.status_code == 200
    data = resp.json()
    assert data["lexicon_proxy"]["n_taste_matched"] == 4

    # Verify exact parity between pure-Python JSON weights engine and sklearn joblib pipeline
    sk_full = predict_recipe_taste(ingredients, model_path=ctx.models_dir / "taste_model.joblib", setting="full")
    py_full = data["ml_predictions"]["setting_1_full"]
    assert py_full["predicted_dominant_taste"] == sk_full["predicted_dominant_taste"]
    for cls, prob in sk_full["probabilities"].items():
        assert np.isclose(py_full["probabilities"][cls], prob, atol=1e-3)

    sk_masked = predict_recipe_taste(ingredients, model_path=ctx.models_dir / "taste_model.joblib", setting="masked")
    py_masked = data["ml_predictions"]["setting_2_masked"]
    assert py_masked["predicted_dominant_taste"] == sk_masked["predicted_dominant_taste"]
    for cls, prob in sk_masked["probabilities"].items():
        assert np.isclose(py_masked["probabilities"][cls], prob, atol=1e-3)

    oov = client.post("/api/analyze", json={"ingredients": ["alien_space_spice_xyz"]})
    assert oov.status_code == 200
    oov_data = oov.json()
    assert oov_data["lexicon_proxy"]["dominant_taste"] == "unspecified"
    assert oov_data["ml_predictions"]["setting_1_full"]["status"] == "no_known_ingredients"

    bad = client.post("/api/analyze", json={"ingredients": ["   "]})
    assert bad.status_code == 400


def test_webapp_cuisines_lexicon_and_similarity(web_env):
    client, _ = web_env
    cuisines = client.get("/api/cuisines").json()
    assert len(cuisines["cuisines"]) == 6

    lex = client.get("/api/lexicon?q=cooking%20oil").json()
    assert lex["count"] == 1
    assert lex["entries"][0]["sweet"] is None

    samples = client.get("/api/recipes/sample").json()["samples"]
    assert len(samples) > 0
    rid = samples[0]["recipe_id"]
    sim = client.get(f"/api/similar/{rid}?top_k=3").json()
    assert sim["query"]["recipe_id"] == rid
    assert all(n["neighbor_recipe_id"] != rid for n in sim["neighbors"])

    missing = client.get("/api/similar/-99999")
    assert missing.status_code == 404


def test_vercel_entrypoint_has_no_heavy_ml_imports():
    """Verify app.main and api.index import zero heavy scientific packages."""
    code = (
        "import sys; import app.main; import api.index; "
        "heavy = [m for m in ('numpy','pandas','scipy','sklearn','matplotlib','seaborn','joblib') if m in sys.modules]; "
        "assert not heavy, f'Heavy modules loaded by Vercel entrypoint: {heavy}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
