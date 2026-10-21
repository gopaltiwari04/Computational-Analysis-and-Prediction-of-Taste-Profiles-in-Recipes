"""Unit tests for Deliverable 4 ML pipeline, data leakage prevention, inference, and retrieval."""
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, MODELS_DIR
from src.taste_prediction import (
    DIRECT_TASTE_ANCHORS,
    mask_ingredient_string,
    predict_recipe_taste,
    retrieve_similar_recipes,
    tokenize_ingredients,
    train_and_evaluate_models,
)


def test_tokenize_ingredients():
    assert tokenize_ingredients("garlic;tomato;olive oil") == ["garlic", "tomato", "olive oil"]
    assert tokenize_ingredients("  garlic ; tomato  ") == ["garlic", "tomato"]
    assert tokenize_ingredients("") == []
    assert tokenize_ingredients(None) == []


def test_mask_ingredient_string():
    raw_str = "garlic;salt;tomato;sugar;olive oil"
    masked = mask_ingredient_string(raw_str, DIRECT_TASTE_ANCHORS)
    remaining = masked.split(";")
    assert "salt" not in remaining
    assert "sugar" not in remaining
    assert "garlic" in remaining
    assert "tomato" in remaining
    assert "olive oil" in remaining


def test_data_leakage_prevention_in_vectorizer():
    # Verify that tokens appearing exclusively in test_corpus are NOT present in fitted vocabulary
    train_corpus = ["garlic;onion;olive oil", "tomato;basil;garlic"]
    test_corpus = ["garlic;novel_secret_ingredient;tomato"]

    y_train = np.array(["umami", "sour"])
    y_test = np.array(["umami"])

    results, pipelines, preds = train_and_evaluate_models(
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        y_train=y_train,
        y_test=y_test,
        setting_label="Leakage_Test",
    )

    vec = pipelines["LogisticRegression (L2, Balanced)"].named_steps["vectorizer"]
    vocab = vec.vocabulary_

    # novel_secret_ingredient was only in test, so it MUST NOT be in the vocabulary
    assert "novel_secret_ingredient" not in vocab, "Leakage detected: test token found in vectorizer vocabulary"
    assert "garlic" in vocab


def test_model_persistence_and_inference():
    # Verify that the serialized model artifact can be loaded and produces valid inference
    model_path = MODELS_DIR / "taste_model.joblib"
    if not model_path.exists():
        pytest.skip("taste_model.joblib not yet trained")

    test_ingredients = ["tomatoes", "garlic cloves", "fresh basil", "olive oil"]
    result = predict_recipe_taste(test_ingredients, model_path=model_path)

    assert result["status"] == "success"
    assert result["predicted_dominant_taste"] in ["sweet", "sour", "bitter", "salty", "umami"]
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)
    prob_sum = sum(result["probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-3)


def test_inference_empty_and_unknown_inputs():
    model_path = MODELS_DIR / "taste_model.joblib"
    if not model_path.exists():
        pytest.skip("taste_model.joblib not yet trained")

    # Empty input
    empty_res = predict_recipe_taste([])
    assert empty_res["status"] == "empty_input"
    assert empty_res["predicted_dominant_taste"] == "unspecified"

    # Completely unknown ingredients
    unknown_res = predict_recipe_taste(["fictional_alien_ingredient_xyz_123"])
    assert unknown_res["status"] == "success"


def test_similarity_retrieval_excludes_self():
    csv_path = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    if not csv_path.exists():
        pytest.skip("recipe_taste_features.csv not yet generated")

    df = pd.read_csv(csv_path)
    sample_id = int(df.iloc[0]["recipe_id"])

    neighbors = retrieve_similar_recipes(df, query_recipe_id=sample_id, top_k=3)
    assert len(neighbors) == 3

    # Invariant: Query recipe is NEVER returned as a neighbor of itself
    assert sample_id not in neighbors["neighbor_recipe_id"].values

    # Invariant: Cosine similarity is in [-1.0, 1.0]
    assert (neighbors["cosine_similarity"] >= -1.0).all()
    assert (neighbors["cosine_similarity"] <= 1.0).all()


def test_similarity_retrieval_invalid_id():
    csv_path = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    if not csv_path.exists():
        pytest.skip("recipe_taste_features.csv not yet generated")

    df = pd.read_csv(csv_path)
    with pytest.raises(ValueError):
        retrieve_similar_recipes(df, query_recipe_id=-999999, top_k=3)
