"""Unit tests for Deliverable 4 ML leakage prevention, anchor masking, persistence, inference, and retrieval."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_ALIASES_PATH, DEFAULT_LEXICON_PATH, TASTE_DIMENSIONS
from src.data_preprocessing import load_aliases
from src.taste_prediction import (
    build_anchor_set,
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


def test_anchor_masking_prevents_alias_bypass():
    anchors = build_anchor_set(DEFAULT_LEXICON_PATH)
    aliases = load_aliases(DEFAULT_ALIASES_PATH)
    raw_str = "garlic;salt;confectioners sugar;dark chocolate;coarse sea salt;olive oil;tomato"
    masked = mask_ingredient_string(raw_str, anchors, aliases=aliases)
    remaining = tokenize_ingredients(masked)
    assert "salt" not in remaining
    assert "confectioners sugar" not in remaining
    assert "dark chocolate" not in remaining
    assert "coarse sea salt" not in remaining
    assert "garlic" in remaining
    assert "olive oil" in remaining
    assert "tomato" in remaining


def test_data_leakage_prevention_in_vectorizer():
    train_corpus = ["garlic;onion;olive oil", "tomato;basil;garlic", "sugar;butter;milk", "apple;cinnamon;butter"]
    test_corpus = ["garlic;novel_secret_ingredient;tomato"]
    y_train = np.array(["umami", "umami", "sweet", "sweet"])
    y_test = np.array(["umami"])

    _, pipelines, _ = train_and_evaluate_models(
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        y_train=y_train,
        y_test=y_test,
        setting_label="Leakage_Test",
    )
    vocab = pipelines["LogisticRegression (L2, Balanced)"].named_steps["vectorizer"].vocabulary_
    assert "novel_secret_ingredient" not in vocab
    assert "garlic" in vocab


def test_model_persistence_and_inference_edges(tmp_path):
    train_corpus = [
        "garlic;onion;tomato;olive oil",
        "tomato;basil;garlic;mushroom",
        "sugar;butter;wheat flour;vanilla",
        "brown sugar;cinnamon;apple;butter",
        "lemon juice;vinegar;tomato;garlic",
        "lime juice;cilantro;vinegar;onion",
    ]
    y_train = np.array(["umami", "umami", "sweet", "sweet", "sour", "sour"])
    _, pipelines, _ = train_and_evaluate_models(
        train_corpus=train_corpus,
        test_corpus=train_corpus[:2],
        y_train=y_train,
        y_test=y_train[:2],
        setting_label="Unit_Bundle",
    )
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipelines["LogisticRegression (L2, Balanced)"],
            "masked_pipeline": pipelines["LogisticRegression (L2, Balanced)"],
            "evaluation_setting": "Setting 1: Full_Ingredients",
            "is_synthetic": True,
        },
        bundle_path,
    )

    res = predict_recipe_taste(["Roma tomatoes", "garlic cloves", "fresh basil", "olive oil"], model_path=bundle_path)
    assert res["status"] == "success"
    assert res["predicted_dominant_taste"] in TASTE_DIMENSIONS
    assert np.isclose(sum(res["probabilities"].values()), 1.0, atol=1e-3)
    assert "tomato" in res["known_ingredients"]

    empty_res = predict_recipe_taste([], model_path=bundle_path)
    assert empty_res["status"] == "empty_input"
    assert empty_res["predicted_dominant_taste"] == "unspecified"

    oov_res = predict_recipe_taste(["fictional_alien_ingredient_xyz_123"], model_path=bundle_path)
    assert oov_res["status"] == "no_known_ingredients"
    assert oov_res["predicted_dominant_taste"] == "unspecified"
    assert oov_res["known_ingredients"] == []


def test_similarity_retrieval_handles_non_contiguous_index_and_errors():
    # Intentionally slice rows so DataFrame .index is [10, 25, 42, 99] (non-contiguous!)
    df = pd.DataFrame(
        {
            "recipe_id": [101, 102, 103, 104],
            "cuisine": ["italian", "mexican", "indian", "japanese"],
            "dominant_taste": ["umami", "sour", "bitter", "umami"],
            "taste_evenness": [0.85, 0.88, 0.82, 0.86],
            "sweet": [0.1, 0.2, 0.1, 0.15],
            "sour": [0.2, 0.5, 0.1, 0.1],
            "bitter": [0.1, 0.1, 0.6, 0.1],
            "salty": [0.2, 0.1, 0.1, 0.25],
            "umami": [0.4, 0.1, 0.1, 0.4],
        },
        index=[10, 25, 42, 99],
    )
    neighbors = retrieve_similar_recipes(df, query_recipe_id=101, top_k=2)
    assert len(neighbors) == 2
    assert 101 not in neighbors["neighbor_recipe_id"].values
    assert (neighbors["cosine_similarity"] >= -1.0).all()
    assert (neighbors["cosine_similarity"] <= 1.0).all()

    with pytest.raises(ValueError):
        retrieve_similar_recipes(df, query_recipe_id=-999, top_k=2)
    with pytest.raises(ValueError):
        retrieve_similar_recipes(df, query_recipe_id=101, top_k=0)
    with pytest.raises(ValueError):
        retrieve_similar_recipes(df.iloc[[0]], query_recipe_id=101, top_k=1)
