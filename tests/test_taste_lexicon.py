"""Unit tests for taste lexicon integrity, boundaries, and chemesthesis distinctions."""
import numpy as np
import pandas as pd
import pytest

from src.config import CHEMESTHETIC_TAGS, CONFIDENCE_LEVELS, DEFAULT_LEXICON_PATH, TASTE_DIMENSIONS


@pytest.fixture
def lexicon_df() -> pd.DataFrame:
    assert DEFAULT_LEXICON_PATH.exists(), f"Lexicon file missing at {DEFAULT_LEXICON_PATH}"
    return pd.read_csv(DEFAULT_LEXICON_PATH)


def test_lexicon_columns_present(lexicon_df):
    expected_cols = [
        "ingredient_norm",
        "category",
        "sweet",
        "sour",
        "bitter",
        "salty",
        "umami",
        "chemesthetic_pungency",
        "confidence_level",
        "flavordb_id",
        "primary_aroma_compounds",
        "annotation_provenance",
    ]
    for col in expected_cols:
        assert col in lexicon_df.columns, f"Missing required column '{col}' in lexicon"


def test_lexicon_taste_ranges(lexicon_df):
    for dim in TASTE_DIMENSIONS:
        vals = lexicon_df[dim].to_numpy()
        assert not np.isnan(vals).any(), f"NaN value found in lexicon column {dim}"
        assert ((vals >= 0.0) & (vals <= 1.0)).all(), f"Values in {dim} must be in [0.0, 1.0]"


def test_lexicon_no_duplicate_ingredients(lexicon_df):
    assert lexicon_df["ingredient_norm"].is_unique, "Duplicate ingredient_norm entries in lexicon"


def test_lexicon_chemesthetic_tags(lexicon_df):
    valid_tags = set(CHEMESTHETIC_TAGS)
    for tag in lexicon_df["chemesthetic_pungency"]:
        assert tag in valid_tags, f"Invalid chemesthetic tag '{tag}'"


def test_lexicon_confidence_levels(lexicon_df):
    valid_conf = set(CONFIDENCE_LEVELS)
    for conf in lexicon_df["confidence_level"]:
        assert conf in valid_conf, f"Invalid confidence level '{conf}'"


def test_chili_pungency_distinction(lexicon_df):
    # Verify that chili heat is recorded as capsaicin chemesthesis, NOT pure taste
    chili_row = lexicon_df[lexicon_df["ingredient_norm"] == "cayenne pepper"].iloc[0]
    assert chili_row["chemesthetic_pungency"] == "capsaicin_heat"

    pepper_row = lexicon_df[lexicon_df["ingredient_norm"] == "black pepper"].iloc[0]
    assert pepper_row["chemesthetic_pungency"] == "piperine_pungency"


def test_reference_standards(lexicon_df):
    # Salt should be purely salty
    salt = lexicon_df[lexicon_df["ingredient_norm"] == "salt"].iloc[0]
    assert salt["salty"] == 1.0
    assert salt["sweet"] == 0.0
    assert salt["sour"] == 0.0

    # Sugar should be purely sweet
    sugar = lexicon_df[lexicon_df["ingredient_norm"] == "sugar"].iloc[0]
    assert sugar["sweet"] == 1.0
    assert sugar["salty"] == 0.0
