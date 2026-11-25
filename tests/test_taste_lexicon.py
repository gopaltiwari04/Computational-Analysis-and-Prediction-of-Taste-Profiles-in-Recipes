"""Unit tests for verified taste lexicon integrity, unknown values, FlavorDB IDs, and chemesthesis."""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ANNOTATION_STATUSES,
    CHEMESTHETIC_TAGS,
    CONFIDENCE_LEVELS,
    DEFAULT_LEXICON_PATH,
    TASTE_DIMENSIONS,
)
from src.data_preprocessing import load_lexicon, validate_lexicon_df


@pytest.fixture
def lexicon_df() -> pd.DataFrame:
    assert DEFAULT_LEXICON_PATH.exists(), f"Lexicon file missing at {DEFAULT_LEXICON_PATH}"
    return load_lexicon(DEFAULT_LEXICON_PATH)


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
        "annotation_status",
        "flavordb_entity_id",
        "flavordb_entity_name",
        "primary_aroma_compounds",
        "known_tastant_or_pungent_compounds",
        "annotation_provenance",
    ]
    for col in expected_cols:
        assert col in lexicon_df.columns, f"Missing required column '{col}' in lexicon"


def test_lexicon_taste_ranges_and_unknown_handling(lexicon_df):
    for dim in TASTE_DIMENSIONS:
        vals = lexicon_df[dim].dropna().to_numpy(dtype=float)
        assert len(vals) > 0
        assert ((vals >= 0.0) & (vals <= 1.0)).all(), f"Values in {dim} must be in [0.0, 1.0]"

    # Taste-unknown lipids must have NaN across all 5 basic tastes rather than invented values
    oils = lexicon_df[lexicon_df["annotation_status"] == "taste_unknown"]
    assert len(oils) >= 2
    for _, row in oils.iterrows():
        assert all(pd.isna(row[d]) for d in TASTE_DIMENSIONS)
        assert row["confidence_level"] == "unknown"


def test_lexicon_no_duplicate_ingredients_and_valid_tags(lexicon_df):
    assert lexicon_df["ingredient_norm"].is_unique
    assert set(lexicon_df["chemesthetic_pungency"]).issubset(set(CHEMESTHETIC_TAGS))
    assert set(lexicon_df["confidence_level"]).issubset(set(CONFIDENCE_LEVELS))
    assert set(lexicon_df["annotation_status"]).issubset(set(ANNOTATION_STATUSES))


def test_lexicon_flavordb_ids_not_fabricated(lexicon_df):
    for _, row in lexicon_df.iterrows():
        eid = str(row["flavordb_entity_id"])
        assert eid == "NONE" or eid.isdigit(), f"Unverified or fabricated FlavorDB ID '{eid}'"
        assert not eid.startswith("FDB000"), "Legacy fabricated FDB000XXX identifier found"

    # Spot-check verified FlavorDB entity IDs
    tomato = lexicon_df[lexicon_df["ingredient_norm"] == "tomato"].iloc[0]
    assert str(tomato["flavordb_entity_id"]) == "364"
    garlic = lexicon_df[lexicon_df["ingredient_norm"] == "garlic"].iloc[0]
    assert str(garlic["flavordb_entity_id"]) == "259"


def test_chili_pungency_and_reference_standards(lexicon_df):
    chili = lexicon_df[lexicon_df["ingredient_norm"] == "cayenne pepper"].iloc[0]
    assert chili["chemesthetic_pungency"] == "capsaicin_heat"

    pepper = lexicon_df[lexicon_df["ingredient_norm"] == "black pepper"].iloc[0]
    assert pepper["chemesthetic_pungency"] == "piperine_pungency"

    salt = lexicon_df[lexicon_df["ingredient_norm"] == "salt"].iloc[0]
    assert salt["salty"] == 1.0 and salt["sweet"] == 0.0 and salt["sour"] == 0.0
    assert salt["annotation_status"] == "reference_standard"

    sugar = lexicon_df[lexicon_df["ingredient_norm"] == "sugar"].iloc[0]
    assert sugar["sweet"] == 1.0 and sugar["salty"] == 0.0
    assert sugar["annotation_status"] == "reference_standard"


def test_validate_lexicon_rejects_invalid_entries(lexicon_df):
    bad_range = lexicon_df.copy()
    bad_range.loc[0, "sweet"] = 1.5
    with pytest.raises(ValueError, match="must lie in"):
        validate_lexicon_df(bad_range)

    bad_id = lexicon_df.copy()
    bad_id.loc[0, "flavordb_entity_id"] = "FDB000999"
    with pytest.raises(ValueError, match="Invalid flavordb_entity_id"):
        validate_lexicon_df(bad_id)

    bad_dup = pd.concat([lexicon_df.iloc[[0]], lexicon_df], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_lexicon_df(bad_dup)
