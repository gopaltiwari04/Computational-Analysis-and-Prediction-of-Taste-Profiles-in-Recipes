"""Unit tests for Deliverable 1 normalization, alias resolution, validation, and deduplication."""
from pathlib import Path
import json
import pandas as pd
import pytest

from src.config import DEFAULT_ALIASES_PATH, DEFAULT_LEXICON_PATH, RunContext
from src.data_preprocessing import (
    generate_sample_recipes_dataset,
    load_raw_recipes,
    normalize_ingredient_name,
    run_deliverable_1,
    validate_raw_recipes_df,
)


def test_normalize_ingredient_basic_and_punctuation():
    assert normalize_ingredient_name("  Extra-Virgin Olive Oil (2 tbsp) ") == "olive oil"
    assert normalize_ingredient_name("Freshly Chopped Garlic Cloves!") == "garlic"
    assert normalize_ingredient_name("kosher salt") == "salt"
    assert normalize_ingredient_name("sea salt") == "salt"
    assert normalize_ingredient_name("diced yellow onions") == "onion"


def test_preserve_meaningful_ingredient_distinctions():
    # Distinct alliums must stay separated
    assert normalize_ingredient_name("garlic") == "garlic"
    assert normalize_ingredient_name("black garlic") == "black garlic"
    assert normalize_ingredient_name("garlic powder") == "garlic powder"
    assert normalize_ingredient_name("onion powder") == "onion powder"

    # Fresh vs dried ginger
    assert normalize_ingredient_name("fresh ginger") == "ginger"
    assert normalize_ingredient_name("ground ginger") == "dried ginger"

    # Bell pepper vs cayenne vs chili powder vs bare pepper
    assert normalize_ingredient_name("bell pepper") == "bell pepper"
    assert normalize_ingredient_name("red bell pepper") == "bell pepper"
    assert normalize_ingredient_name("cayenne pepper") == "cayenne pepper"
    assert normalize_ingredient_name("red chili powder") == "cayenne pepper"
    assert normalize_ingredient_name("chili powder") == "chili powder"
    assert normalize_ingredient_name("pepper") == "pepper"

    # Whole fruit vs juice
    assert normalize_ingredient_name("lemon") == "lemon"
    assert normalize_ingredient_name("lemon juice") == "lemon juice"
    assert normalize_ingredient_name("lime") == "lime"
    assert normalize_ingredient_name("lime juice") == "lime juice"

    # Stocks vs muscle meats; salted vs unsalted butter
    assert normalize_ingredient_name("beef") == "beef"
    assert normalize_ingredient_name("beef broth") == "beef broth"
    assert normalize_ingredient_name("butter") == "butter"
    assert normalize_ingredient_name("unsalted butter") == "butter"
    assert normalize_ingredient_name("salted butter") == "salted butter"


def test_normalize_plurals_typos_and_aliases():
    assert normalize_ingredient_name("onions") == "onion"
    assert normalize_ingredient_name("carrots") == "carrot"
    assert normalize_ingredient_name("tomatoes") == "tomato"
    assert normalize_ingredient_name("eggs") == "egg"
    assert normalize_ingredient_name("egg yolks") == "egg"
    assert normalize_ingredient_name("scallions") == "green onion"
    assert normalize_ingredient_name("green onions") == "green onion"
    assert normalize_ingredient_name("shallots") == "shallot"
    assert normalize_ingredient_name("purple onion") == "red onion"
    assert normalize_ingredient_name("parmigiano-reggiano") == "parmesan cheese"
    assert normalize_ingredient_name("mayonaise") == "mayonnaise"
    assert normalize_ingredient_name("ground tumeric") == "turmeric"
    assert normalize_ingredient_name("confectioners sugar") == "powdered sugar"


def test_normalize_edge_cases():
    assert normalize_ingredient_name("") == ""
    assert normalize_ingredient_name("   ") == ""
    assert normalize_ingredient_name(None) == ""
    assert normalize_ingredient_name(123) == ""
    assert normalize_ingredient_name("(optional) 1/2 cup water") == "water"
    assert normalize_ingredient_name("123 !@#") == ""


def test_validate_raw_recipes_df_and_negative_cases():
    bad_df = pd.DataFrame([{"id": 1, "cuisine": "italian"}])  # missing 'ingredients'
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_recipes_df(bad_df)

    mixed_df = pd.DataFrame(
        [
            {"id": 1, "cuisine": "italian", "ingredients": ["tomato", "garlic"]},
            {"id": None, "cuisine": "italian", "ingredients": ["salt"]},
            {"id": 2, "cuisine": "   ", "ingredients": ["salt"]},
            {"id": 3, "cuisine": "indian", "ingredients": []},
            {"id": 4, "cuisine": "french", "ingredients": ["1/2", "(optional)"]},
        ]
    )
    valid_df, breakdown = validate_raw_recipes_df(mixed_df)
    assert len(valid_df) == 1
    assert int(valid_df.iloc[0]["id"]) == 1
    assert breakdown["missing_or_invalid_id"] == 1
    assert breakdown["missing_cuisine"] == 1
    assert breakdown["invalid_or_empty_ingredients"] == 1
    assert breakdown["empty_after_normalization"] == 1


def test_cross_cuisine_deduplication_preservation(tmp_path):
    recipes = [
        {"id": 1, "cuisine": "italian", "ingredients": ["garlic", "olive oil", "tomato"]},
        {"id": 2, "cuisine": "italian", "ingredients": ["tomato", "olive oil", "garlic"]},  # same-cuisine dup
        {"id": 3, "cuisine": "spanish", "ingredients": ["garlic", "olive oil", "tomato"]},  # cross-cuisine valid
    ]
    raw_path = tmp_path / "recipes.json"
    raw_path.write_text(json.dumps(recipes), encoding="utf-8")

    ctx = RunContext(
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        results_dir=tmp_path / "results",
        models_dir=tmp_path / "models",
        manifest_path=tmp_path / "manifest.json",
        lexicon_path=DEFAULT_LEXICON_PATH,
        aliases_path=DEFAULT_ALIASES_PATH,
    )
    stats = run_deliverable_1(raw_data_path=raw_path, use_sample_data=False, ctx=ctx)
    assert stats["n_recipes_clean"] == 2
    assert stats["within_cuisine_duplicates_removed"] == 1
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["cross_cuisine_shared_signatures_preserved"] == 1
    assert manifest["is_synthetic"] is False


def test_synthetic_sample_generator():
    data = generate_sample_recipes_dataset(seed=42)
    assert len(data) >= 180
    ids = [r["id"] for r in data]
    assert len(ids) > len(set(ids))
    assert any(r["cuisine"] == "" for r in data)
    assert any(len(r["ingredients"]) == 0 for r in data)


def test_load_raw_recipes_strictness(tmp_path):
    non_existent = tmp_path / "missing_recipes.json"
    with pytest.raises(FileNotFoundError):
        load_raw_recipes(raw_json_path=non_existent, use_sample_data=False)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_recipes(raw_json_path=bad_json, use_sample_data=False)

    df, meta = load_raw_recipes(raw_json_path=non_existent, use_sample_data=True)
    assert len(df) > 0
    assert meta["is_synthetic"] is True
