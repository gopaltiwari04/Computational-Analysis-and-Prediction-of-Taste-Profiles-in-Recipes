"""Unit tests for Deliverable 1 ingredient normalization, alias resolution, and dataset loading."""
from pathlib import Path
import pytest

from src.data_preprocessing import (
    generate_sample_recipes_dataset,
    load_raw_recipes,
    normalize_ingredient_name,
)


def test_normalize_ingredient_basic_and_punctuation():
    assert normalize_ingredient_name("  Extra-Virgin Olive Oil (2 tbsp) ") == "olive oil"
    assert normalize_ingredient_name("Freshly Chopped Garlic Cloves!") == "garlic"
    assert normalize_ingredient_name("kosher salt") == "salt"
    assert normalize_ingredient_name("sea salt") == "salt"
    assert normalize_ingredient_name("diced yellow onions") == "onion"


def test_preserve_meaningful_ingredient_distinctions():
    # Black garlic and regular garlic have distinct taste profiles and must not collapse
    assert normalize_ingredient_name("garlic") == "garlic"
    assert normalize_ingredient_name("black garlic") == "black garlic"

    # Fresh ginger vs dried ground ginger
    assert normalize_ingredient_name("fresh ginger") == "ginger"
    assert normalize_ingredient_name("ground ginger") == "dried ginger"

    # Sweet bell pepper vs spicy cayenne chili
    assert normalize_ingredient_name("bell pepper") == "bell pepper"
    assert normalize_ingredient_name("red bell pepper") == "bell pepper"
    assert normalize_ingredient_name("cayenne pepper") == "cayenne pepper"
    assert normalize_ingredient_name("red chili powder") == "cayenne pepper"

    # Unsalted butter vs salted butter
    assert normalize_ingredient_name("butter") == "butter"
    assert normalize_ingredient_name("unsalted butter") == "butter"
    assert normalize_ingredient_name("salted butter") == "salted butter"


def test_normalize_plurals_and_aliases():
    assert normalize_ingredient_name("onions") == "onion"
    assert normalize_ingredient_name("carrots") == "carrot"
    assert normalize_ingredient_name("tomatoes") == "tomato"
    assert normalize_ingredient_name("eggs") == "egg"
    assert normalize_ingredient_name("scallions") == "green onion"
    assert normalize_ingredient_name("green onions") == "green onion"
    assert normalize_ingredient_name("shallots") == "shallot"
    assert normalize_ingredient_name("purple onion") == "red onion"
    assert normalize_ingredient_name("parmigiano-reggiano") == "parmesan cheese"


def test_normalize_edge_cases():
    assert normalize_ingredient_name("") == ""
    assert normalize_ingredient_name("   ") == ""
    assert normalize_ingredient_name(None) == ""
    assert normalize_ingredient_name(123) == ""
    assert normalize_ingredient_name("(optional) 1/2 cup water") == "water"


def test_synthetic_sample_generator():
    data = generate_sample_recipes_dataset(seed=42)
    assert len(data) >= 180
    assert any(r["cuisine"] == "Italian" for r in data)
    assert any(r["cuisine"] == "Indian" for r in data)
    assert any(r["cuisine"] == "Mexican" for r in data)
    assert any(r["cuisine"] == "Japanese" for r in data)
    assert any(r["cuisine"] == "Thai" for r in data)

    # Injected edge cases verification
    # Duplicate ID
    ids = [r["id"] for r in data]
    assert len(ids) > len(set(ids)), "Injected duplicate ID not found in synthetic generator"
    # Empty cuisine record
    assert any(r["cuisine"] == "" for r in data), "Injected empty cuisine record missing"
    # Empty ingredients record
    assert any(len(r["ingredients"]) == 0 for r in data), "Injected empty ingredients record missing"


def test_load_raw_recipes_strictness():
    # When file does not exist and use_sample_data is False, must raise FileNotFoundError
    non_existent = Path("data/raw/fictional_non_existent_recipe_file_12345.json")
    with pytest.raises(FileNotFoundError):
        load_raw_recipes(raw_json_path=non_existent, use_sample_data=False)

    # When use_sample_data is True, should return DataFrame with synthetic flag
    df, meta = load_raw_recipes(raw_json_path=non_existent, use_sample_data=True)
    assert len(df) > 0
    assert meta["is_synthetic"] is True
