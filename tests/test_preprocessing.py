"""Unit tests for Deliverable 1 normalization and Deliverable 2 taste profile math."""
import numpy as np

from src.data_preprocessing import normalize_ingredient_name
from src.taste_profile import compute_shannon_entropy_and_balance


def test_normalize_ingredient_basic_and_punctuation():
    assert normalize_ingredient_name("  Extra-Virgin Olive Oil (2 tbsp) ") == "olive oil"
    assert normalize_ingredient_name("Freshly Chopped Garlic Cloves!") == "garlic"


def test_preserve_meaningful_ingredient_distinctions():
    # Black garlic and regular garlic have distinct taste profiles and must not collapse
    assert normalize_ingredient_name("garlic") == "garlic"
    assert normalize_ingredient_name("black garlic") == "black garlic"
    assert normalize_ingredient_name("fresh ginger") == "ginger"
    assert normalize_ingredient_name("ground ginger") == "dried ginger"


def test_shannon_entropy_and_balance_bounds():
    # Uniform distribution across 5 tastes should yield balance == 1.0
    uniform_p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    h_unif, j_unif = compute_shannon_entropy_and_balance(uniform_p)
    assert np.isclose(j_unif, 1.0, atol=1e-3)
    assert h_unif > 1.60

    # Single dominant taste should yield entropy == 0.0 and balance == 0.0
    single_p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    h_single, j_single = compute_shannon_entropy_and_balance(single_p)
    assert h_single == 0.0
    assert j_single == 0.0