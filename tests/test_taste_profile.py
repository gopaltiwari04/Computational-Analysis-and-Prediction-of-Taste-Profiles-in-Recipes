"""Unit tests for Deliverable 2 taste profile mathematical formulations and invariants."""
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, TASTE_DIMENSIONS
from src.taste_profile import (
    aggregate_recipe_taste_vector,
    compute_shannon_entropy_and_balance,
    normalize_to_simplex,
)


def test_shannon_entropy_uniform_distribution():
    # Uniform across all 5 dimensions: H = ln(5) approx 1.609438, J = 1.0
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    h, j = compute_shannon_entropy_and_balance(p)
    expected_h = float(np.log(5))
    assert np.isclose(h, expected_h, atol=1e-3)
    assert np.isclose(j, 1.0, atol=1e-3)


def test_shannon_entropy_delta_distribution():
    # Single dominant modality: H = 0.0, J = 0.0
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    h, j = compute_shannon_entropy_and_balance(p)
    assert h == 0.0
    assert j == 0.0


def test_shannon_entropy_two_modality_mixture():
    # Two equal modalities: H = ln(2) approx 0.6931, J = ln(2)/ln(5) approx 0.4307
    p = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    h, j = compute_shannon_entropy_and_balance(p)
    expected_h = float(np.log(2))
    expected_j = float(np.log(2) / np.log(5))
    assert np.isclose(h, expected_h, atol=1e-3)
    assert np.isclose(j, expected_j, atol=1e-3)


def test_shannon_entropy_zero_vector():
    # Zero vector: H = 0.0, J = 0.0
    p = np.zeros(5)
    h, j = compute_shannon_entropy_and_balance(p)
    assert h == 0.0
    assert j == 0.0


def test_normalize_to_simplex_invariants():
    # Arbitrary non-negative vector
    raw = np.array([0.3, 0.1, 0.0, 0.4, 0.2])
    simplex = normalize_to_simplex(raw)
    assert np.isclose(simplex.sum(), 1.0, atol=1e-7)
    assert (simplex >= 0.0).all()

    # Zero vector returns all zeros
    zero_simplex = normalize_to_simplex(np.zeros(5))
    assert np.all(zero_simplex == 0.0)


def test_aggregate_recipe_taste_vector_unweighted():
    # Two ingredients: [0.2, 0.0, 0.4, 0.1, 0.5] and [0.4, 0.2, 0.0, 0.3, 0.1]
    matrix = np.array([
        [0.2, 0.0, 0.4, 0.1, 0.5],
        [0.4, 0.2, 0.0, 0.3, 0.1],
    ])
    mean_vec = aggregate_recipe_taste_vector(matrix, weights=None)
    expected = np.array([0.3, 0.1, 0.2, 0.2, 0.3])
    assert np.allclose(mean_vec, expected, atol=1e-6)


def test_aggregate_recipe_taste_vector_quantity_weighted():
    # Weighting: 3 parts ing1, 1 part ing2 (total 4)
    matrix = np.array([
        [0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0, 0.0],
    ])
    weights = np.array([3.0, 1.0])
    weighted_vec = aggregate_recipe_taste_vector(matrix, weights=weights)
    # Expected: [3/4 * 0.4, 1/4 * 0.8, 0, 0, 0] = [0.3, 0.2, 0, 0, 0]
    expected = np.array([0.3, 0.2, 0.0, 0.0, 0.0])
    assert np.allclose(weighted_vec, expected, atol=1e-6)


def test_recipe_taste_features_csv_invariants():
    csv_path = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    if not csv_path.exists():
        pytest.skip("recipe_taste_features.csv not yet generated")

    df = pd.read_csv(csv_path)
    assert len(df) > 0

    # Invariant: Each row's normalized taste dimensions must sum to 1.0 within float tolerance
    sums = df[[f"{dim}_norm" for dim in TASTE_DIMENSIONS]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), "Simplex normalization failed for some recipes"

    # Invariant: Entropy within [0, ln(5)], balance within [0, 1]
    assert (df["taste_entropy"] >= 0.0).all()
    assert (df["taste_entropy"] <= np.log(5) + 1e-4).all()
    assert (df["taste_balance"] >= 0.0).all()
    assert (df["taste_balance"] <= 1.0 + 1e-4).all()

    # Invariant: dominant taste is one of the 5 modalities or unspecified
    valid_dominants = set(TASTE_DIMENSIONS) | {"unspecified"}
    assert set(df["dominant_taste"].unique()).issubset(valid_dominants)
