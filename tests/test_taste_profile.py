"""Unit tests for Deliverable 2 taste profile mathematics, NaN handling, and invariants."""
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_PROCESSED_DIR, TASTE_DIMENSIONS
from src.taste_profile import (
    aggregate_recipe_taste_vector,
    compute_shannon_entropy_and_evenness,
    normalize_to_simplex,
)


def test_shannon_entropy_analytical_cases():
    # Uniform: H = ln(5), J = 1.0
    h_u, j_u = compute_shannon_entropy_and_evenness(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert np.isclose(h_u, np.log(5), atol=1e-5)
    assert np.isclose(j_u, 1.0, atol=1e-5)

    # Delta: H = 0.0, J = 0.0
    h_d, j_d = compute_shannon_entropy_and_evenness(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert h_d == 0.0 and j_d == 0.0

    # Two-modality equal mixture: H = ln(2), J = ln(2)/ln(5)
    h_2, j_2 = compute_shannon_entropy_and_evenness(np.array([0.5, 0.5, 0.0, 0.0, 0.0]))
    assert np.isclose(h_2, np.log(2), atol=1e-5)
    assert np.isclose(j_2, np.log(2) / np.log(5), atol=1e-5)

    # Zero vector: H = 0.0, J = 0.0
    h_z, j_z = compute_shannon_entropy_and_evenness(np.zeros(5))
    assert h_z == 0.0 and j_z == 0.0

    with pytest.raises(ValueError):
        compute_shannon_entropy_and_evenness(np.array([-0.1, 0.5, 0.6, 0.0, 0.0]))


def test_normalize_to_simplex_with_nan_and_negative_checks():
    raw = np.array([0.3, 0.1, np.nan, 0.4, 0.2])
    simplex = normalize_to_simplex(raw)
    assert np.isclose(simplex.sum(), 1.0, atol=1e-7)
    assert simplex[2] == 0.0
    assert (simplex >= 0.0).all()

    assert np.all(normalize_to_simplex(np.zeros(5)) == 0.0)
    assert np.all(normalize_to_simplex(np.full(5, np.nan)) == 0.0)

    with pytest.raises(ValueError):
        normalize_to_simplex(np.array([0.5, -0.2, 0.1, 0.0, 0.0]))


def test_aggregate_recipe_taste_vector_unweighted_weighted_and_nan():
    # Unweighted with NaN ignored per dimension
    matrix = np.array([
        [0.2, np.nan, 0.4, 0.1, 0.5],
        [0.4, np.nan, 0.0, 0.3, 0.1],
    ])
    mean_vec = aggregate_recipe_taste_vector(matrix, weights=None)
    assert np.isclose(mean_vec[0], 0.3)
    assert np.isnan(mean_vec[1])
    assert np.allclose(mean_vec[[0, 2, 3, 4]], [0.3, 0.2, 0.2, 0.3])

    # Quantity-weighted with NaN on one ingredient dimension
    matrix_w = np.array([
        [0.4, np.nan, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0, 0.0],
    ])
    weights = np.array([3.0, 1.0])
    weighted = aggregate_recipe_taste_vector(matrix_w, weights=weights)
    assert np.isclose(weighted[0], 0.3)
    assert np.isclose(weighted[1], 0.8)  # weighted over known ingredients only!

    # Validation checks
    with pytest.raises(ValueError):
        aggregate_recipe_taste_vector(np.array([[1.2, 0.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        aggregate_recipe_taste_vector(matrix, weights=np.array([-1.0, 2.0]))
    with pytest.raises(ValueError):
        aggregate_recipe_taste_vector(matrix, weights=np.array([0.0, 0.0]))


def test_recipe_taste_features_csv_invariants():
    csv_path = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    if not csv_path.exists():
        pytest.skip("recipe_taste_features.csv not yet generated")

    df = pd.read_csv(csv_path)
    assert len(df) > 0
    assert "taste_evenness" in df.columns

    non_zero = df[df["dominant_taste"] != "unspecified"]
    sums = non_zero[[f"{dim}_norm" for dim in TASTE_DIMENSIONS]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)

    assert (df["taste_entropy"] >= 0.0).all()
    assert (df["taste_entropy"] <= np.log(5) + 1e-4).all()
    assert (df["taste_evenness"] >= 0.0).all()
    assert (df["taste_evenness"] <= 1.0 + 1e-4).all()
    assert set(df["dominant_taste"].unique()).issubset(set(TASTE_DIMENSIONS) | {"unspecified"})
