"""Unit tests for Deliverable 3 statistical hypothesis testing and multiple comparisons."""
import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR, TASTE_DIMENSIONS
from src.cuisine_analysis import compute_clr_transform, holm_bonferroni_correction


def test_holm_bonferroni_analytical_example():
    # 3 hypotheses with raw p-values: [0.01, 0.04, 0.03]
    # Sorted order:
    # rank 0: p=0.01, multiplier=3 => adj = 0.03
    # rank 1: p=0.03, multiplier=2 => adj = max(0.03, 0.06) = 0.06
    # rank 2: p=0.04, multiplier=1 => adj = max(0.06, 0.04) = 0.06
    # Restored to original indices: [0.03, 0.06, 0.06]
    raw_p = [0.01, 0.04, 0.03]
    adj_p = holm_bonferroni_correction(raw_p)
    expected = [0.03, 0.06, 0.06]
    assert np.allclose(adj_p, expected, atol=1e-5)


def test_holm_bonferroni_monotonicity_and_bounds():
    raw_p = [0.001, 0.02, 0.015, 0.05, 0.08, 0.50]
    adj_p = holm_bonferroni_correction(raw_p)

    # Must be bounded in [0.0, 1.0]
    for p in adj_p:
        assert 0.0 <= p <= 1.0

    # Adjusted p-values must be >= raw p-values
    for r, a in zip(raw_p, adj_p):
        assert a >= r - 1e-9

    # Monotonicity check on sorted order
    sorted_pairs = sorted(zip(raw_p, adj_p), key=lambda x: x[0])
    sorted_adjs = [x[1] for x in sorted_pairs]
    for i in range(len(sorted_adjs) - 1):
        assert sorted_adjs[i] <= sorted_adjs[i + 1] + 1e-9


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correction([]) == []


def test_clr_transform_sums_to_zero():
    # Invariant: Centered log-ratio coordinates sum to 0 across parts for any simplex vector
    simplex = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.4, 0.1, 0.1, 0.3, 0.1],
        [0.6, 0.0, 0.1, 0.1, 0.2],
    ])
    clr = compute_clr_transform(simplex)
    assert clr.shape == (3, 5)
    row_sums = np.sum(clr, axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-5), "CLR coordinates must sum to zero"


def test_cuisine_comparison_csv_invariants():
    csv_path = RESULTS_DIR / "cuisine_comparison.csv"
    if not csv_path.exists():
        pytest.skip("cuisine_comparison.csv not yet generated")

    df = pd.read_csv(csv_path)
    assert len(df) > 0

    # Invariant: Each cuisine must have positive sample size
    assert (df["n_recipes"] > 0).all()

    # Invariant: Most distinctive taste must be one of the 5 modalities
    assert set(df["most_distinctive_taste"].unique()).issubset(set(TASTE_DIMENSIONS))

    # Invariant: Distinctiveness z-score must be non-negative
    assert (df["max_distinctiveness_z"] >= 0.0).all()
