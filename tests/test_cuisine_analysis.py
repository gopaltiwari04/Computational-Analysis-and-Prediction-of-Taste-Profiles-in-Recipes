"""Unit tests for Deliverable 3 statistical correction, CLR, PERMANOVA, and invariants."""
import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR, TASTE_DIMENSIONS
from src.cuisine_analysis import (
    compute_clr_transform,
    holm_bonferroni_correction,
    run_compositional_permanova,
)


def test_holm_bonferroni_analytical_and_edge_cases():
    raw_p = [0.01, 0.04, 0.03]
    adj_p = holm_bonferroni_correction(raw_p)
    assert np.allclose(adj_p, [0.03, 0.06, 0.06], atol=1e-6)
    assert holm_bonferroni_correction([]) == []

    with pytest.raises(ValueError):
        holm_bonferroni_correction([-0.01, 0.2])
    with pytest.raises(ValueError):
        holm_bonferroni_correction([0.1, 1.2])


def test_holm_bonferroni_monotonicity_and_bounds():
    raw_p = [0.001, 0.02, 0.015, 0.05, 0.08, 0.50]
    adj_p = holm_bonferroni_correction(raw_p)
    for r, a in zip(raw_p, adj_p):
        assert 0.0 <= a <= 1.0
        assert a >= r - 1e-12

    sorted_adjs = [x[1] for x in sorted(zip(raw_p, adj_p), key=lambda x: x[0])]
    for i in range(len(sorted_adjs) - 1):
        assert sorted_adjs[i] <= sorted_adjs[i + 1] + 1e-12


def test_clr_transform_multiplicative_zeros_and_validation():
    simplex = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.4, 0.1, 0.1, 0.3, 0.1],
        [0.6, 0.0, 0.1, 0.1, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    clr = compute_clr_transform(simplex)
    assert clr.shape == (4, 5)
    assert np.allclose(np.sum(clr, axis=1), 0.0, atol=1e-6)
    assert np.allclose(clr[0], 0.0, atol=1e-6)
    assert np.allclose(clr[3], 0.0, atol=1e-6)

    with pytest.raises(ValueError):
        compute_clr_transform(np.array([[0.5, -0.1, 0.2, 0.2, 0.2]]))


def test_compositional_permanova_separation():
    # Two sharply separated compositions on the simplex
    g1 = np.tile([0.7, 0.05, 0.05, 0.1, 0.1], (15, 1))
    g2 = np.tile([0.05, 0.05, 0.1, 0.1, 0.7], (15, 1))
    clr = compute_clr_transform(np.vstack([g1, g2]))
    labels = np.array(["A"] * 15 + ["B"] * 15)
    res = run_compositional_permanova(clr, labels, n_permutations=99)
    assert res["r_squared"] > 0.99
    assert res["p_value"] <= 0.02


def test_cuisine_comparison_csv_invariants():
    csv_path = RESULTS_DIR / "cuisine_comparison.csv"
    if not csv_path.exists():
        pytest.skip("cuisine_comparison.csv not yet generated")

    df = pd.read_csv(csv_path)
    assert len(df) > 0
    assert (df["n_recipes"] > 0).all()
    assert set(df["most_distinctive_taste"].unique()).issubset(set(TASTE_DIMENSIONS))
    assert (df["max_distinctiveness_z"] >= -1e-6).all()
