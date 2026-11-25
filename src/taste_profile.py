"""
Deliverable 2: Computational Recipe Taste Profile.
Aggregates ingredient-level taste proxy attributes into recipe-level vectors
with explicit unknown (NaN) handling, projects onto the 5D probability simplex,
computes Shannon entropy (nats) and Pielou's distributional evenness (J),
and generates distribution visualizations and worked methodology reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    RunContext,
    TASTE_DIMENSIONS,
    default_context,
    display_path,
)


def compute_shannon_entropy_and_evenness(prob_vector: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Shannon entropy H (in nats) and Pielou's distributional evenness J in [0, 1].
    - H(p) = -sum_{d: p_d > 0} p_d * ln(p_d)
    - J(p) = H(p) / ln(K), where K = len(TASTE_DIMENSIONS) = 5
    - Zero vector or single non-zero modality returns (0.0, 0.0).
    - Uniform distribution [1/K, ..., 1/K] returns (ln(K), 1.0).
    """
    p = np.asarray(prob_vector, dtype=float)
    if p.ndim != 1 or len(p) != len(TASTE_DIMENSIONS):
        raise ValueError(f"Expected 1D vector of length {len(TASTE_DIMENSIONS)}, got shape {p.shape}.")
    if np.isinf(p).any() or np.any(p[~np.isnan(p)] < -1e-12):
        raise ValueError("Probability vector must be finite and non-negative.")

    p_clean = np.where(np.isnan(p), 0.0, np.maximum(p, 0.0))
    total = float(np.sum(p_clean))
    if total <= 1e-12:
        return 0.0, 0.0
    if not np.isclose(total, 1.0, atol=1e-6):
        p_clean = p_clean / total

    p_pos = p_clean[p_clean > 1e-12]
    if len(p_pos) <= 1:
        return 0.0, 0.0

    entropy = float(-np.sum(p_pos * np.log(p_pos)))
    max_entropy = float(np.log(len(TASTE_DIMENSIONS)))
    evenness = entropy / max_entropy if max_entropy > 0 else 0.0
    evenness = min(1.0, max(0.0, evenness))
    return round(entropy, 6), round(evenness, 6)


# Backwards-compatible alias
compute_shannon_entropy_and_balance = compute_shannon_entropy_and_evenness


def aggregate_recipe_taste_vector(
    matched_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate matched ingredient vectors into an unweighted or quantity-weighted mean vector,
    ignoring unknown (NaN) entries per taste dimension.
    - Validates non-NaN entries lie in [0.0, 1.0] and weights (if provided) are non-negative.
    - If all ingredients are NaN on a dimension, returns NaN for that dimension.
    - When quantities are unavailable (weights=None), never invents weights.
    """
    mat = np.asarray(matched_matrix, dtype=float)
    if mat.size == 0 or len(mat) == 0:
        return np.full(len(TASTE_DIMENSIONS), np.nan, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != len(TASTE_DIMENSIONS):
        raise ValueError(f"Expected 2D matrix with {len(TASTE_DIMENSIONS)} columns, got shape {mat.shape}.")
    if np.isinf(mat).any():
        raise ValueError("Ingredient taste matrix cannot contain infinite values.")
    valid_vals = mat[~np.isnan(mat)]
    if np.any((valid_vals < 0.0) | (valid_vals > 1.0)):
        raise ValueError("Ingredient taste intensities must lie in [0.0, 1.0] or be NaN.")

    if weights is None:
        out = np.full(len(TASTE_DIMENSIONS), np.nan, dtype=float)
        for col in range(mat.shape[1]):
            col_vals = mat[:, col]
            known = col_vals[~np.isnan(col_vals)]
            if len(known) > 0:
                out[col] = float(np.mean(known))
        return out

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(mat):
        raise ValueError(f"Weights must be 1D of length {len(mat)}, got shape {w.shape}.")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("Ingredient quantity weights must be finite and non-negative.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("Sum of quantity weights must be strictly positive when weights are provided.")

    out = np.full(len(TASTE_DIMENSIONS), np.nan, dtype=float)
    for col in range(mat.shape[1]):
        col_vals = mat[:, col]
        mask = (~np.isnan(col_vals)) & (w > 0.0)
        if np.any(mask):
            w_sub = w[mask]
            out[col] = float(np.sum(col_vals[mask] * w_sub) / np.sum(w_sub))
    return out


def normalize_to_simplex(raw_vector: np.ndarray) -> np.ndarray:
    """
    L1-normalize a non-negative vector (treating NaN dimensions as unobserved 0.0 share)
    onto the probability simplex sum(p) == 1. Returns all zeros if total <= 1e-9.
    """
    v = np.asarray(raw_vector, dtype=float)
    if v.ndim != 1 or len(v) != len(TASTE_DIMENSIONS):
        raise ValueError(f"Expected 1D vector of length {len(TASTE_DIMENSIONS)}, got shape {v.shape}.")
    if np.isinf(v).any():
        raise ValueError("Raw taste vector cannot contain infinite values.")
    valid = v[~np.isnan(v)]
    if np.any(valid < 0.0):
        raise ValueError("Raw taste vector cannot contain negative values.")

    v_zeroed = np.where(np.isnan(v), 0.0, v)
    s = float(np.sum(v_zeroed))
    if s > 1e-9:
        return v_zeroed / s
    return np.zeros_like(v_zeroed)


def _load_manifest_context(context: RunContext) -> Dict[str, object]:
    if context.manifest_path.exists():
        with open(context.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"is_synthetic": False, "dataset_name": "Unknown Recipe Corpus"}


def run_deliverable_2(ctx: Optional[RunContext] = None) -> pd.DataFrame:
    """
    Execute Deliverable 2:
    1. Reads clean recipes and taste-known ingredient vectors.
    2. Computes unweighted raw mean intensities (ignoring NaN per dimension) and L1 simplex shares.
    3. Calculates dominant taste, Shannon entropy (nats), and Pielou's evenness (J).
    4. Saves recipe_taste_features.csv.
    5. Generates distribution and representative profile visualizations.
    6. Produces taste_profile_methodology.md with worked example and sensory disclaimers.
    """
    context = ctx or default_context()
    context.ensure()
    manifest = _load_manifest_context(context)
    is_synthetic = bool(manifest.get("is_synthetic", False))
    corpus_tag = "SYNTHETIC SMOKE TEST" if is_synthetic else "REAL CORPUS"

    recipes_df = pd.read_csv(context.processed_dir / "recipes_clean.csv")
    rec_ings_df = pd.read_csv(context.processed_dir / "recipe_ingredients.csv")
    mapping_df = pd.read_csv(context.processed_dir / "ingredient_flavor_mapping.csv")

    matched_lex = mapping_df[mapping_df["match_status"] == "matched"].set_index("ingredient_norm")

    merged = rec_ings_df[rec_ings_df["is_matched"]].merge(
        matched_lex[TASTE_DIMENSIONS],
        left_on="ingredient_norm",
        right_index=True,
        how="inner",
    )

    # Per-recipe dimension-wise mean ignoring NaN (unknown values are not coerced to 0)
    grouped_means = (
        merged.groupby("recipe_id")[TASTE_DIMENSIONS]
        .mean()
        .reset_index()
    )

    valid_recipes = recipes_df[recipes_df["sufficient_for_analysis"]].copy()
    profile_df = valid_recipes.merge(grouped_means, on="recipe_id", how="inner")

    raw_matrix = profile_df[TASTE_DIMENSIONS].to_numpy(dtype=float)
    n_known_dims = np.sum(~np.isnan(raw_matrix), axis=1).astype(int)
    norm_matrix = np.vstack([normalize_to_simplex(row) for row in raw_matrix])

    for idx, dim in enumerate(TASTE_DIMENSIONS):
        raw_col = raw_matrix[:, idx]
        profile_df[f"{dim}_raw"] = np.where(np.isnan(raw_col), np.nan, np.round(raw_col, 6))
        profile_df[f"{dim}_norm"] = np.round(norm_matrix[:, idx], 6)
        profile_df[dim] = np.round(norm_matrix[:, idx], 6)

    profile_df["n_known_taste_dims"] = n_known_dims

    dominant_labels = []
    for row_norm in norm_matrix:
        if float(np.sum(row_norm)) <= 1e-9:
            dominant_labels.append("unspecified")
        else:
            dominant_labels.append(TASTE_DIMENSIONS[int(np.argmax(row_norm))])
    profile_df["dominant_taste"] = dominant_labels

    entropies = []
    evennesses = []
    for row_p in norm_matrix:
        h_val, j_val = compute_shannon_entropy_and_evenness(row_p)
        entropies.append(h_val)
        evennesses.append(j_val)

    profile_df["taste_entropy"] = entropies
    profile_df["taste_evenness"] = evennesses

    output_csv = context.processed_dir / "recipe_taste_features.csv"
    profile_df.to_csv(output_csv, index=False)

    # Visualization 1: Violin / quartile plot of normalized taste shares
    simplex_df = profile_df[profile_df["dominant_taste"] != "unspecified"]
    plot_source = simplex_df if not simplex_df.empty else profile_df

    plt.figure(figsize=(9.5, 5.8))
    melted = plot_source.melt(
        id_vars=["recipe_id", "cuisine"],
        value_vars=TASTE_DIMENSIONS,
        var_name="Taste Dimension",
        value_name="Normalized Taste Share",
    )
    sns.violinplot(
        data=melted,
        x="Taste Dimension",
        y="Normalized Taste Share",
        hue="Taste Dimension",
        inner="quartile",
        palette="Set2",
        cut=0,
        legend=False,
    )
    plt.title(
        f"Modeled Recipe Taste Shares (L1 Simplex, N={len(plot_source):,})\n"
        f"[{corpus_tag} — Unweighted Ingredient-Presence Proxy]",
        fontsize=11,
    )
    plt.xlabel("Basic Gustatory Modality")
    plt.ylabel("Normalized Taste Share (Sum = 1.0)")
    plt.ylim(-0.02, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(context.results_dir / "taste_profile_distributions.png", dpi=200)
    plt.close()

    # Visualization 2: Representative stacked bar profiles (pick well-covered recipe per cuisine)
    top_cuisines = plot_source["cuisine"].value_counts().head(5).index.tolist()
    rep_pool = plot_source[plot_source["cuisine"].isin(top_cuisines)].sort_values(
        ["n_known_taste_dims", "n_ingredients_matched", "recipe_id"],
        ascending=[False, False, True],
    )
    sample_examples = (
        rep_pool.groupby("cuisine")
        .first()
        .loc[top_cuisines]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottom = np.zeros(len(sample_examples))
    colors = ["#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#e76f51"]

    for dim, color in zip(TASTE_DIMENSIONS, colors):
        values = sample_examples[dim].to_numpy()
        ax.bar(
            sample_examples["cuisine"] + "\n(#" + sample_examples["recipe_id"].astype(str) + ")",
            values,
            bottom=bottom,
            label=dim.capitalize(),
            color=color,
            edgecolor="white",
        )
        bottom += values

    ax.set_ylabel("Normalized Taste Share")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Representative Recipe Taste-Share Profiles Across Top Cuisines [{corpus_tag}]\n"
        "(Unweighted Ingredient-Presence Proxy — Not Gram-Weighted Sensory Intensities)",
        fontsize=10.5,
    )
    ax.legend(title="Basic Modality", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(context.results_dir / "example_recipe_profiles.png", dpi=200)
    plt.close()

    # Worked example from actual corpus
    ex_pool = plot_source if not plot_source.empty else profile_df
    ex = ex_pool.iloc[0]
    ex_ings = rec_ings_df[
        (rec_ings_df["recipe_id"] == ex["recipe_id"]) & (rec_ings_df["is_matched"])
    ].merge(matched_lex[TASTE_DIMENSIONS], left_on="ingredient_norm", right_index=True)

    ex_table = ex_ings[["ingredient_norm"] + TASTE_DIMENSIONS].fillna("unknown (NaN)").to_markdown(index=False)
    def _fmt_raw(val: object) -> str:
        return "unknown" if pd.isna(val) else f"{float(val):.4f}"

    corpus_banner = (
        "> **Corpus Notice:** Generated from **SYNTHETIC SMOKE TEST (`--use-sample-data`)** — "
        "illustrative pipeline verification only."
        if is_synthetic
        else "> **Corpus Notice:** Generated from **REAL EMPIRICAL RECIPE CORPUS** (`"
        + str(manifest.get("dataset_name", "External Corpus"))
        + "`)."
    )

    methodology_md = f"""# Deliverable 2: Recipe Taste Profile Methodology

{corpus_banner}

## 1. Mathematical Formulation
For a recipe $r$ containing taste-annotated matched ingredient set $M_r$ ($|M_r| \\ge 2$), each ingredient $i \\in M_r$ maps to a 5-dimensional basic-taste proxy vector $\\mathbf{{t}}(i) \\in ([0, 1] \\cup \\{{\\text{{NaN}}\\}})^5$ over $\\mathcal{{D}} = \\{{\\text{{sweet}}, \\text{{sour}}, \\text{{bitter}}, \\text{{salty}}, \\text{{umami}}\\}}$.

1. **Unweighted Mean Intensity Vector (with Missing-Score Exclusion):**
   Let $M_{{r,d}} = \\{{i \\in M_r : t_d(i) \\text{{ is known}}\\}}$. Then:
   $$u_d(r) = \\frac{{1}}{{|M_{{r,d}}|}} \\sum_{{i \\in M_{{r,d}}}} t_d(i), \\quad \\text{{or }} \\text{{NaN}} \\text{{ if }} |M_{{r,d}}| = 0.$$
   *(When reliable ingredient mass weights $w_i$ are present, `aggregate_recipe_taste_vector` computes $u_d(r) = \\sum_{{i \\in M_{{r,d}}}} w_i t_d(i) / \\sum_{{i \\in M_{{r,d}}}} w_i$. Because standard recipe corpora lack gram quantities, no heuristic weights are invented; unweighted presence is used.)*

2. **$L_1$-Normalized Taste Share Profile (4-Simplex):**
   Treating unobserved ($\\text{{NaN}}$) dimensions as $0$ contribution to the total modeled taste sum $S(r) = \\sum_{{d : u_d(r) \\text{{ known}}}} u_d(r)$:
   $$p_d(r) = \\frac{{u_d(r)}}{{S(r)}} \\text{{ if }} S(r) > 0, \\quad \\text{{else }} 0, \\quad \\text{{so }} \\sum_{{d \\in \\mathcal{{D}}}} p_d(r) = 1 \\text{{ (when }} S(r) > 0\\text{{)}}.$$

3. **Modeled Taste-Share Evenness (Normalized Shannon Entropy):**
   $$H(r) = -\\sum_{{d \\in \\mathcal{{D}}: p_d(r) > 0}} p_d(r) \\ln p_d(r) \\quad \\text{{[nats]}}$$
   $$J(r) = \\frac{{H(r)}}{{\\ln(5)}} \\in [0, 1]$$
   - $J(r) = 1.0$: perfectly uniform distribution across all 5 modeled taste shares.
   - $J(r) = 0.0$: single-modality concentration (or zero-vector).

## 2. Corpus Summary Statistics
- **Sufficient Recipes Profiled:** {len(profile_df):,} (Unspecified zero-vector profiles: {int((profile_df['dominant_taste'] == 'unspecified').sum()):,})
- **Mean Shannon Entropy ($H$):** {profile_df['taste_entropy'].mean():.4f} nats (SD = {profile_df['taste_entropy'].std():.4f}; theoretical max = $\\ln 5 \\approx 1.6094$)
- **Mean Taste-Share Evenness ($J$):** {profile_df['taste_evenness'].mean():.4f} (SD = {profile_df['taste_evenness'].std():.4f})

## 3. Worked Example from Corpus: Recipe ID `{int(ex['recipe_id'])}` (`{ex['cuisine']}`)
### Matched Ingredient Vectors
{ex_table}

### Aggregated Recipe Profile
- **Raw Mean Vector ($\\mathbf{{u}}$):** sweet={_fmt_raw(ex['sweet_raw'])}, sour={_fmt_raw(ex['sour_raw'])}, bitter={_fmt_raw(ex['bitter_raw'])}, salty={_fmt_raw(ex['salty_raw'])}, umami={_fmt_raw(ex['umami_raw'])}
- **Normalized Taste Shares ($\\mathbf{{p}}$):** sweet={ex['sweet_norm']:.4f}, sour={ex['sour_norm']:.4f}, bitter={ex['bitter_norm']:.4f}, salty={ex['salty_norm']:.4f}, umami={ex['umami_norm']:.4f}
- **Dominant Taste Modality:** `{ex['dominant_taste']}`
- **Shannon Entropy ($H$):** `{ex['taste_entropy']:.4f}` nats
- **Taste-Share Evenness ($J$):** `{ex['taste_evenness']:.4f}`

## 4. Methodological & Sensory Limitations
1. **Presence vs. Quantity/Concentration:** Because public recipe lists lack mass weights, unweighted averaging assigns equal influence to a tablespoon of salt or spice and a cup of grain or broth. The output represents an *ingredient-assembly taste proxy*, not physical concentration in a cooked serving.
2. **Distributional Evenness vs. Human Perceived Balance:** Pielou's evenness $J(r)$ measures mathematical uniformity of the 5 modeled taste shares; it does **not** directly measure human gastronomic harmony or culinary palatability (a dessert does not require equal bitterness, saltiness, and sourness to be well-balanced).
3. **Cooking Transformations Omitted:** Caramelization, Maillard reactions, fermentation, dilution, and taste-taste suppression (e.g., fat or sugar masking bitterness) are not captured by static ingredient lookups.
4. **Proxy Instrument:** Outputs must be interpreted as computational proxies rather than empirical sensory panel scores.
"""
    (context.reports_dir / "taste_profile_methodology.md").write_text(methodology_md, encoding="utf-8")
    return profile_df
