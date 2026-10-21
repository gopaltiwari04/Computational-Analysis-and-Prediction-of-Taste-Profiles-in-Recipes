"""
Deliverable 2: Computational Recipe Taste Profile.
Aggregates ingredient-level taste proxy attributes into recipe-level vectors,
computes Shannon entropy and Pielou's evenness (taste distribution evenness),
and produces distribution visualizations and worked methodology reports.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    TASTE_DIMENSIONS,
    ensure_directories,
)


def compute_shannon_entropy_and_balance(prob_vector: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Shannon entropy H (in nats) and Pielou's evenness J in [0, 1].
    - H(p) = -sum_{d: p_d > 0} p_d * ln(p_d)
    - J(p) = H(p) / ln(K), where K = len(TASTE_DIMENSIONS) = 5
    - If prob_vector is a zero vector or has <= 1 non-zero element, returns (0.0, 0.0).
    - If prob_vector is uniform [1/K, ..., 1/K], returns (ln(K), 1.0).
    """
    p = np.asarray(prob_vector, dtype=float)
    p_pos = p[p > 1e-12]
    if len(p_pos) <= 1:
        return 0.0, 0.0

    entropy = float(-np.sum(p_pos * np.log(p_pos)))
    max_entropy = float(np.log(len(TASTE_DIMENSIONS)))
    evenness = entropy / max_entropy if max_entropy > 0 else 0.0
    # Guard against float rounding exceeding 1.0
    evenness = min(1.0, max(0.0, evenness))
    return round(entropy, 4), round(evenness, 4)


def aggregate_recipe_taste_vector(
    matched_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate matched ingredient vectors into an unweighted or quantity-weighted mean vector.
    - When weights is None: u_d = (1 / |M_r|) * sum_{i in M_r} t_d(i)
    - When weights is provided: u_d = sum(w_i * t_d(i)) / sum(w_i)
    """
    if matched_matrix.size == 0 or len(matched_matrix) == 0:
        return np.zeros(len(TASTE_DIMENSIONS), dtype=float)

    if weights is None:
        return np.nanmean(matched_matrix, axis=0)

    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return np.nanmean(matched_matrix, axis=0)
    w_norm = w / w.sum()
    return np.sum(matched_matrix * w_norm[:, np.newaxis], axis=0)


def normalize_to_simplex(raw_vector: np.ndarray) -> np.ndarray:
    """
    L1-normalize a non-negative vector to the probability simplex sum(p) == 1.
    If the vector sums to 0, returns a zero vector.
    """
    v = np.asarray(raw_vector, dtype=float)
    s = float(np.sum(v))
    if s > 1e-9:
        return v / s
    return np.zeros_like(v)


def run_deliverable_2() -> pd.DataFrame:
    """
    Execute Deliverable 2:
    1. Reads clean recipes and matched ingredient vectors.
    2. Computes unweighted raw mean intensities and L1-normalized simplex shares.
    3. Calculates dominant taste, Shannon entropy (nats), and Pielou's evenness.
    4. Saves recipe_taste_features.csv.
    5. Generates distribution and representative profile visualizations.
    6. Produces taste_profile_methodology.md with worked examples and sensory disclaimers.
    """
    ensure_directories()

    recipes_df = pd.read_csv(DATA_PROCESSED_DIR / "recipes_clean.csv")
    rec_ings_df = pd.read_csv(DATA_PROCESSED_DIR / "recipe_ingredients.csv")
    mapping_df = pd.read_csv(DATA_PROCESSED_DIR / "ingredient_flavor_mapping.csv")

    matched_lex = mapping_df[mapping_df["match_status"] == "matched"].set_index("ingredient_norm")

    # Filter to matched ingredients and join taste dimensions
    merged = rec_ings_df[rec_ings_df["is_matched"]].merge(
        matched_lex[TASTE_DIMENSIONS],
        left_on="ingredient_norm",
        right_index=True,
        how="inner",
    )

    # Compute raw mean taste intensities per recipe
    grouped_means = merged.groupby("recipe_id")[TASTE_DIMENSIONS].mean().reset_index()

    # Retain only sufficient recipes
    valid_recipes = recipes_df[recipes_df["sufficient_for_analysis"]].copy()
    profile_df = valid_recipes.merge(grouped_means, on="recipe_id", how="inner")

    raw_matrix = profile_df[TASTE_DIMENSIONS].to_numpy(dtype=float)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        raw_matrix,
        np.where(row_sums > 1e-9, row_sums, 1.0),
    )

    # Populate raw means and normalized shares
    for idx, dim in enumerate(TASTE_DIMENSIONS):
        profile_df[f"{dim}_raw"] = raw_matrix[:, idx].round(4)
        profile_df[f"{dim}_norm"] = norm_matrix[:, idx].round(4)
        profile_df[dim] = norm_matrix[:, idx].round(4)

    # Dominant taste calculation (handling zero-vector case safely)
    dominant_labels = []
    for row_r in raw_matrix:
        if row_r.sum() < 1e-9:
            dominant_labels.append("unspecified")
        else:
            dominant_labels.append(TASTE_DIMENSIONS[int(np.argmax(row_r))])
    profile_df["dominant_taste"] = dominant_labels

    # Shannon entropy and Pielou's evenness
    entropies = []
    balances = []
    for row_p in norm_matrix:
        h_val, j_val = compute_shannon_entropy_and_balance(row_p)
        entropies.append(h_val)
        balances.append(j_val)

    profile_df["taste_entropy"] = entropies
    profile_df["taste_balance"] = balances

    # Save processed recipe taste features
    output_csv = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    profile_df.to_csv(output_csv, index=False)

    # Visualization 1: Violin / box plot of normalized taste shares across corpus
    plt.figure(figsize=(9, 5.5))
    melted = profile_df.melt(
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
        legend=False,
    )
    plt.title("Corpus Distribution of Modeled Recipe Taste Shares (L1 Simplex)")
    plt.xlabel("Gustatory Modality")
    plt.ylabel("Normalized Taste Share")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "taste_profile_distributions.png", dpi=200)
    plt.close()

    # Visualization 2: Representative stacked bar profiles (1 recipe from top 5 cuisines)
    top_cuisines = profile_df["cuisine"].value_counts().head(5).index.tolist()
    sample_examples = (
        profile_df[profile_df["cuisine"].isin(top_cuisines)]
        .groupby("cuisine")
        .first()
        .loc[top_cuisines]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
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
    ax.set_title("Representative Recipe Taste Profiles Across Distinct Cuisines")
    ax.legend(title="Taste Modality", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "example_recipe_profiles.png", dpi=200)
    plt.close()

    # Select representative recipe for worked example
    ex = profile_df.iloc[0]
    ex_ings = rec_ings_df[
        (rec_ings_df["recipe_id"] == ex["recipe_id"]) & (rec_ings_df["is_matched"])
    ].merge(matched_lex[TASTE_DIMENSIONS], left_on="ingredient_norm", right_index=True)

    methodology_md = f"""# Deliverable 2: Recipe Taste Profile Methodology

## 1. Mathematical Formulation
For a recipe $r$ containing matched ingredient set $M_r$ ($|M_r| \\ge 2$), each ingredient $i \\in M_r$ maps to a 5-dimensional gustatory taste proxy vector $\\mathbf{{t}}(i) \\in [0, 1]^5$ over the basis $\\mathcal{{D}} = \\{{\\text{{sweet}}, \\text{{sour}}, \\text{{bitter}}, \\text{{salty}}, \\text{{umami}}\\}}$.

1. **Unweighted Mean Intensity Vector:**
   $$\\mathbf{{u}}_d(r) = \\frac{{1}}{{|M_r|}} \\sum_{{i \\in M_r}} \\mathbf{{t}}_d(i)$$
   *(Note: If reliable mass/volume quantities $w_i$ are available, the formulation supports quantity-weighted aggregation $\\mathbf{{u}}_d(r) = \\sum w_i \\mathbf{{t}}_d(i) / \\sum w_i$. In standard recipe corpora lacking gram-level weights, unweighted presence is the only defensible baseline.)*

2. **$L_1$-Normalized Taste Simplex Profile:**
   $$\\mathbf{{p}}_d(r) = \\frac{{\\mathbf{{u}}_d(r)}}{{\\sum_{{d' \\in \\mathcal{{D}}}} \\mathbf{{u}}_{{d'}}(r)}}, \\quad \\text{{such that }} \\sum_{{d \\in \\mathcal{{D}}}} \\mathbf{{p}}_d(r) = 1$$

3. **Taste Distributional Evenness (Normalized Shannon Entropy):**
   $$H(r) = -\\sum_{{d \\in \\mathcal{{D}}: \\mathbf{{p}}_d(r) > 0}} \\mathbf{{p}}_d(r) \\ln \\mathbf{{p}}_d(r) \\quad \\text{{[nats]}}$$
   $$J(r) = \\frac{{H(r)}}{{\\ln(5)}} \\in [0, 1]$$
   - $J(r) = 1.0$ indicates equal shares across all five taste dimensions (maximum dispersion).
   - $J(r) = 0.0$ indicates single-modality dominance (minimum dispersion).

## 2. Worked Example from Corpus: Recipe ID `{int(ex['recipe_id'])}` (`{ex['cuisine']}`)
### Matched Ingredient Vectors
{ex_ings[['ingredient_norm'] + TASTE_DIMENSIONS].to_markdown(index=False)}

### Aggregated Recipe Profile
- **Raw Mean Vector ($\\mathbf{{u}}$):** sweet={ex['sweet_raw']:.4f}, sour={ex['sour_raw']:.4f}, bitter={ex['bitter_raw']:.4f}, salty={ex['salty_raw']:.4f}, umami={ex['umami_raw']:.4f}
- **Normalized Profile ($\\mathbf{{p}}$):** sweet={ex['sweet_norm']:.4f}, sour={ex['sour_norm']:.4f}, bitter={ex['bitter_norm']:.4f}, salty={ex['salty_norm']:.4f}, umami={ex['umami_norm']:.4f}
- **Dominant Taste Modality:** `{ex['dominant_taste']}`
- **Shannon Entropy ($H$):** `{ex['taste_entropy']:.4f}` nats
- **Distributional Evenness ($J$):** `{ex['taste_balance']:.4f}`

## 3. Critical Methodological & Sensory Disclaimers
1. **Ingredient Presence vs. Concentration:** In the absence of measured gram weights, an unweighted average attributes equal presence weight to high-volume staples (e.g., rice, milk) and low-volume spices (e.g., cayenne, turmeric). The resulting vector reflects the *taste orientation of the ingredient assembly*, not physical molecular concentration in the plated dish.
2. **Entropy vs. Human Culinary Balance:** Normalized entropy $J(r)$ quantifies the mathematical uniformity of the modeled taste shares across the 5 dimensions. It is not an empirical measure of gastronomic harmony or hedonic palatability; a culinary dish does not require equal intensities of all five basic tastes to be perceived as balanced by human diners.
3. **Chemical Transformation Omitted:** Culinary processing (caramelization, Maillard browning, acid hydrolysis, salting, fermentation) transforms tastants dynamically during cooking. These static representations serve as an ingredient-level taste proxy rather than wet-lab sensory evaluations.
"""
    (REPORTS_DIR / "taste_profile_methodology.md").write_text(methodology_md, encoding="utf-8")
    return profile_df
