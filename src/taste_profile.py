"""
Deliverable 2: Computational Recipe Taste Profile.
Aggregates ingredient-level taste attributes into recipe-level vectors,
diversity/balance metrics, and generates distribution and example profile plots.
"""
from typing import Dict, Tuple

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
    """Calculate Shannon entropy H and Pielou's evenness (balance) J in [0, 1]."""
    p = prob_vector[prob_vector > 1e-12]
    if len(p) <= 1:
        return 0.0, 0.0
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(TASTE_DIMENSIONS)))
    balance = entropy / max_entropy if max_entropy > 0 else 0.0
    return round(entropy, 4), round(balance, 4)


def run_deliverable_2() -> pd.DataFrame:
    """Compute recipe-level taste profiles, save CSV, plots, and methodology report."""
    ensure_directories()

    recipes_df = pd.read_csv(DATA_PROCESSED_DIR / "recipes_clean.csv")
    rec_ings_df = pd.read_csv(DATA_PROCESSED_DIR / "recipe_ingredients.csv")
    mapping_df = pd.read_csv(DATA_PROCESSED_DIR / "ingredient_flavor_mapping.csv")

    matched_lex = mapping_df[mapping_df["match_status"] == "matched"].set_index("ingredient_norm")

    # Join recipe ingredients with matched taste scores
    merged = rec_ings_df[rec_ings_df["is_matched"]].merge(
        matched_lex[TASTE_DIMENSIONS],
        left_on="ingredient_norm",
        right_index=True,
        how="inner",
    )

    # Aggregate raw mean taste intensities per recipe
    grouped_means = merged.groupby("recipe_id")[TASTE_DIMENSIONS].mean().reset_index()

    # Merge back with recipes_clean metadata (retaining only sufficient recipes)
    valid_recipes = recipes_df[recipes_df["sufficient_for_analysis"]].copy()
    profile_df = valid_recipes.merge(grouped_means, on="recipe_id", how="inner")

    # Compute L1-normalized shares, dominant taste, entropy, and balance
    raw_matrix = profile_df[TASTE_DIMENSIONS].to_numpy(dtype=float)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        raw_matrix,
        np.where(row_sums > 1e-9, row_sums, 1.0),
    )

    for idx, dim in enumerate(TASTE_DIMENSIONS):
        profile_df[f"{dim}_raw"] = raw_matrix[:, idx].round(4)
        profile_df[f"{dim}_norm"] = norm_matrix[:, idx].round(4)
        profile_df[dim] = norm_matrix[:, idx].round(4)

    dominant_indices = np.argmax(raw_matrix, axis=1)
    profile_df["dominant_taste"] = [TASTE_DIMENSIONS[i] for i in dominant_indices]

    entropies = []
    balances = []
    for row_p in norm_matrix:
        h_val, j_val = compute_shannon_entropy_and_balance(row_p)
        entropies.append(h_val)
        balances.append(j_val)

    profile_df["taste_entropy"] = entropies
    profile_df["taste_balance"] = balances

    # Save recipe_taste_features.csv
    output_csv = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
    profile_df.to_csv(output_csv, index=False)

    # Visualization 1: Distribution of recipe taste dimensions across corpus
    plt.figure(figsize=(10, 6))
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
    plt.title("Distribution of Normalized Recipe Taste Profiles Across Corpus")
    plt.xlabel("Basic Taste Modality")
    plt.ylabel("Normalized Taste Share (L1 Simplex)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "taste_profile_distributions.png", dpi=200)
    plt.close()

    # Visualization 2: Example recipe taste profiles (1 representative recipe from 5 cuisines)
    sample_examples = profile_df.groupby("cuisine").first().reset_index().head(5)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(sample_examples))
    colors = ["#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#e76f51"]

    for dim, color in zip(TASTE_DIMENSIONS, colors):
        values = sample_examples[dim].to_numpy()
        ax.bar(
            sample_examples["cuisine"] + " (#" + sample_examples["recipe_id"].astype(str) + ")",
            values,
            bottom=bottom,
            label=dim.capitalize(),
            color=color,
            edgecolor="white",
        )
        bottom += values

    ax.set_ylabel("Normalized Taste Share")
    ax.set_title("Example Recipe-Level Taste Profiles Across Cuisines")
    ax.legend(title="Taste Modality", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "example_recipe_profiles.png", dpi=200)
    plt.close()

    # Select first recipe for the worked example in the methodology report
    ex = profile_df.iloc[0]
    ex_ings = rec_ings_df[
        (rec_ings_df["recipe_id"] == ex["recipe_id"]) & (rec_ings_df["is_matched"])
    ].merge(matched_lex[TASTE_DIMENSIONS], left_on="ingredient_norm", right_index=True)

    methodology_md = (
        "# Deliverable 2: Recipe Taste Profile Methodology\n\n"
        "## 1. Mathematical Formulation\n"
        "For a recipe r with matched ingredient set M_r (size >= 2), each ingredient i in M_r "
        "maps to a 5-dimensional gustatory vector t(i) in [0, 1]^5 over "
        "D = {sweet, sour, bitter, salty, umami}.\n\n"
        "1. **Unweighted Mean Intensity:** u_d(r) = (1 / |M_r|) * sum_{i in M_r} t_d(i)\n"
        "2. **L1-Normalized Taste Profile:** p_d(r) = u_d(r) / sum_{d' in D} u_{d'}(r)\n"
        "3. **Taste Diversity & Balance (Normalized Shannon Entropy):** "
        "H(r) = -sum_{d in D} p_d(r) ln p_d(r), and J(r) = H(r) / ln(5) in [0, 1]\n\n"
        f"## 2. Worked Example from Corpus: Recipe ID `{int(ex['recipe_id'])}` ({ex['cuisine']})\n"
        "### Matched Ingredient Vectors\n"
        f"{ex_ings[['ingredient_norm'] + TASTE_DIMENSIONS].to_markdown(index=False)}\n\n"
        "### Aggregated Recipe Profile\n"
        f"- **Raw Mean Vector (u):** sweet={ex['sweet_raw']:.4f}, sour={ex['sour_raw']:.4f}, "
        f"bitter={ex['bitter_raw']:.4f}, salty={ex['salty_raw']:.4f}, umami={ex['umami_raw']:.4f}\n"
        f"- **Normalized Profile (p):** sweet={ex['sweet_norm']:.4f}, sour={ex['sour_norm']:.4f}, "
        f"bitter={ex['bitter_norm']:.4f}, salty={ex['salty_norm']:.4f}, umami={ex['umami_norm']:.4f}\n"
        f"- **Dominant Taste:** `{ex['dominant_taste']}`\n"
        f"- **Taste Entropy (H):** `{ex['taste_entropy']:.4f}` nats | "
        f"**Taste Balance (J):** `{ex['taste_balance']:.4f}`\n\n"
        "## 3. Methodological Scope & Sensory Disclaimer\n"
        "This representation is an **ingredient-composition proxy** derived from unweighted presence "
        "of matched ingredients. It is **not** equivalent to gas chromatography-mass spectrometry (GC-MS), "
        "electronic tongue laboratory measurements, or trained human sensory panel evaluation.\n"
    )
    (REPORTS_DIR / "taste_profile_methodology.md").write_text(methodology_md, encoding="utf-8")
    return profile_df
