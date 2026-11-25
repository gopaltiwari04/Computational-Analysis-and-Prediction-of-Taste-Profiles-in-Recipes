"""
Deliverable 3: Cross-Cuisine Taste Profile Analysis.
Performs descriptive statistics, non-parametric Kruskal-Wallis omnibus tests with
Holm-Bonferroni multiple-comparison adjustment, eta-squared effect sizes with
bootstrap confidence intervals, Aitchison compositional CLR/PERMANOVA analysis,
standardized centroid PCA biplots, and dynamic corpus-accurate reporting.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_SEED,
    RunContext,
    TASTE_DIMENSIONS,
    default_context,
)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down multiple-testing correction:
    1. Sort p-values ascending: p_(1) <= ... <= p_(m).
    2. Compute p_adj_(k) = max_{j <= k} min(1.0, (m - j) * p_(j)) (0-indexed j).
    3. Map back to original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    validated = []
    for p in p_values:
        val = float(p)
        if np.isnan(val) or val < 0.0 or val > 1.0:
            raise ValueError(f"p-values must lie in [0.0, 1.0], got {p}.")
        validated.append(val)

    indexed = sorted(enumerate(validated), key=lambda x: x[1])
    corrected = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj)
        corrected[orig_idx] = running_max

    return corrected


def compute_clr_transform(simplex_matrix: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Centered Log-Ratio (clr) transformation for compositional data (Aitchison, 1986)
    with multiplicative zero replacement (Martin-Fernandez et al., 2003):
    - Replaces exact zeros with eps and multiplicatively rescales positive parts
      so each row remains strictly on the simplex before log-centering.
    - Maps from D-part simplex into unconstrained Euclidean space summing to 0 per row.
    """
    mat = np.asarray(simplex_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[1] == 0:
        raise ValueError(f"Expected 2D simplex matrix, got shape {mat.shape}.")
    if np.any(mat < -1e-12) or np.any(~np.isfinite(mat)):
        raise ValueError("Simplex matrix must contain finite non-negative numbers.")

    mat = np.maximum(mat, 0.0)
    replaced = np.empty_like(mat)
    for i, row in enumerate(mat):
        zero_mask = row <= 1e-12
        n_zeros = int(np.sum(zero_mask))
        if n_zeros == 0:
            replaced[i] = row / np.sum(row)
        elif n_zeros == len(row):
            replaced[i] = np.full_like(row, 1.0 / len(row))
        else:
            row_norm = row / np.sum(row)
            delta = eps * n_zeros
            if delta >= 1.0:
                delta = 0.5
                eps_row = delta / n_zeros
            else:
                eps_row = eps
            scale = 1.0 - delta
            out_row = np.where(zero_mask, eps_row, row_norm * scale)
            replaced[i] = out_row

    log_mat = np.log(replaced)
    return log_mat - np.mean(log_mat, axis=1, keepdims=True)


def run_compositional_permanova(
    clr_matrix: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 299,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Multivariate one-way PERMANOVA in Aitchison (CLR Euclidean) geometry.
    Tests joint distributional equality across all 5 compositional parts simultaneously,
    avoiding the independence violation of testing simplex parts separately.
    """
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    k = len(unique_labels)
    n = len(labels)
    if k < 2 or n <= k:
        return {"pseudo_F": 0.0, "r_squared": 0.0, "p_value": 1.0, "n_permutations": 0}

    centered = clr_matrix - np.mean(clr_matrix, axis=0, keepdims=True)
    ss_total = float(np.sum(centered ** 2))
    if ss_total <= 1e-12:
        return {"pseudo_F": 0.0, "r_squared": 0.0, "p_value": 1.0, "n_permutations": 0}

    def _ss_between(group_inv: np.ndarray) -> float:
        counts = np.bincount(group_inv, minlength=k).astype(float)
        ss_b = 0.0
        for col in range(centered.shape[1]):
            sums = np.bincount(group_inv, weights=centered[:, col], minlength=k)
            valid = counts > 0
            ss_b += float(np.sum((sums[valid] ** 2) / counts[valid]))
        return ss_b

    ss_between_obs = _ss_between(inverse)
    ss_within_obs = max(1e-12, ss_total - ss_between_obs)
    r2_obs = ss_between_obs / ss_total
    f_obs = (ss_between_obs / (k - 1)) / (ss_within_obs / (n - k))

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        perm_inv = rng.permutation(inverse)
        ss_b_perm = _ss_between(perm_inv)
        ss_w_perm = max(1e-12, ss_total - ss_b_perm)
        f_perm = (ss_b_perm / (k - 1)) / (ss_w_perm / (n - k))
        if f_perm >= f_obs - 1e-12:
            exceed += 1

    p_val = float((exceed + 1) / (n_permutations + 1))
    return {
        "pseudo_F": round(float(f_obs), 4),
        "r_squared": round(float(r2_obs), 4),
        "p_value": p_val,
        "n_permutations": int(n_permutations),
    }


def _bootstrap_kruskal_eta_ci(
    values: np.ndarray,
    group_codes: np.ndarray,
    k_groups: int,
    n_boot: int = 100,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """Compute percentile bootstrap 95% CI for Kruskal-Wallis eta_H^2."""
    n = len(values)
    if n <= k_groups + 2 or k_groups < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    etas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v_b = values[idx]
        g_b = group_codes[idx]
        groups = [v_b[g_b == g] for g in range(k_groups)]
        groups = [grp for grp in groups if len(grp) > 0]
        if len(groups) < 2:
            continue
        try:
            h_b, _ = stats.kruskal(*groups)
            eta_b = max(0.0, float((h_b - len(groups) + 1) / max(1, n - len(groups))))
            etas.append(eta_b)
        except ValueError:
            continue
    if not etas:
        return 0.0, 0.0
    return round(float(np.percentile(etas, 2.5)), 4), round(float(np.percentile(etas, 97.5)), 4)


def _format_pvalue(p: float) -> str:
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.6f}"


def _load_manifest(context: RunContext) -> Dict[str, object]:
    if context.manifest_path.exists():
        with open(context.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"is_synthetic": False, "dataset_name": "Unknown Recipe Corpus"}


def run_deliverable_3(ctx: Optional[RunContext] = None) -> pd.DataFrame:
    """
    Execute Deliverable 3:
    1. Loads recipe taste features and excludes zero-vector 'unspecified' profiles from simplex tests.
    2. Computes per-cuisine descriptive statistics, sample sizes, 95% CIs, and distinctiveness z-lifts.
    3. Runs Kruskal-Wallis omnibus tests with Holm-Bonferroni correction and eta_H^2 + 95% bootstrap CIs.
    4. Runs Aitchison CLR sensitivity univariate tests and multivariate PERMANOVA on the 4-simplex.
    5. Generates heatmap, boxplots, distinctiveness z-lift bars, and standardized PCA biplot.
    6. Writes cuisine_comparison.csv and dynamically generated reports/cross_cuisine_analysis.md.
    """
    context = ctx or default_context()
    context.ensure()
    manifest = _load_manifest(context)
    is_synthetic = bool(manifest.get("is_synthetic", False))
    corpus_tag = "SYNTHETIC SMOKE TEST" if is_synthetic else "REAL CORPUS"

    raw_df = pd.read_csv(context.processed_dir / "recipe_taste_features.csv")
    n_unspecified = int((raw_df["dominant_taste"] == "unspecified").sum())
    df = raw_df[raw_df["dominant_taste"] != "unspecified"].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid non-zero taste profiles available for Deliverable 3 analysis.")

    cuisines = sorted(df["cuisine"].unique())
    n_cuisines = len(cuisines)
    n_total = len(df)

    evenness_col = "taste_evenness" if "taste_evenness" in df.columns else "taste_balance"

    # 1. Per-cuisine descriptive statistics
    summary_rows = []
    global_means = df[TASTE_DIMENSIONS].mean()
    global_stds = df[TASTE_DIMENSIONS].std().replace(0, 1e-9)

    for cuisine, grp in df.groupby("cuisine"):
        n_rec = len(grp)
        ev_vals = grp[evenness_col].dropna().to_numpy(dtype=float)
        ev_mean = float(np.mean(ev_vals)) if len(ev_vals) else 0.0
        ev_std = float(np.std(ev_vals, ddof=1)) if len(ev_vals) > 1 else 0.0

        row: Dict[str, object] = {
            "cuisine": cuisine,
            "n_recipes": n_rec,
            "mean_taste_evenness": round(ev_mean, 4),
            "std_taste_evenness": round(ev_std, 4),
        }
        best_dim = None
        best_z = -999.0
        for dim in TASTE_DIMENSIONS:
            vals = grp[dim].dropna().to_numpy(dtype=float)
            m_val = float(np.mean(vals))
            s_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se_val = s_val / np.sqrt(max(1, len(vals)))
            med_val = float(np.median(vals))
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            z_lift = float((m_val - global_means[dim]) / global_stds[dim])

            row[f"{dim}_mean"] = round(m_val, 4)
            row[f"{dim}_std"] = round(s_val, 4)
            row[f"{dim}_ci95_halfwidth"] = round(1.96 * se_val, 4)
            row[f"{dim}_median"] = round(med_val, 4)
            row[f"{dim}_iqr"] = round(q75 - q25, 4)
            row[f"{dim}_z_distinctiveness"] = round(z_lift, 4)

            if z_lift > best_z:
                best_z = z_lift
                best_dim = dim

        row["most_distinctive_taste"] = best_dim
        row["max_distinctiveness_z"] = round(best_z, 4)
        summary_rows.append(row)

    cuisine_comp_df = pd.DataFrame(summary_rows).sort_values("cuisine").reset_index(drop=True)

    # 2. Hypothesis testing across cuisines
    tested_dimensions = TASTE_DIMENSIONS + [evenness_col]
    _, cuisine_codes = np.unique(df["cuisine"].to_numpy(), return_inverse=True)
    stat_rows = []
    raw_pvals = []

    for idx_dim, dim in enumerate(tested_dimensions):
        groups = [grp[dim].dropna().to_numpy() for _, grp in df.groupby("cuisine")]
        groups = [g for g in groups if len(g) > 0]
        k_groups = len(groups)
        kw_stat, kw_p = stats.kruskal(*groups)
        f_stat, f_p = stats.f_oneway(*groups)

        eta_sq = max(0.0, float((kw_stat - k_groups + 1) / max(1, (n_total - k_groups))))
        ci_low, ci_high = _bootstrap_kruskal_eta_ci(
            df[dim].to_numpy(dtype=float),
            cuisine_codes,
            k_groups,
            n_boot=80,
            seed=RANDOM_SEED + idx_dim,
        )

        raw_pvals.append(float(kw_p))
        stat_rows.append(
            {
                "dimension": dim,
                "df_between": k_groups - 1,
                "df_within": n_total - k_groups,
                "kruskal_H": round(float(kw_stat), 4),
                "p_value_raw": float(kw_p),
                "p_value_raw_str": _format_pvalue(float(kw_p)),
                "anova_F": round(float(f_stat), 4),
                "effect_size_eta_sq": round(float(eta_sq), 4),
                "eta_sq_ci95": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "effect_magnitude": (
                    "large" if eta_sq >= 0.14 else "medium" if eta_sq >= 0.06 else "small"
                ),
            }
        )

    adj_pvals = holm_bonferroni_correction(raw_pvals)
    for row, adj_p in zip(stat_rows, adj_pvals):
        row["p_value_holm_adjusted"] = float(adj_p)
        row["p_value_holm_str"] = _format_pvalue(float(adj_p))
        row["significant_at_0_05"] = bool(adj_p < 0.05)

    stats_df = pd.DataFrame(stat_rows)

    # 3. Compositional Sensitivity Analysis: Multiplicative-Zero CLR Kruskal-Wallis + PERMANOVA
    clr_matrix = compute_clr_transform(df[TASTE_DIMENSIONS].to_numpy(dtype=float))
    clr_stat_rows = []
    clr_raw_pvals = []
    for idx, dim in enumerate(TASTE_DIMENSIONS):
        clr_groups = [clr_matrix[df["cuisine"] == c, idx] for c in cuisines]
        c_stat, c_p = stats.kruskal(*clr_groups)
        clr_eta = max(0.0, float((c_stat - n_cuisines + 1) / max(1, (n_total - n_cuisines))))
        clr_raw_pvals.append(float(c_p))
        clr_stat_rows.append(
            {
                "dimension": f"{dim}_clr",
                "kruskal_H": round(float(c_stat), 4),
                "effect_size_eta_sq": round(clr_eta, 4),
                "p_value_raw_str": _format_pvalue(float(c_p)),
            }
        )
    clr_adj = holm_bonferroni_correction(clr_raw_pvals)
    for row, adj_p in zip(clr_stat_rows, clr_adj):
        row["p_value_holm_str"] = _format_pvalue(float(adj_p))
        row["significant_at_0_05"] = bool(adj_p < 0.05)
    clr_df = pd.DataFrame(clr_stat_rows)

    permanova = run_compositional_permanova(clr_matrix, df["cuisine"].to_numpy(), n_permutations=299)

    cuisine_comp_df.to_csv(context.results_dir / "cuisine_comparison.csv", index=False)

    # Visualization 1: Cuisine Mean Taste Heatmap
    heatmap_matrix = cuisine_comp_df.set_index("cuisine")[[f"{d}_mean" for d in TASTE_DIMENSIONS]]
    heatmap_matrix.columns = [d.capitalize() for d in TASTE_DIMENSIONS]

    fig_height = max(5.5, 0.42 * n_cuisines)
    plt.figure(figsize=(9.5, fig_height))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Normalized Taste Share"},
    )
    plt.title(
        f"Cross-Cuisine Mean Modeled Taste Shares ({n_cuisines} Cuisines, N={n_total:,})\n"
        f"[{corpus_tag} — Unweighted Ingredient Proxy]",
        fontsize=11,
    )
    plt.ylabel("Cuisine")
    plt.xlabel("Basic Gustatory Modality")
    plt.tight_layout()
    plt.savefig(context.results_dir / "cuisine_taste_heatmap.png", dpi=200)
    plt.close()

    # Visualization 2: Boxplots across top cuisines by sample size
    top_k_cuisines = (
        cuisine_comp_df.sort_values("n_recipes", ascending=False)["cuisine"]
        .head(min(6, n_cuisines))
        .tolist()
    )
    melted_top = df[df["cuisine"].isin(top_k_cuisines)].melt(
        id_vars=["recipe_id", "cuisine"],
        value_vars=TASTE_DIMENSIONS,
        var_name="Taste Dimension",
        value_name="Normalized Share",
    )
    plt.figure(figsize=(12, 6.5))
    sns.boxplot(
        data=melted_top,
        x="Taste Dimension",
        y="Normalized Share",
        hue="cuisine",
        palette="tab10",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    plt.title(
        f"Modeled Taste Share Distributions Across Top {len(top_k_cuisines)} Cuisines [{corpus_tag}]\n"
        "(Boxes: IQR; Horizontal lines: median; Circles: mean)",
        fontsize=11,
    )
    plt.xlabel("Basic Gustatory Modality")
    plt.ylabel("Normalized Taste Share")
    plt.legend(title="Cuisine", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(context.results_dir / "cuisine_taste_boxplots.png", dpi=200)
    plt.close()

    # Visualization 3: Distinctiveness Z-Score Lift Bar Chart
    plt.figure(figsize=(max(9.5, 0.52 * n_cuisines), 5.8))
    sorted_distinct = cuisine_comp_df.sort_values("max_distinctiveness_z", ascending=False)
    bars = plt.bar(
        sorted_distinct["cuisine"],
        sorted_distinct["max_distinctiveness_z"],
        color="#2a9d8f",
        edgecolor="black",
    )
    for bar, dim_label in zip(bars, sorted_distinct["most_distinctive_taste"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.025,
            str(dim_label).upper(),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(
        f"Most Elevated Modality per Cuisine (Recipe-Weighted Standardized Z-Lift) [{corpus_tag}]",
        fontsize=11,
    )
    plt.ylabel("Z-Score Elevation vs. Corpus Mean")
    plt.xlabel("Cuisine")
    plt.xticks(rotation=45, ha="right")
    y_min = min(0.0, float(sorted_distinct["max_distinctiveness_z"].min()) - 0.1)
    y_max = max(0.2, float(sorted_distinct["max_distinctiveness_z"].max()) * 1.25 + 0.15)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(context.results_dir / "cuisine_distinctive_tastes.png", dpi=200)
    plt.close()

    # Visualization 4: Standardized Centroid PCA Biplot
    pca_var_sum = 0.0
    if n_cuisines >= 2:
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(heatmap_matrix.to_numpy())
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        coords = pca.fit_transform(scaled_matrix)
        pca_var_sum = float(pca.explained_variance_ratio_[:2].sum() * 100.0)
        loadings = pca.components_.T

        plt.figure(figsize=(9.5, 6.8))
        plt.scatter(coords[:, 0], coords[:, 1], s=110, c="#e76f51", edgecolors="black", zorder=3)
        for idx, c_name in enumerate(heatmap_matrix.index):
            plt.annotate(
                c_name,
                (coords[idx, 0] + 0.06, coords[idx, 1] + 0.06),
                fontsize=8.5,
                fontweight="normal",
            )
        arrow_scale = max(1.0, float(np.max(np.abs(coords))) * 0.75)
        for j, dim_name in enumerate(heatmap_matrix.columns):
            dx, dy = loadings[j, 0] * arrow_scale, loadings[j, 1] * arrow_scale
            plt.arrow(0, 0, dx, dy, color="#264653", alpha=0.65, width=0.015, head_width=0.09, zorder=2)
            plt.text(dx * 1.12, dy * 1.12, dim_name, color="#264653", fontsize=9, fontweight="bold")

        plt.axhline(0, color="lightgrey", linestyle="--")
        plt.axvline(0, color="lightgrey", linestyle="--")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Centroid Variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Centroid Variance)")
        plt.title(
            f"Exploratory PCA Biplot of Standardized Cuisine Mean Taste Shares (K={n_cuisines}) [{corpus_tag}]\n"
            "(Centroid projection only; ignores large within-cuisine recipe dispersion)",
            fontsize=10.5,
        )
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(context.results_dir / "cuisine_pca_projection.png", dpi=200)
        plt.close()

    # Dynamically summarize empirical findings from computed results (zero hardcoded stories!)
    top_by_dim_lines = []
    for dim in TASTE_DIMENSIONS:
        grp_dim = cuisine_comp_df[cuisine_comp_df["most_distinctive_taste"] == dim].sort_values(
            "max_distinctiveness_z", ascending=False
        )
        if not grp_dim.empty:
            examples = ", ".join(
                f"{r['cuisine']} (z={r['max_distinctiveness_z']:+.2f}, mean={r[f'{dim}_mean']:.3f})"
                for _, r in grp_dim.head(4).iterrows()
            )
            top_by_dim_lines.append(f"   - **{dim.capitalize()} elevation:** {examples}")
    dynamic_cluster_bullets = "\n".join(top_by_dim_lines)

    strongest_effect = stats_df.sort_values("effect_size_eta_sq", ascending=False).iloc[0]
    weakest_effect = stats_df.sort_values("effect_size_eta_sq", ascending=True).iloc[0]
    highest_evenness = cuisine_comp_df.sort_values("mean_taste_evenness", ascending=False).iloc[0]
    lowest_evenness = cuisine_comp_df.sort_values("mean_taste_evenness", ascending=True).iloc[0]

    corpus_banner = (
        "> **Corpus Notice:** Computed on **SYNTHETIC SMOKE TEST (`--use-sample-data`)** — "
        "results illustrate software behavior only and MUST NOT be cited as empirical culinary evidence."
        if is_synthetic
        else "> **Corpus Notice:** Computed on **REAL EMPIRICAL CORPUS** (`"
        + str(manifest.get("dataset_name", "External Corpus"))
        + "`). All findings remain **exploratory observational associations** on an unweighted ingredient proxy."
    )

    stats_display = stats_df[
        [
            "dimension",
            "df_between",
            "df_within",
            "kruskal_H",
            "p_value_raw_str",
            "p_value_holm_str",
            "effect_size_eta_sq",
            "eta_sq_ci95",
            "effect_magnitude",
            "significant_at_0_05",
        ]
    ].rename(
        columns={
            "p_value_raw_str": "p_value_raw",
            "p_value_holm_str": "p_value_holm_adjusted",
        }
    )

    clr_display = clr_df[
        ["dimension", "kruskal_H", "effect_size_eta_sq", "p_value_raw_str", "p_value_holm_str", "significant_at_0_05"]
    ].rename(
        columns={
            "p_value_raw_str": "p_value_raw",
            "p_value_holm_str": "p_value_holm_adjusted",
        }
    )

    summary_cols = [
        "cuisine",
        "n_recipes",
        "sweet_mean",
        "sour_mean",
        "bitter_mean",
        "salty_mean",
        "umami_mean",
        "most_distinctive_taste",
        "max_distinctiveness_z",
        "mean_taste_evenness",
    ]

    report_md = f"""# Deliverable 3: Cross-Cuisine Taste Profile Analysis

{corpus_banner}

## 1. Sample Sizes & Per-Cuisine Descriptive Summary
We analyze {n_cuisines} cuisines totaling $N = {n_total:,}$ non-zero taste profiles (excluding {n_unspecified:,} unspecified zero-vector profiles). Each cuisine's sample size, mean normalized taste shares, most elevated modality (standardized $z$-score lift vs. corpus mean/SD), and mean modeled taste-share evenness ($J$) are reported below:

{cuisine_comp_df[summary_cols].to_markdown(index=False)}

## 2. Exploratory Omnibus Hypothesis Tests & Effect Sizes
- **Null Hypothesis ($H_0$):** Across the $k = {n_cuisines}$ represented cuisines, the distributions of each modeled taste share (and evenness $J$) are identical.
- **Alternative Hypothesis ($H_1$):** At least one cuisine exhibits a stochastically shifted distribution.
- **Family-Wise Significance Threshold:** $\\alpha = 0.05$ controlled via the **Holm–Bonferroni step-down procedure** across $m = {len(tested_dimensions)}$ tested dimensions.
- **Effect Size & Uncertainty:** Non-parametric Kruskal–Wallis $\\eta_H^2 = (H - k + 1)/(N - k)$ (Tomczak & Tomczak, 2014) with 95% percentile bootstrap confidence intervals.

{stats_display.to_markdown(index=False)}

## 3. Compositional Closure Limitation & Joint Multivariate Check (Aitchison CLR + PERMANOVA)
Because the five normalized taste shares sum to $1.0$ ($\\sum_d p_d = 1$), they occupy a 4-dimensional simplex with inherent negative covariance: increasing one share mathematically lowers at least one other share. Consequently, testing the 5 dimensions separately violates independence across tests.

To verify that cross-cuisine separation persists in unconstrained Aitchison geometry:
1. **Multivariate Compositional PERMANOVA (Joint 5-Part Test):** Using multiplicative zero replacement ($\\epsilon = 10^{{-4}}$) and Centered Log-Ratio (CLR) coordinates, one-way PERMANOVA across all {n_cuisines} cuisines yields:
   - **Pseudo-$F$:** `{permanova['pseudo_F']:.4f}`
   - **Compositional Variance Explained ($R^2_{{\\text{{Aitchison}}}}$):** `{permanova['r_squared']*100:.2f}%`
   - **Permutation $p$-value ({permanova['n_permutations']} permutations):** `{_format_pvalue(permanova['p_value'])}`
2. **Univariate CLR Sensitivity Checks (Holm–Bonferroni Adjusted):**

{clr_display.to_markdown(index=False)}

## 4. Data-Driven Empirical Summary
1. **Effect Size Ordering:** Cross-cuisine divergence is largest on **`{strongest_effect['dimension']}`** ($\\eta_H^2 = {strongest_effect['effect_size_eta_sq']:.4f}$, {strongest_effect['effect_magnitude']}) and smallest on **`{weakest_effect['dimension']}`** ($\\eta_H^2 = {weakest_effect['effect_size_eta_sq']:.4f}$, {weakest_effect['effect_magnitude']}). Note that with large sample sizes, even small shifts yield tiny $p$-values; effect sizes ($\\eta_H^2$, $R^2_{{\\text{{Aitchison}}}}$) govern substantive magnitude.
2. **Cuisine Elevation Profiles ($z$-Score Lifts):**
{dynamic_cluster_bullets}
3. **Taste-Share Evenness Range:** Mean Pielou evenness $J$ spans from `{lowest_evenness['cuisine']}` ($J = {lowest_evenness['mean_taste_evenness']:.4f}$) to `{highest_evenness['cuisine']}` ($J = {highest_evenness['mean_taste_evenness']:.4f}$).
4. **Exploratory PCA Biplot:** The first two principal components explain **{pca_var_sum:.1f}%** of variance among the $K={n_cuisines}$ standardized cuisine centroids (`results/cuisine_pca_projection.png`). Because centroid PCA collapses $N={n_total:,}$ recipes down to $K={n_cuisines}$ averages, it omits wide within-cuisine dispersion (`results/cuisine_taste_boxplots.png`) and must be treated strictly as exploratory visualization.

## 5. Exploratory vs. Confirmatory Scope & Limitations
All analyses are **exploratory and observational** on ingredient-presence proxies. Statistically significant $p$-values do not establish causal culinary rules, biological taste determinism, or account for omitted gram quantities, sub-regional cuisines, dish types (desserts vs. entrees), or tabletop seasoning.
"""
    (context.reports_dir / "cross_cuisine_analysis.md").write_text(report_md, encoding="utf-8")
    return cuisine_comp_df
