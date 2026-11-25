# Deliverable 3: Cross-Cuisine Taste Profile Analysis

> **Corpus Notice:** Computed on **REAL EMPIRICAL CORPUS** (`Kaggle / Yummly What's Cooking Dataset`). All findings remain **exploratory observational associations** on an unweighted ingredient proxy.

## 1. Sample Sizes & Per-Cuisine Descriptive Summary
We analyze 20 cuisines totaling $N = 37,447$ non-zero taste profiles (excluding 0 unspecified zero-vector profiles). Each cuisine's sample size, mean normalized taste shares, most elevated modality (standardized $z$-score lift vs. corpus mean/SD), and mean modeled taste-share evenness ($J$) are reported below:

| cuisine      |   n_recipes |   sweet_mean |   sour_mean |   bitter_mean |   salty_mean |   umami_mean | most_distinctive_taste   |   max_distinctiveness_z |   mean_taste_evenness |
|:-------------|------------:|-------------:|------------:|--------------:|-------------:|-------------:|:-------------------------|------------------------:|----------------------:|
| brazilian    |         361 |       0.2718 |      0.1698 |        0.1652 |       0.193  |       0.2002 | sweet                    |                  0.254  |                0.8287 |
| british      |         735 |       0.3119 |      0.1119 |        0.1216 |       0.2705 |       0.1841 | sweet                    |                  0.5177 |                0.7847 |
| cajun_creole |        1467 |       0.2343 |      0.122  |        0.1847 |       0.1928 |       0.2661 | bitter                   |                  0.2578 |                0.8919 |
| chinese      |        2579 |       0.2403 |      0.1057 |        0.1061 |       0.2494 |       0.2986 | umami                    |                  0.5314 |                0.8773 |
| filipino     |         714 |       0.2622 |      0.1025 |        0.1064 |       0.2558 |       0.2731 | umami                    |                  0.3117 |                0.8369 |
| french       |        2454 |       0.285  |      0.1422 |        0.1483 |       0.2221 |       0.2023 | sweet                    |                  0.341  |                0.8129 |
| greek        |        1136 |       0.1746 |      0.223  |        0.1824 |       0.2198 |       0.2002 | sour                     |                  0.6935 |                0.9056 |
| indian       |        2899 |       0.2283 |      0.1479 |        0.2403 |       0.2079 |       0.1757 | bitter                   |                  0.8289 |                0.8761 |
| irish        |         630 |       0.2775 |      0.1108 |        0.1507 |       0.2845 |       0.1764 | salty                    |                  0.4106 |                0.8136 |
| italian      |        7417 |       0.1856 |      0.1334 |        0.1729 |       0.2386 |       0.2695 | umami                    |                  0.2807 |                0.8632 |
| jamaican     |         489 |       0.2594 |      0.1254 |        0.1988 |       0.2107 |       0.2058 | bitter                   |                  0.4027 |                0.8743 |
| japanese     |        1313 |       0.2641 |      0.1309 |        0.1096 |       0.239  |       0.2564 | sweet                    |                  0.203  |                0.8545 |
| korean       |         798 |       0.2634 |      0.104  |        0.1236 |       0.2286 |       0.2805 | umami                    |                  0.3752 |                0.8884 |
| mexican      |        6035 |       0.1977 |      0.1843 |        0.166  |       0.1965 |       0.2554 | sour                     |                  0.3497 |                0.8907 |
| moroccan     |         790 |       0.2515 |      0.149  |        0.247  |       0.1707 |       0.1818 | bitter                   |                  0.8975 |                0.8881 |
| russian      |         474 |       0.2945 |      0.1566 |        0.1194 |       0.2375 |       0.1921 | sweet                    |                  0.4032 |                0.8315 |
| southern_us  |        3976 |       0.2889 |      0.1365 |        0.1293 |       0.2606 |       0.1846 | sweet                    |                  0.3664 |                0.8143 |
| spanish      |         928 |       0.23   |      0.159  |        0.183  |       0.2044 |       0.2236 | bitter                   |                  0.2406 |                0.8705 |
| thai         |        1477 |       0.2477 |      0.1658 |        0.1278 |       0.1958 |       0.2629 | umami                    |                  0.2232 |                0.9015 |
| vietnamese   |         775 |       0.2602 |      0.1446 |        0.1245 |       0.2117 |       0.2591 | umami                    |                  0.1904 |                0.8938 |

## 2. Exploratory Omnibus Hypothesis Tests & Effect Sizes
- **Null Hypothesis ($H_0$):** Across the $k = 20$ represented cuisines, the distributions of each modeled taste share (and evenness $J$) are identical.
- **Alternative Hypothesis ($H_1$):** At least one cuisine exhibits a stochastically shifted distribution.
- **Family-Wise Significance Threshold:** $\alpha = 0.05$ controlled via the **Holm–Bonferroni step-down procedure** across $m = 6$ tested dimensions.
- **Effect Size & Uncertainty:** Non-parametric Kruskal–Wallis $\eta_H^2 = (H - k + 1)/(N - k)$ (Tomczak & Tomczak, 2014) with 95% percentile bootstrap confidence intervals.

| dimension      |   df_between |   df_within |   kruskal_H | p_value_raw   | p_value_holm_adjusted   |   effect_size_eta_sq | eta_sq_ci95      | effect_magnitude   | significant_at_0_05   |
|:---------------|-------------:|------------:|------------:|:--------------|:------------------------|---------------------:|:-----------------|:-------------------|:----------------------|
| sweet          |           19 |       37427 |     3457.64 | < 1e-300      | < 1e-300                |               0.0919 | [0.0870, 0.0985] | medium             | True                  |
| sour           |           19 |       37427 |     2575.19 | < 1e-300      | < 1e-300                |               0.0683 | [0.0628, 0.0729] | medium             | True                  |
| bitter         |           19 |       37427 |     6048.29 | < 1e-300      | < 1e-300                |               0.1611 | [0.1530, 0.1670] | large              | True                  |
| salty          |           19 |       37427 |     1626.07 | < 1e-300      | < 1e-300                |               0.0429 | [0.0395, 0.0467] | small              | True                  |
| umami          |           19 |       37427 |     5039.13 | < 1e-300      | < 1e-300                |               0.1341 | [0.1285, 0.1390] | medium             | True                  |
| taste_evenness |           19 |       37427 |     2439.79 | < 1e-300      | < 1e-300                |               0.0647 | [0.0594, 0.0701] | medium             | True                  |

## 3. Compositional Closure Limitation & Joint Multivariate Check (Aitchison CLR + PERMANOVA)
Because the five normalized taste shares sum to $1.0$ ($\sum_d p_d = 1$), they occupy a 4-dimensional simplex with inherent negative covariance: increasing one share mathematically lowers at least one other share. Consequently, testing the 5 dimensions separately violates independence across tests.

To verify that cross-cuisine separation persists in unconstrained Aitchison geometry:
1. **Multivariate Compositional PERMANOVA (Joint 5-Part Test):** Using multiplicative zero replacement ($\epsilon = 10^{-4}$) and Centered Log-Ratio (CLR) coordinates, one-way PERMANOVA across all 20 cuisines yields:
   - **Pseudo-$F$:** `79.2489`
   - **Compositional Variance Explained ($R^2_{\text{Aitchison}}$):** `3.87%`
   - **Permutation $p$-value (299 permutations):** `0.003333`
2. **Univariate CLR Sensitivity Checks (Holm–Bonferroni Adjusted):**

| dimension   |   kruskal_H |   effect_size_eta_sq | p_value_raw   | p_value_holm_adjusted   | significant_at_0_05   |
|:------------|------------:|---------------------:|:--------------|:------------------------|:----------------------|
| sweet_clr   |     3453.49 |               0.0918 | < 1e-300      | < 1e-300                | True                  |
| sour_clr    |     2166.65 |               0.0574 | < 1e-300      | < 1e-300                | True                  |
| bitter_clr  |     4915.52 |               0.1308 | < 1e-300      | < 1e-300                | True                  |
| salty_clr   |     2347.56 |               0.0622 | < 1e-300      | < 1e-300                | True                  |
| umami_clr   |     3139.86 |               0.0834 | < 1e-300      | < 1e-300                | True                  |

## 4. Data-Driven Empirical Summary
1. **Effect Size Ordering:** Cross-cuisine divergence is largest on **`bitter`** ($\eta_H^2 = 0.1611$, large) and smallest on **`salty`** ($\eta_H^2 = 0.0429$, small). Note that with large sample sizes, even small shifts yield tiny $p$-values; effect sizes ($\eta_H^2$, $R^2_{\text{Aitchison}}$) govern substantive magnitude.
2. **Cuisine Elevation Profiles ($z$-Score Lifts):**
   - **Sweet elevation:** british (z=+0.52, mean=0.312), russian (z=+0.40, mean=0.294), southern_us (z=+0.37, mean=0.289), french (z=+0.34, mean=0.285)
   - **Sour elevation:** greek (z=+0.69, mean=0.223), mexican (z=+0.35, mean=0.184)
   - **Bitter elevation:** moroccan (z=+0.90, mean=0.247), indian (z=+0.83, mean=0.240), jamaican (z=+0.40, mean=0.199), cajun_creole (z=+0.26, mean=0.185)
   - **Salty elevation:** irish (z=+0.41, mean=0.284)
   - **Umami elevation:** chinese (z=+0.53, mean=0.299), korean (z=+0.38, mean=0.281), filipino (z=+0.31, mean=0.273), italian (z=+0.28, mean=0.270)
3. **Taste-Share Evenness Range:** Mean Pielou evenness $J$ spans from `british` ($J = 0.7847$) to `greek` ($J = 0.9056$).
4. **Exploratory PCA Biplot:** The first two principal components explain **76.4%** of variance among the $K=20$ standardized cuisine centroids (`results/cuisine_pca_projection.png`). Because centroid PCA collapses $N=37,447$ recipes down to $K=20$ averages, it omits wide within-cuisine dispersion (`results/cuisine_taste_boxplots.png`) and must be treated strictly as exploratory visualization.

## 5. Exploratory vs. Confirmatory Scope & Limitations
All analyses are **exploratory and observational** on ingredient-presence proxies. Statistically significant $p$-values do not establish causal culinary rules, biological taste determinism, or account for omitted gram quantities, sub-regional cuisines, dish types (desserts vs. entrees), or tabletop seasoning.
