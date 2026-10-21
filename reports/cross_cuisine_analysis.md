# Deliverable 3: Cross-Cuisine Taste Profile Analysis

## 1. Sample Sizes & Cuisine Summary Statistics
We evaluate taste differences across 20 cuisines ($N = 37,033$ total sufficient recipes). Each cuisine's sample size, mean normalized taste shares, most distinctive taste modality, and mean distributional evenness ($J$) are detailed below:

| cuisine      |   n_recipes |   sweet_mean |   sour_mean |   bitter_mean |   salty_mean |   umami_mean | most_distinctive_taste   |   max_distinctiveness_z |   mean_taste_balance |
|:-------------|------------:|-------------:|------------:|--------------:|-------------:|-------------:|:-------------------------|------------------------:|---------------------:|
| brazilian    |         352 |       0.2839 |      0.1565 |        0.169  |       0.1879 |       0.2027 | sour                     |                  0.2763 |               0.8189 |
| british      |         716 |       0.3511 |      0.0844 |        0.1162 |       0.2603 |       0.188  | sweet                    |                  0.6375 |               0.7739 |
| cajun_creole |        1451 |       0.2483 |      0.1041 |        0.1986 |       0.1825 |       0.2664 | bitter                   |                  0.2565 |               0.8765 |
| chinese      |        2571 |       0.2537 |      0.0914 |        0.1082 |       0.2358 |       0.3109 | umami                    |                  0.6218 |               0.8692 |
| filipino     |         720 |       0.277  |      0.0944 |        0.1167 |       0.2414 |       0.2705 | umami                    |                  0.2856 |               0.8321 |
| french       |        2422 |       0.3046 |      0.1226 |        0.1611 |       0.216  |       0.1958 | sweet                    |                  0.3459 |               0.8041 |
| greek        |        1122 |       0.1944 |      0.1929 |        0.2193 |       0.1965 |       0.1969 | sour                     |                  0.6149 |               0.8934 |
| indian       |        2893 |       0.2436 |      0.1433 |        0.2277 |       0.2041 |       0.1812 | bitter                   |                  0.5303 |               0.8769 |
| irish        |         613 |       0.3224 |      0.0766 |        0.1422 |       0.2761 |       0.1828 | sweet                    |                  0.4575 |               0.7858 |
| italian      |        7292 |       0.1986 |      0.1139 |        0.2029 |       0.2271 |       0.2576 | bitter                   |                  0.2967 |               0.855  |
| jamaican     |         487 |       0.2806 |      0.1168 |        0.1889 |       0.2064 |       0.2073 | sweet                    |                  0.1956 |               0.8697 |
| japanese     |        1280 |       0.2698 |      0.1115 |        0.1118 |       0.2398 |       0.2672 | umami                    |                  0.2581 |               0.838  |
| korean       |         796 |       0.2807 |      0.0876 |        0.1332 |       0.2027 |       0.2959 | umami                    |                  0.4967 |               0.8785 |
| mexican      |        5996 |       0.2037 |      0.1721 |        0.1838 |       0.1868 |       0.2536 | sour                     |                  0.4211 |               0.8862 |
| moroccan     |         783 |       0.255  |      0.1408 |        0.2616 |       0.1638 |       0.1787 | bitter                   |                  0.8487 |               0.8851 |
| russian      |         471 |       0.3293 |      0.1239 |        0.1213 |       0.2284 |       0.1972 | sweet                    |                  0.5005 |               0.8098 |
| southern_us  |        3897 |       0.3249 |      0.098  |        0.1353 |       0.2585 |       0.1833 | sweet                    |                  0.4734 |               0.7851 |
| spanish      |         918 |       0.2422 |      0.1419 |        0.2016 |       0.1923 |       0.2219 | bitter                   |                  0.2851 |               0.8609 |
| thai         |        1478 |       0.26   |      0.1586 |        0.1213 |       0.1931 |       0.267  | sour                     |                  0.2952 |               0.8907 |
| vietnamese   |         775 |       0.2719 |      0.1386 |        0.1195 |       0.2056 |       0.2643 | umami                    |                  0.2343 |               0.8861 |

## 2. Omnibus Hypothesis Testing & Holm–Bonferroni Correction
We test the null hypothesis $H_0$: *taste modality distributions are identical across world cuisines*, against the alternative $H_1$: *at least one cuisine deviates systematically*. Because simplex proportions are bounded in $[0, 1]$ and right-skewed, we employ the non-parametric **Kruskal–Wallis $H$-test** (with degrees of freedom $df = 19$), supplemented by one-way ANOVA $F$-statistics and Kruskal–Wallis effect sizes ($\eta_H^2 = \frac{H - k + 1}{N - k}$, Tomczak & Tomczak, 2014).

All $p$-values are adjusted using the **Holm–Bonferroni step-down procedure** (family-wise significance threshold $\alpha = 0.05$):

| dimension     |   df_between |   df_within |   kruskal_H |   p_value_raw |   anova_F |   effect_size_eta_sq | effect_magnitude   |   p_value_holm_adjusted | significant_at_0_05   |
|:--------------|-------------:|------------:|------------:|--------------:|----------:|---------------------:|:-------------------|------------------------:|:----------------------|
| sweet         |           19 |       37013 |     3839.71 |  0            |  184.329  |               0.1032 | medium             |            0            | True                  |
| sour          |           19 |       37013 |     3931.58 |  0            |  167.99   |               0.1057 | medium             |            0            | True                  |
| bitter        |           19 |       37013 |     6443.41 |  0            |  334.735  |               0.1736 | large              |            0            | True                  |
| salty         |           19 |       37013 |     1485.28 |  6.39282e-304 |   65.3659 |               0.0396 | small              |            6.39282e-304 | True                  |
| umami         |           19 |       37013 |     4766.65 |  0            |  249.842  |               0.1283 | medium             |            0            | True                  |
| taste_balance |           19 |       37013 |     3101.58 |  0            |  135.642  |               0.0833 | medium             |            0            | True                  |

## 3. Compositional Sensitivity Analysis (Centered Log-Ratio)
Because normalized recipe taste shares lie on the 4-simplex ($\sum_{d} p_d = 1$), an increase in one modality mathematically forces a decrease in others (negative compositional covariance). To confirm that statistical significance is not merely an artifact of simplex closure, we re-test differences using Aitchison's **Centered Log-Ratio (clr)** transformation:

| dimension   |   kruskal_H |   p_value_raw |   p_value_holm_adjusted | significant_at_0_05   |
|:------------|------------:|--------------:|------------------------:|:----------------------|
| sweet_clr   |     3682.42 |             0 |                       0 | True                  |
| sour_clr    |     3300.85 |             0 |                       0 | True                  |
| bitter_clr  |     5405.83 |             0 |                       0 | True                  |
| salty_clr   |     2215.53 |             0 |                       0 | True                  |
| umami_clr   |     3062.11 |             0 |                       0 | True                  |

All tested modalities remain statistically significant under the clr transformation ($p_{adj} < 0.05$).

## 4. Key Empirical Observations
1. **Taste Signature Clustering:** Cuisines exhibit distinct signature taste elevations ($z$-score lift):
   - East Asian cuisines (Japanese, Chinese, Korean, Filipino, Vietnamese) exhibit dominant **umami** and **salty** enrichment driven by fermented condiments (soy sauce, miso, fish sauce).
   - Southeast Asian and Latin American cuisines (Thai, Mexican) exhibit elevated **sour** (citrus juice, tamarind, vinegar) and **taste evenness ($J$)**, balancing acid, sugar, and savory elements.
   - Indian cuisine displays elevated **bitter** and savory spice signatures driven by polyphenolic/turmeric/cumin/fenugreek additions.
   - Western bakery/comfort recipes (Southern US, British, French) show marked **sweet** enrichment.
2. **Within-Cuisine Variance vs. Between-Cuisine Distance:** As shown in `results/cuisine_taste_boxplots.png`, individual dishes vary broadly around their cultural centroids.
3. **PCA Projection:** The first two principal components explain **83.0%** of cross-cuisine taste variance (`results/cuisine_pca_projection.png`), separating ferment/umami-oriented cuisines from acid/citrus and sweetener-dominant traditions.

## 5. Methodological Scope & Non-Causal Nature
Statistical significance in Kruskal–Wallis tests demonstrates an **observational association** between regional culinary traditions and ingredient-derived taste profiles within this corpus. It does **not** establish genetic taste determinism, causal culinary rules, or account for intra-regional sub-cuisines, dish categories (e.g. desserts vs. entrees), or discretionary tabletop seasonings.
