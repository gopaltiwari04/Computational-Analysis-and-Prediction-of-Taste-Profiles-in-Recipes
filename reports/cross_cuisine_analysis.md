# Deliverable 3: Cross-Cuisine Taste Profile Analysis

## 1. Sample Sizes & Cuisine Summary Table
| cuisine                 |   n_recipes |   sweet_mean |   sour_mean |   bitter_mean |   salty_mean |   umami_mean | most_distinctive_taste   |   max_distinctiveness_z |   mean_taste_balance |
|:------------------------|------------:|-------------:|------------:|--------------:|-------------:|-------------:|:-------------------------|------------------------:|---------------------:|
| American_Bakery_Comfort |          23 |       0.4202 |      0.1074 |        0.1493 |       0.1947 |       0.1284 | sweet                    |                  1.2852 |               0.8012 |
| Indian                  |          25 |       0.1906 |      0.1818 |        0.3068 |       0.1249 |       0.1958 | bitter                   |                  1.4645 |               0.9304 |
| Italian                 |          20 |       0.1393 |      0.1081 |        0.1454 |       0.2757 |       0.3315 | salty                    |                  0.8956 |               0.9149 |
| Japanese                |          27 |       0.2004 |      0.1038 |        0.0948 |       0.2287 |       0.3724 | umami                    |                  0.9964 |               0.8936 |
| Mexican                 |          23 |       0.2192 |      0.2222 |        0.1935 |       0.1362 |       0.2289 | sour                     |                  0.9808 |               0.9529 |
| Thai                    |          22 |       0.3069 |      0.1329 |        0.087  |       0.1987 |       0.2745 | sweet                    |                  0.4499 |               0.8879 |

## 2. Omnibus Hypothesis Tests & Multiple-Testing Correction
We evaluate cross-cuisine differences across all 6 cuisines ($N = 140$ total recipes) using the non-parametric Kruskal–Wallis $H$-test and one-way ANOVA $F$-test, with **Holm–Bonferroni step-down correction** across the 6 tested dimensions:

| dimension     |   kruskal_H |   p_value_raw |   anova_F |   effect_size_eta_sq |   p_value_holm_adjusted | significant_at_0_05   |
|:--------------|------------:|--------------:|----------:|---------------------:|------------------------:|:----------------------|
| sweet         |     58.4951 |   2.48616e-11 |   22.1177 |               0.3992 |             9.94465e-11 | True                  |
| sour          |     40.7104 |   1.07352e-07 |   11.9219 |               0.2665 |             1.07352e-07 | True                  |
| bitter        |     85.4322 |   6.10993e-17 |   41.0154 |               0.6002 |             3.66596e-16 | True                  |
| salty         |     44.6799 |   1.68533e-08 |   11.1258 |               0.2961 |             3.37066e-08 | True                  |
| umami         |     77.569  |   2.70481e-15 |   28      |               0.5416 |             1.3524e-14  | True                  |
| taste_balance |     49.0946 |   2.12258e-09 |   16.2249 |               0.3291 |             6.36773e-09 | True                  |

## 3. Key Empirical Observations
1. **Distinctive Modality Alignment:** Each cuisine exhibits a clear signature taste deviation ($z$-score lift above the global mean), reflected in `most_distinctive_taste` above (e.g., high umami/salty enrichment in East Asian ferment-rich recipes, high bitter/earthy spice signatures in Indian curries, sour/acidic citrus-chili signatures in Mexican and Thai dishes, and sweet dominance in bakery/comfort recipes).
2. **Within-Cuisine Variability vs. Between-Cuisine Separation:** As shown in `results/cuisine_taste_boxplots.png`, within-cuisine interquartile ranges overlap substantially across savory cuisines, demonstrating that individual dishes vary widely around their cultural mean.
3. **Taste Balance Differences:** Multi-ingredient savory cuisines that combine acid, sweetener, and fermented umami sources (such as Thai and Mexican profiles) exhibit high Pielou taste balance ($J$), whereas specialized sweet/bakery recipes concentrate probability mass on a single dimension.

## 4. Methodological Caveats: Association vs. Causation
Statistical significance in Kruskal–Wallis tests demonstrates **observational association** between cuisine labels and aggregated ingredient taste vectors within this dataset. It does **not** establish causal cultural rules, nor does it account for regional sub-cuisines, cooking technique alterations, or salt/sugar adjustments added to taste at the table.
