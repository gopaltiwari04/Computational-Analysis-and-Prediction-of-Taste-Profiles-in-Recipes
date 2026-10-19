# Final Report: Computational Analysis and Prediction of Taste Profiles in Recipes

## 1. Abstract
This study presents an end-to-end computational framework for modeling and predicting the basic taste profiles (`sweet`, `sour`, `bitter`, `salty`, `umami`) of culinary recipes from their normalized ingredient lists. Building on Computational Gastronomy literature, we decouple volatile flavor compound metadata from non-volatile gustatory taste intensities, construct normalized 5-dimensional recipe taste profiles, analyze cross-cuisine variation using non-parametric Kruskal–Wallis tests with Holm–Bonferroni correction, and evaluate lightweight machine learning classifiers (`LogisticRegression`, `RandomForestClassifier`) against a `DummyClassifier` baseline. Furthermore, we explicitly demonstrate and mitigate circularity in rule-derived taste targets via a masked-anchor contextual evaluation protocol and unsupervised cosine-similarity recipe retrieval.

## 2. Introduction
Computational Gastronomy applies complex systems, statistics, and machine learning to culinary data to uncover the principles governing food preparation and sensory perception. While recipes are culturally transmitted as unstructured ingredient lists, the sensory experience of a dish emerges from the aggregate chemical tastants and aroma compounds contributed by those ingredients. This project addresses three research questions:
1. How accurately can raw recipe ingredient strings be standardized and linked to validated gustatory taste dimensions?
2. Do world cuisines exhibit statistically significant differences in their computationally derived taste profiles and taste balance?
3. Can lightweight classifiers predict a recipe's dominant taste from contextual ingredient co-occurrence even when primary taste-anchor ingredients (such as salt, sugar, or lemon juice) are masked?

## 3. Related Work
- **Flavor Networks & Food Pairing:** Ahn et al. (2011) constructed a bipartite network of 381 ingredients and 1,021 volatile flavor compounds, finding that Western cuisines tend to pair ingredients sharing flavor compounds whereas East Asian cuisines avoid compound overlap.
- **Regional Indian Cuisines:** Jain et al. (2015) analyzed 2,543 recipes across eight regional Indian cuisines, showing strong negative food pairing driven by spice categories (particularly cayenne, green bell pepper, ginger, and garam masala components).
- **FlavorDB Molecular Repository:** Garg et al. (2018) unified 2,254 flavor molecules across 936 natural ingredients in FlavorDB (`https://doi.org/10.1093/nar/gkx957`), linking chemical descriptors to receptor resources such as BitterDB and SweetenersDB.
- **Global Culinary Patterns:** Singh and Bagler (2018) surveyed data-driven investigations across traditional world recipes (`arXiv:1803.04343`), highlighting the need for standardized ingredient preprocessing to overcome spelling and synonym noise.

## 4. Dataset Description
The pipeline processes recipe records containing `id`, `cuisine`, and raw `ingredients` lists (compatible with the Kaggle/Yummly *What's Cooking* `train.json` schema and provided with an offline reproducible multi-cuisine corpus). Ingredients are matched against `data/raw/ingredient_taste_lexicon.csv`, which maps 65 core culinary ingredients across 5 gustatory dimensions (`sweet`, `sour`, `bitter`, `salty`, `umami`) alongside FlavorDB identifiers and primary volatile molecules. Exact corpus counts, token coverage percentages, and unmatched ingredient logs are generated dynamically in `reports/dataset_summary.md`.

## 5. Methodology
### 5.1 Deliverable 1: Preprocessing & Normalization
Raw strings are lowercased, stripped of parenthetical volume notes and non-informative preparation participles (`chopped`, `minced`, `diced`), and mapped via a deterministic alias table (`INGREDIENT_ALIASES`) that preserves culinary distinctions (`garlic` vs. `black garlic`; `ginger` vs. `dried ginger`). Recipes with fewer than 2 matched ingredients or less than 25% match coverage are flagged (`sufficient_for_analysis = False`).

### 5.2 Deliverable 2: Recipe Taste Vector & Entropy Balance
For each recipe $r$, matched ingredient vectors $\mathbf{t}(i) \in [0,1]^5$ are averaged and L1-normalized onto the 5-taste simplex $\mathbf{p}(r)$. Taste diversity is quantified via Shannon entropy $H(r) = -\sum_{d} p_d(r)\ln p_d(r)$ and normalized by $\ln 5$ to yield Pielou's taste balance $J(r) \in [0, 1]$.

### 5.3 Deliverable 3: Cross-Cuisine Statistical Testing
Because simplex proportions are bounded and skewed, cross-cuisine differences across each taste dimension are tested via the non-parametric Kruskal–Wallis $H$-test alongside one-way ANOVA $F$-statistics. Raw $p$-values are adjusted using the Holm–Bonferroni step-down procedure ($\alpha = 0.05$). Cuisine distinctiveness is measured via standardized $z$-score lift $z_{c,d} = (\bar{p}_{c,d} - \mu_d)/\sigma_d$.

### 5.4 Deliverable 4: Lightweight Prediction & Circularity Control
Deduplicated recipes are partitioned via a 75%/25% stratified split (`random_state=42`). We compare `DummyClassifier`, L2-regularized `LogisticRegression`, and `RandomForestClassifier` under:
1. `Full_Ingredients` (evaluating recovery of the aggregation rule), and
2. `Masked_Anchors` (removing 12 primary sweet/sour/salty/umami/bitter anchor ingredients to evaluate contextual co-occurrence prediction).
An unsupervised cosine-similarity `NearestNeighbors` retriever is also implemented.

## 6. Experimental Setup
All experiments run in Python 3.11+ with fixed random seeds (`RANDOM_SEED = 42`). Executing `python run_pipeline.py` regenerates all processed CSV tables, statistical summaries, and 300-DPI visualizations in `results/`.

## 7. Results and Visualizations
Upon running `python run_pipeline.py`, exact numerical tables and figures are written to:
- `reports/dataset_summary.md` (Corpus size, unique ingredients, token match rate, unmatched ingredient audit).
- `results/taste_profile_distributions.png` & `results/example_recipe_profiles.png` (Simplex distributions and stacked bar recipe profiles).
- `results/cuisine_taste_heatmap.png`, `results/cuisine_taste_boxplots.png`, `results/cuisine_distinctive_tastes.png`, and `results/cuisine_pca_projection.png` (Cross-cuisine statistical comparisons, $z$-score lifts, and PCA clustering).
- `results/model_comparison.csv`, `results/confusion_matrix.png`, and `results/feature_importance.png` (Classifier accuracy, macro-F1, macro-precision, macro-recall, and top Random Forest features).

## 8. Discussion
The separation of volatile aroma compounds from gustatory taste scores prevents false inferences (such as assuming an ingredient rich in volatile terpenes is sweet or sour). Furthermore, contrasting `Full_Ingredients` against `Masked_Anchors` in Deliverable 4 illustrates a core lesson in data science: high accuracy on rule-derived targets does not imply external empirical validity, yet non-trivial performance when anchor ingredients are masked confirms that culinary traditions assemble co-occurring companion ingredients around consistent taste themes.

## 9. Limitations
1. **Absence of Ingredient Quantities:** Public recipe datasets list ingredients without mass or volume weights; unweighted averaging over-represents trace spices relative to bulk staples.
2. **Cooking Physics & Chemistry Omitted:** Thermal reactions (Maillard browning, caramelization) and acid/fat suppression alter perceived taste intensity in ways static ingredient lookup tables cannot capture.
3. **Lexicon Coverage:** Unmatched long-tail ingredients are logged in `ingredient_flavor_mapping.csv` and excluded from taste aggregation.
4. **Rule-Derived Target Circularity:** No human sensory panel ratings exist at the recipe level; `dominant_taste` is an ingredient-derived proxy.

## 10. Conclusion
This project delivers a lightweight, reproducible, and scientifically transparent computational gastronomy framework that connects recipe ingredient composition to interpretable taste representations, cross-cuisine statistical insights, and honest predictive modeling.

## 11. References
1. Garg, N., et al. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
2. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
3. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
4. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343