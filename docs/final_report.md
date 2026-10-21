# Computational Analysis and Prediction of Taste Profiles in Recipes

**Authors:** Undergraduate Computational Gastronomy Research Project  
**Affiliation:** Department of Computer Science & Center for Computational Biology  
**Date:** October 2026  
**Repository:** [https://github.com/gopaltiwari04/Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes](https://github.com/gopaltiwari04/Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes)

---

## 1. Abstract
Computational Gastronomy models culinary patterns and human sensory perception through data science, complex networks, and machine learning. While molecular databases document volatile aroma compounds across raw ingredients, connecting recipe composition directly to non-volatile gustatory taste profiles (`sweet`, `sour`, `bitter`, `salty`, `umami`) has remained challenging due to unstructured text, lack of mass quantities, and circularity in rule-derived targets. 

In this work, we present an end-to-end reproducible scientific framework evaluated on **38,985 cleaned recipes across 20 world cuisines** from the Kaggle / Yummly dataset. We:
1. Standardize recipe ingredients and map them to a 92-item curated taste proxy lexicon anchored against standard tastant reference solutions, achieving a **61.1% token coverage rate** across 416,203 ingredient tokens while explicitly isolating volatile aroma molecules and trigeminal chemesthetic sensations (e.g., capsaicin heat).
2. Formulate 5-dimensional probability simplex representations ($\sum p_d = 1$) and quantify taste distributional evenness via normalized Shannon entropy ($J \in [0, 1]$, mean $J = 0.8529$).
3. Conduct non-parametric omnibus hypothesis tests across all 20 cuisines using Kruskal–Wallis tests with Holm–Bonferroni family-wise multiple-comparison corrections ($df = 19$, $p_{adj} < 10^{-300}$) and confirm significance under Aitchison's Centered Log-Ratio (clr) transformation.
4. Benchmark leak-free classification models (`LogisticRegression`, `RandomForestClassifier`, `DummyClassifier`), demonstrating that while full-ingredient models recover the algebraic proxy rule with 94.8% accuracy and 0.940 macro-F1, masking 20 direct taste anchors drops macro-F1 to **0.631**, proving that companion ingredients co-occur in culturally structured patterns that predict dish taste balance even when primary tastants are hidden.
5. Provide a production-grade inference API, cosine-similarity retrieval, and a 34-test automated test suite.

---

## 2. Introduction & Research Questions
Traditional culinary science relies on qualitative cultural descriptions or expensive sensory panels. Recent computational approaches pioneered food-pairing networks (Ahn et al., 2011; Jain et al., 2015) based on volatile compound sharing. However, volatile aroma compounds primarily activate olfactory receptors in the nasal cavity via retronasal olfaction, whereas basic taste perception emerges from non-volatile tastants binding lingual taste buds.

To build a defensible bridge between recipe ingredients and taste perception, this study investigates four central research questions:
- **RQ1 (Data Engineering & Provenance):** Can unstructured ingredient lists from multi-cuisine corpora be normalized to resolve preparation variants without collapsing distinct ingredients, while separating trigeminal pungency from basic taste modalities?
- **RQ2 (Mathematical Modeling):** How can recipe taste profiles be represented on a 5-dimensional simplex, and how does normalized Shannon entropy capture taste distributional evenness?
- **RQ3 (Cross-Cuisine Hypothesis Testing):** Do world cuisines exhibit statistically significant differences in taste distributions under non-parametric omnibus tests (Kruskal–Wallis) with family-wise multiple-testing adjustments (Holm–Bonferroni) and compositional sensitivity checks?
- **RQ4 (Predictive Modeling & Circularity Control):** How well can lightweight classifiers predict a recipe's dominant taste when direct taste-anchor ingredients are masked, isolating contextual co-occurrence from circular algebraic recovery?

---

## 3. Related Work & Theoretical Foundations
1. **Flavor Networks & Compound Sharing:** Ahn et al. (2011) introduced the flavor network of 381 ingredients and 1,021 volatile compounds, demonstrating that Western recipes tend to pair ingredients sharing volatile molecules, whereas East Asian cuisines avoid compound sharing.
2. **Indian Cuisine & Negative Food Pairing:** Jain et al. (2015) analyzed 2,543 traditional Indian recipes, showing that spices (cayenne, turmeric, coriander, cumin) drive pronounced negative food pairing.
3. **Molecular Repositories (FlavorDB):** Garg et al. (2018) curated 2,254 flavor molecules across 936 ingredients in FlavorDB (`https://doi.org/10.1093/nar/gkx957`). While FlavorDB catalogs sensory taste tags, it does not provide continuous 0–1 taste intensities for dishes.
4. **Global Culinary Surveys:** Singh & Bagler (2018) surveyed computational gastronomy, emphasizing the critical role of standardized ingredient normalization in preventing noise.
5. **Compositional Data Geometry:** Aitchison (1986) demonstrated that simplex vectors ($\sum p_d = 1$) reside in restricted Euclidean space with negative covariance, necessitating centered log-ratio (clr) transformations to confirm marginal univariate tests.
6. **Non-Parametric Effect Size:** Tomczak & Tomczak (2014) established $\eta_H^2 = \frac{H - k + 1}{N - k}$ as the standard effect size for Kruskal–Wallis tests.

---

## 4. Dataset Description, Provenance, & Licensing
Experiments were conducted on the official Kaggle / Yummly *What's Cooking* dataset (`data/raw/train.json`):
- **Source:** Kaggle Competition / Yummly Inc. (`https://www.kaggle.com/c/whats-cooking/data`)
- **License / Terms:** Provided by Yummly Inc. and Kaggle for research and educational purposes.
- **Raw Records:** 39,774 recipe objects with `id`, `cuisine`, and `ingredients`.
- **Cleaning & Filtering:** Missing cuisines: 0. Empty ingredient lists: 0. Invalid records: 0.
- **Deduplication:** Canonical normalization identified 789 duplicate recipes sharing identical normalized ingredient bags within the same cuisine. Legitimate identical ingredient sets across different cuisines (14 instances) were preserved.
- **Clean Unique Recipes:** **38,985 recipes** across **20 cuisines**.
- **Lexicon Matching:** 91 matched canonical ingredients covered **254,309 tokens** out of 416,203 total occurrences (**61.1% token match rate**).
- **Sufficient Recipes:** **37,033 recipes** (95.0%) met the sufficiency threshold of $\ge 2$ matched ingredients and $\ge 25\%$ match coverage ratio.

A complete machine-readable audit trail is stored in `data/dataset_manifest.json`.

---

## 5. Taste Lexicon Curation & Sensory Methodology
The reference lexicon (`data/raw/ingredient_taste_lexicon.csv`) contains 92 curated ingredients evaluated across:
- The 5 basic gustatory modalities (`sweet`, `sour`, `bitter`, `salty`, `umami` $\in [0, 1]$), calibrated against aqueous sensory standards (sucrose for sweet, citric acid for sour, caffeine/quinine for bitter, NaCl for salty, monosodium glutamate for umami).
- **Chemesthetic Pungency:** Somatosensory sensations mediated by trigeminal nerve receptors (TRPV1, TRPA1) are isolated under `chemesthetic_pungency`:
  - `capsaicin_heat`: cayenne pepper, jalapeno, green chili
  - `piperine_pungency`: black pepper
  - `gingerol_pungency`: fresh ginger, dried ginger
  - `allicin_pungency`: garlic, shallot
  - `isothiocyanate_pungency`: mustard
- **Confidence Level:** Annotated as `high` (pure single-modality tastants), `medium` (fermented/aged products), or `low_proxy` (complex starches/lipids).
- **Retronasal Aroma Compounds:** Volatile molecules from FlavorDB (e.g. *allicin*, *linalool*, *cuminaldehyde*) are explicitly documented under `primary_aroma_compounds` and separated from basic taste intensities.

---

## 6. Mathematical Formulation

### 6.1 Recipe Taste Vector Aggregation
For recipe $r$ with matched ingredient set $M_r$ ($|M_r| \ge 2$), each ingredient $i \in M_r$ maps to gustatory proxy $\mathbf{t}(i) \in [0, 1]^5$ over basis $\mathcal{D} = \{\text{sweet}, \text{sour}, \text{bitter}, \text{salty}, \text{umami}\}$.
In unweighted presence mode:
$$\mathbf{u}_d(r) = \frac{1}{|M_r|} \sum_{i \in M_r} \mathbf{t}_d(i)$$
*(Optional mass-weighted formulation $\mathbf{u}_d(r) = \frac{\sum w_i \mathbf{t}_d(i)}{\sum w_i}$ is implemented for corpora with gram weights).*

### 6.2 Simplex Normalization
The vector is projected onto the 4-simplex:
$$\mathbf{p}_d(r) = \frac{\mathbf{u}_d(r)}{\sum_{d' \in \mathcal{D}} \mathbf{u}_{d'}(r)}, \quad \sum_{d \in \mathcal{D}} \mathbf{p}_d(r) = 1$$

### 6.3 Distributional Evenness (Shannon Entropy)
Taste diversity is calculated in nats:
$$H(r) = -\sum_{d \in \mathcal{D}: \mathbf{p}_d(r) > 0} \mathbf{p}_d(r) \ln \mathbf{p}_d(r)$$
Pielou's evenness normalizes $H(r)$ by maximum entropy $\ln(5) \approx 1.6094$:
$$J(r) = \frac{H(r)}{\ln(5)} \in [0, 1]$$
- $J = 1.0$: Uniform distribution across all 5 tastes (maximum dispersion).
- $J = 0.0$: Complete dominance by a single taste modality.

---

## 7. Cross-Cuisine Statistical Analysis

### 7.1 Descriptive Summary Across 20 Cuisines
Table 1 displays sample sizes, mean normalized shares, distinctive taste modality ($z$-score lift), and mean evenness ($J$) across all 20 cuisines ($N = 37,033$ recipes):

| Cuisine | $N$ Recipes | Sweet | Sour | Bitter | Salty | Umami | Distinctive Modality | Max $z$-Lift | Mean Evenness ($J$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Brazilian** | 352 | 0.2839 | 0.1565 | 0.1690 | 0.1879 | 0.2027 | Sour | +0.2763 | 0.8189 |
| **British** | 716 | 0.3511 | 0.0844 | 0.1162 | 0.2603 | 0.1880 | Sweet | +0.6375 | 0.7739 |
| **Cajun/Creole**| 1,451 | 0.2483 | 0.1041 | 0.1986 | 0.1825 | 0.2664 | Bitter | +0.2565 | 0.8765 |
| **Chinese** | 2,571 | 0.2537 | 0.0914 | 0.1082 | 0.2358 | 0.3109 | Umami | +0.6218 | 0.8692 |
| **Filipino** | 720 | 0.2770 | 0.0944 | 0.1167 | 0.2414 | 0.2705 | Umami | +0.2856 | 0.8321 |
| **French** | 2,422 | 0.3046 | 0.1226 | 0.1611 | 0.2160 | 0.1958 | Sweet | +0.3459 | 0.8041 |
| **Greek** | 1,122 | 0.1944 | 0.1929 | 0.2193 | 0.1965 | 0.1969 | Sour | +0.6149 | 0.8934 |
| **Indian** | 2,893 | 0.2436 | 0.1433 | 0.2277 | 0.2041 | 0.1812 | Bitter | +0.5303 | 0.8769 |
| **Irish** | 613 | 0.3224 | 0.0766 | 0.1422 | 0.2761 | 0.1828 | Sweet | +0.4575 | 0.7858 |
| **Italian** | 7,292 | 0.1986 | 0.1139 | 0.2029 | 0.2271 | 0.2576 | Bitter | +0.2967 | 0.8550 |
| **Jamaican** | 487 | 0.2806 | 0.1168 | 0.1889 | 0.2064 | 0.2073 | Sweet | +0.1956 | 0.8697 |
| **Japanese** | 1,280 | 0.2698 | 0.1115 | 0.1118 | 0.2398 | 0.2672 | Umami | +0.2581 | 0.8380 |
| **Korean** | 796 | 0.2807 | 0.0876 | 0.1332 | 0.2027 | 0.2959 | Umami | +0.4967 | 0.8785 |
| **Mexican** | 5,996 | 0.2037 | 0.1721 | 0.1838 | 0.1868 | 0.2536 | Sour | +0.4211 | 0.8862 |
| **Moroccan** | 783 | 0.2550 | 0.1408 | 0.2616 | 0.1638 | 0.1787 | Bitter | +0.8487 | 0.8851 |
| **Russian** | 471 | 0.3293 | 0.1239 | 0.1213 | 0.2284 | 0.1972 | Sweet | +0.5005 | 0.8098 |
| **Southern US**| 3,897 | 0.3249 | 0.0980 | 0.1353 | 0.2585 | 0.1833 | Sweet | +0.4734 | 0.7851 |
| **Spanish** | 918 | 0.2422 | 0.1419 | 0.2016 | 0.1923 | 0.2219 | Bitter | +0.2851 | 0.8609 |
| **Thai** | 1,478 | 0.2600 | 0.1586 | 0.1213 | 0.1931 | 0.2670 | Sour | +0.2952 | 0.8907 |
| **Vietnamese** | 775 | 0.2719 | 0.1386 | 0.1195 | 0.2056 | 0.2643 | Umami | +0.2343 | 0.8861 |

### 7.2 Omnibus Hypothesis Tests & Multiple-Testing Correction
Non-parametric Kruskal–Wallis tests ($df = 19$, $N = 37,033$) with Holm–Bonferroni step-down adjustment:

| Modality | Kruskal–Wallis $H$ | Raw $p$-value | Holm–Bonferroni $p_{adj}$ | Effect Size ($\eta_H^2$) | Effect Magnitude |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sweet** | 3,839.71 | $< 10^{-300}$ | $< 10^{-300}$ | 0.1032 | Medium |
| **Sour** | 3,931.58 | $< 10^{-300}$ | $< 10^{-300}$ | 0.1057 | Medium |
| **Bitter** | 6,443.41 | $< 10^{-300}$ | $< 10^{-300}$ | 0.1736 | Large |
| **Salty** | 1,485.28 | $6.39 \times 10^{-304}$ | $6.39 \times 10^{-304}$ | 0.0396 | Small |
| **Umami** | 4,766.65 | $< 10^{-300}$ | $< 10^{-300}$ | 0.1283 | Medium |
| **Taste Balance ($J$)** | 3,101.58 | $< 10^{-300}$ | $< 10^{-300}$ | 0.0833 | Medium |

### 7.3 Compositional Centered Log-Ratio (clr) Sensitivity Check
Under Aitchison's clr transformation, every modality retained statistical significance ($p_{adj} < 10^{-300}$, Kruskal $H$ ranging from 2,215 to 5,405), proving that observed cultural variations are robust and not artifacts of simplex closure.

---

## 8. Machine Learning Benchmarks & Circularity Analysis

### 8.1 Leak-Free Experimental Protocol
- **Partitioning:** Stratified 75% train ($N = 27,738$) and 25% test ($N = 9,247$), fixed seed 42.
- **Leak-Free Preprocessing:** `CountVectorizer` was fit exclusively on the training partition after removing duplicate recipe signatures.
- **Setting 1 (Full Ingredients):** Measures recovery of the algebraic proxy aggregation rule.
- **Setting 2 (Masked Anchors):** Masks 20 direct taste anchors (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.). 83.2% of recipes retained $\ge 2$ informative ingredients.

### 8.2 Model Comparison Results

| Evaluation Setting | Model | 5-Fold CV Macro-F1 | Test Accuracy | Test Macro-F1 | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Setting 1 (Full Ingredients)** | DummyClassifier (Most Frequent) | 0.0987 $\pm$ 0.000 | 0.3277 | 0.0987 | 0.0655 | 0.2000 |
| **Setting 1 (Full Ingredients)** | **LogisticRegression (L2, Balanced)** | **0.9385 $\pm$ 0.003** | **0.9477** | **0.9400** | **0.9318** | **0.9498** |
| **Setting 1 (Full Ingredients)** | RandomForest (150 trees, Balanced) | 0.7282 $\pm$ 0.009 | 0.7496 | 0.7292 | 0.7203 | 0.7633 |
| **Setting 2 (Masked Anchors)** | DummyClassifier (Most Frequent) | 0.0988 $\pm$ 0.000 | 0.3280 | 0.0988 | 0.0656 | 0.2000 |
| **Setting 2 (Masked Anchors)** | **LogisticRegression (L2, Balanced)** | **0.6280 $\pm$ 0.001** | **0.6583** | **0.6310** | **0.6223** | **0.6516** |
| **Setting 2 (Masked Anchors)** | RandomForest (150 trees, Balanced) | 0.5724 $\pm$ 0.007 | 0.5964 | 0.5650 | 0.5655 | 0.6007 |

### 8.3 Key Insights from Predictive Modeling
1. **Rule Recovery vs. Co-Occurrence Generalization:** Logistic Regression in Setting 1 achieves 94.8% accuracy, verifying that linear weights recover the additive scoring rule. In Setting 2, masking the 20 primary taste markers drops macro-F1 to **0.631**, substantially outperforming the Dummy baseline (0.099). This confirms that culinary traditions assemble companion ingredients (e.g., alliums, starches, aromatic spices) into predictable contextual motifs.
2. **Feature Importances:** Random Forest impurity analysis confirms that top features in Setting 1 correspond directly to anchor ingredients (`sugar`, `salt`, `lemon juice`, `soy sauce`, `cocoa powder`).

---

## 9. Threats to Validity & Limitations
1. **Presence vs. Mass Concentration:** Public recipe datasets lack measured mass or volume weights; unweighted presence attributes identical weight to pinch spices and bulk starches.
2. **Thermal & Chemical Reactions Omitted:** Cooking processes (caramelization, Maillard browning, acid hydrolysis, salting, deglazing) dynamically alter tastants in ways static ingredient lookup tables cannot model.
3. **Absence of Sensory Ground Truth:** No public dataset provides recipe-level human sensory panel scores. The target `dominant_taste` is an ingredient-based proxy.
4. **Intra-Cuisine Heterogeneity:** Regional sub-cuisines and individual dish types (desserts vs. curries) vary broadly around national centroids.

---

## 10. Reproducibility & Engineering Quality
- **Automated Testing:** 34 comprehensive pytest unit and integration tests passing in ~12 seconds.
- **Master Pipeline CLI:** Single command reproduces all deliverables from clean checkout:
  ```bash
  python run_pipeline.py
  ```
- **Synthetic Smoke Testing:**
  ```bash
  python run_pipeline.py --use-sample-data
  ```
- **Interactive Inference API:**
  ```bash
  python run_pipeline.py --predict "tomatoes, garlic, olive oil, fresh basil"
  ```

---

## 11. Conclusion
This study delivers a research-grade, scientifically defensible computational gastronomy framework that maps 38,985 recipes to interpretable 5D taste simplex profiles, reveals statistically significant cross-cuisine taste differences under rigorous non-parametric and compositional tests, and demonstrates leak-free contextual taste prediction while transparently addressing target circularity.

---

## 12. References
1. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
2. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall, London.
3. Garg, N., Sethupathy, A., Tuwani, R., NK, R., Dokania, S., Iyer, A., Gupta, A., Agrawal, S., Singh, N., Shukla, S., Kathuria, K., Badhwar, R., Kanji, R., Jain, A., Kaur, A., Nagpal, R., & Bagler, G. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
4. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
5. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343
6. Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. An overview of some recommended measures of effect size. *Trends in Sport Sciences*, 1(21), 19–25.
