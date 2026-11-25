# Computational Analysis and Prediction of Taste Profiles in Recipes

**Authors:** Gopal Krishna Tewari (B.Tech CSAM, Roll No. 2023223), Nitin Kumar (B.Tech CSB, Roll No. 2023358)  
**Affiliation:** Department of Computer Science & Center for Computational Biology  
**Repository:** [https://github.com/gopaltiwari04/Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes](https://github.com/gopaltiwari04/Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes)  
**Version:** 1.1.0 | **Date:** November 2026  

---

## 1. Abstract
Computational Gastronomy bridges culinary texts, food chemistry, and statistical machine learning. While molecular repositories such as FlavorDB catalog volatile aroma compounds across natural ingredients, linking unstructured recipe lists directly to non-volatile basic gustatory modalities (`sweet`, `sour`, `bitter`, `salty`, `umami`) requires rigorous separation of taste, aroma, and trigeminal chemesthesis, alongside strict controls for compositional closure and rule-derived label circularity.

Evaluating the verified 4-deliverable pipeline on **39,774 raw recipes across 20 world cuisines** from the Kaggle / Yummly *What's Cooking* corpus (`38,979` clean unique recipes; `37,447` meeting sufficiency criteria), we find:
1. A curated 122-entry taste proxy lexicon (`data/raw/ingredient_taste_lexicon.csv`) cross-verified against FlavorDB entity IDs achieves **64.05% token coverage** (`266,841 / 416,607` ingredient occurrences) while explicitly preserving `NaN` for refined oils and unestablished sub-dimensions and tracking chemesthetic pungency separately.
2. Projected onto the 4-simplex, recipes exhibit a corpus mean Shannon entropy of **$H = 1.3892$ nats** and Pielou taste-share evenness **$J = 0.8631$** (`SD = 0.1414`).
3. Non-parametric Kruskal–Wallis tests with Holm–Bonferroni correction detect statistically significant cross-cuisine shifts across all five modalities ($\eta_H^2$ ranging from `0.0429` for `salty` to `0.1611` for `bitter`, $p_{\text{adj}} < 10^{-300}$), and joint multivariate Aitchison CLR PERMANOVA confirms compositional differentiation (Pseudo-$F = 79.25$, $R^2_{\text{Aitchison}} = 3.87\%$, $p = 0.0033$), demonstrating simultaneously that cultural shifts are real yet within-cuisine recipe diversity accounts for over 96% of compositional variance.
4. In leak-free supervised classification (`LogisticRegression`), full-ingredient features recover the algebraic proxy label at **94.11% accuracy / 0.9345 macro-F1** (95% CI `[0.9290, 0.9397]`). Masking 43 canonical taste anchors plus anchor-family variants and removing 182 post-masking train/test collisions drops performance to **60.82% accuracy / 0.5942 macro-F1** (95% CI `[0.5830, 0.6038]`), still far exceeding the majority-class `DummyClassifier` (`0.1006` macro-F1), quantifying genuine companion-ingredient co-occurrence structure without conflating proxy recovery with human sensory validation.

---

## 2. Research Questions & Objectives
- **RQ1 (Normalization & Provenance):** Can unstructured ingredient strings be standardized without collapsing distinct culinary ingredients (`garlic` vs. `black garlic` vs. `garlic powder`; `beef` vs. `beef broth`; `lemon` vs. `lemon juice`) while validating FlavorDB metadata and logging unmatched items?
- **RQ2 (Simplex Modeling):** How do unweighted presence proxies map recipes onto the 5D probability simplex when missing (`NaN`) dimension scores are excluded rather than imputed as zero?
- **RQ3 (Cross-Cuisine Analysis):** How strongly do world cuisines differ in modeled taste shares under both univariate rank tests and joint compositional Aitchison PERMANOVA?
- **RQ4 (Predictive Circularity Control):** How much predictive signal remains for the proxy `dominant_taste` label once primary taste anchors and surface-alias variants are systematically masked?

---

## 3. Related Work
1. **Flavor Pairing Networks:** Ahn et al. (2011) analyzed 381 ingredients and 1,021 volatile flavor molecules, identifying contrasting compound-sharing patterns between Western and East Asian cuisines.
2. **Indian Culinary Spices:** Jain et al. (2015) studied 2,543 Indian recipes, finding negative volatile compound pairing driven by spices such as cayenne, turmeric, cumin, and coriander.
3. **Molecular Flavor Repositories:** Garg et al. (2018) released FlavorDB (`doi:10.1093/nar/gkx957`), indexing natural ingredients (`entity_id`), taxonomic sources, and volatile molecules. Crucially, FlavorDB molecules document volatile aroma profiles rather than recipe-level 0–1 basic-taste intensities.
4. **Computational Gastronomy Surveys:** Singh & Bagler (2018) highlighted ingredient standardization as a foundational bottleneck in culinary data mining.
5. **Compositional Data Analysis & Effect Sizes:** Aitchison (1986) established log-ratio transformations (CLR) for simplex data ($\sum p_d = 1$); Martín-Fernández et al. (2003) formalized multiplicative zero replacement; Tomczak & Tomczak (2014) formalized Kruskal–Wallis $\eta_H^2$.

---

## 4. Dataset Provenance & Licensing
- **Dataset:** Kaggle / Yummly *What's Cooking* (`data/raw/train.json`, SHA-256 `f18e7e568d1c606774bfb134ece51ef0a143cd5069e6f9241bb6036b45b7d2b0`)
- **Source URL:** https://www.kaggle.com/c/whats-cooking/data
- **License / Terms:** Unknown / Kaggle Playground Competition Rules (provided by Yummly Inc. for non-commercial educational/research use; raw JSON is gitignored and not redistributed).
- **Synthetic Smoke Corpus:** Available strictly via explicit `--use-sample-data` (`140` cleaned synthetic recipes across `6` template cuisines) and explicitly labeled `SYNTHETIC` in all outputs. Missing real data raises `FileNotFoundError` rather than falling back silently.

---

## 5. Dataset Statistics (`data/dataset_manifest.json`)
| Metric | Real Kaggle Corpus | Synthetic Smoke Test (`--use-sample-data`) |
| :--- | :---: | :---: |
| Initial Raw Records | 39,774 | 184 |
| Invalid Records Removed | 0 | 2 |
| Duplicate IDs / Within-Cuisine Duplicates Removed | 0 / 795 | 1 / 41 |
| Cross-Cuisine Identical Sets Preserved | 55 | 0 |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **38,979** | **140** |
| Cuisines Represented | 20 | 6 |
| Unique Normalized Ingredients | 6,195 | 76 |
| Lexicon Matched (`matched` / `taste_unknown` / `unmatched`) | 120 / 2 / 6,073 | 70 / 2 / 4 |
| Token Coverage (`matched_ingredient_tokens / total`) | 64.05% (266,841 / 416,607) | 98.40% (1,045 / 1,062) |
| **Sufficient Recipes ($M_r \ge 2$, coverage $\ge 25\%$)** | **37,447 (96.07%)** | **140 (100.00%)** |

---

## 6. Taste-Annotation Methodology (`data/raw/ingredient_taste_lexicon.csv`)
The 122-entry lexicon links normalized ingredients to:
- **Five Basic Gustatory Modalities (`[0, 1]` or `NaN`):** `sweet`, `sour`, `bitter`, `salty`, `umami`, explicitly distinguished by `annotation_status` (`reference_standard` for pure sucrose, NaCl, citric/acetic acid solutions, starch, water; `curated_proxy` for literature-informed culinary proxies; `taste_unknown` with `NaN` for refined cooking/peanut oils and unestablished sub-dimensions).
- **FlavorDB Cross-Verification:** All numeric `flavordb_entity_id` values were verified against FlavorDB (`entity_id`, e.g., Tomato=`364`, Garlic=`259`, Ginger=`333`, Lemon=`240`, Lime=`241`, Vinegar=`322`, Soybean Sauce=`291`). Entries absent from FlavorDB carry `NONE`.
- **Aroma vs. Tastants vs. Chemesthesis:** Volatile FlavorDB molecules (`primary_aroma_compounds`, e.g. *Diallyl Disulfide*, *Citral*, *2,3,5,6-Tetramethylpyrazine*) are separated from non-volatile tastants (`known_tastant_or_pungent_compounds`) and trigeminal TRP sensations (`chemesthetic_pungency`: `capsaicin_heat`, `piperine_pungency`, `gingerol_pungency`, `allicin_pungency`, `isothiocyanate_pungency`). Chili heat is never encoded as a sixth basic taste.

---

## 7. Mathematical Formulation (`src/taste_profile.py`)
For recipe $r$ with taste-matched ingredient set $M_r$, let $M_{r,d} = \{i \in M_r : t_d(i) \text{ is non-NaN}\}$:
1. **Unweighted Presence Mean:** $u_d(r) = \frac{1}{|M_{r,d}|}\sum_{i \in M_{r,d}} t_d(i)$ (or $\text{NaN}$ if $|M_{r,d}| = 0$). Quantity weighting ($\sum w_i t_d(i)/\sum w_i$) is implemented for corpora providing measured weights, without inventing weights otherwise.
2. **4-Simplex Projection:** $p_d(r) = u_d(r) / \sum_{d': u_{d'}\text{ known}} u_{d'}(r)$ (setting unobserved $\text{NaN}$ shares to $0$).
3. **Shannon Entropy & Evenness:** $H(r) = -\sum_{p_d>0} p_d(r)\ln p_d(r)$ [nats], and Pielou's evenness $J(r) = H(r)/\ln(5) \in [0, 1]$, measuring distributional uniformity of modeled shares rather than human culinary balance.

---

## 8. Experimental Setup & Statistical Methodology (`src/cuisine_analysis.py`)
All real-data experiments ran with `RANDOM_SEED = 42` in Python 3.11.9 (`scipy 1.17.1`, `scikit-learn 1.9.1`).
- **Univariate Non-Parametric Tests:** Kruskal–Wallis $H$-tests ($H_0$: identical distributions across $k=20$ cuisines, $df=19$, $\alpha=0.05$) adjusted via Holm–Bonferroni step-down correction, reporting $\eta_H^2$ with 95% bootstrap CIs.
- **Compositional Closure Control:** Multiplicative zero replacement ($\epsilon=10^{-4}$; Martín-Fernández et al., 2003) followed by Centered Log-Ratio (CLR) transformation (Aitchison, 1986), evaluated with both univariate CLR Kruskal–Wallis tests and **joint one-way CLR PERMANOVA** (299 permutations).
- **Centroid PCA Biplot:** `StandardScaler` + 2-component PCA on the $20 \times 5$ matrix of cuisine means (`results/cuisine_pca_projection.png`).

---

## 9. ML Methodology (`src/taste_prediction.py`)
- **Pre-Split Deduplication & Leak-Free Pipelines:** Exact normalized bags deduplicated prior to stratified 75%/25% splitting (`N_train=28,044`, `N_test=9,348`). `CountVectorizer` (`binary=True`, `min_df=2`) is nested inside each `sklearn.pipeline.Pipeline` during 5-fold Stratified CV so validation folds never leak into vocabulary pruning.
- **Target Interpretation & Anchor Masking:**
  - *Setting 1 (`Full_Ingredients`):* Measures recovery of the algebraic rule generating `dominant_taste`.
  - *Setting 2 (`Masked_Anchors`):* Strips 43 canonical anchors ($\ge 0.75$ intensity + direct tastants) and regex-matched anchor-family variants (`ANCHOR_WORD_REGEX`), retains 99.74% of recipes (`37,295 / 37,392`), and removes `182` post-masking duplicate collisions spanning train/test (`N_train=27,971`, `N_test=9,142`).
  - *Empirical Sensory Validation:* Absent because public recipes lack human sensory ratings.
- **Unsupervised Retrieval:** Cosine nearest neighbors in 5D simplex space (`retrieve_similar_recipes`), guaranteed never to return the query ID.

---

## 10. Experimental Results (Real Kaggle / Yummly Corpus)

### 10.1 Cross-Cuisine Descriptive Summary ($N = 37,447$)
| Cuisine | $N$ | Sweet | Sour | Bitter | Salty | Umami | Elevated Modality ($z$-Lift) | Mean Evenness ($J$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| brazilian | 361 | 0.2718 | 0.1698 | 0.1652 | 0.1930 | 0.2002 | sweet (+0.2540) | 0.8287 |
| british | 735 | 0.3119 | 0.1119 | 0.1216 | 0.2705 | 0.1841 | sweet (+0.5177) | 0.7847 |
| cajun_creole | 1,467 | 0.2343 | 0.1220 | 0.1847 | 0.1928 | 0.2661 | bitter (+0.2578) | 0.8919 |
| chinese | 2,579 | 0.2403 | 0.1057 | 0.1061 | 0.2494 | 0.2986 | umami (+0.5314) | 0.8773 |
| filipino | 714 | 0.2622 | 0.1025 | 0.1064 | 0.2558 | 0.2731 | umami (+0.3117) | 0.8369 |
| french | 2,454 | 0.2850 | 0.1422 | 0.1483 | 0.2221 | 0.2023 | sweet (+0.3410) | 0.8129 |
| greek | 1,136 | 0.1746 | 0.2230 | 0.1824 | 0.2198 | 0.2002 | sour (+0.6935) | 0.9056 |
| indian | 2,899 | 0.2283 | 0.1479 | 0.2403 | 0.2079 | 0.1757 | bitter (+0.8289) | 0.8761 |
| irish | 630 | 0.2775 | 0.1108 | 0.1507 | 0.2845 | 0.1764 | salty (+0.4106) | 0.8136 |
| italian | 7,417 | 0.1856 | 0.1334 | 0.1729 | 0.2386 | 0.2695 | umami (+0.2807) | 0.8632 |
| jamaican | 489 | 0.2594 | 0.1254 | 0.1988 | 0.2107 | 0.2058 | bitter (+0.4027) | 0.8743 |
| japanese | 1,313 | 0.2641 | 0.1309 | 0.1096 | 0.2390 | 0.2564 | sweet (+0.2030) | 0.8545 |
| korean | 798 | 0.2634 | 0.1040 | 0.1236 | 0.2286 | 0.2805 | umami (+0.3752) | 0.8884 |
| mexican | 6,035 | 0.1977 | 0.1843 | 0.1660 | 0.1965 | 0.2554 | sour (+0.3497) | 0.8907 |
| moroccan | 790 | 0.2515 | 0.1490 | 0.2470 | 0.1707 | 0.1818 | bitter (+0.8975) | 0.8881 |
| russian | 474 | 0.2945 | 0.1566 | 0.1194 | 0.2375 | 0.1921 | sweet (+0.4032) | 0.8315 |
| southern_us | 3,976 | 0.2889 | 0.1365 | 0.1293 | 0.2606 | 0.1846 | sweet (+0.3664) | 0.8143 |
| spanish | 928 | 0.2300 | 0.1590 | 0.1830 | 0.2044 | 0.2236 | bitter (+0.2406) | 0.8705 |
| thai | 1,477 | 0.2477 | 0.1658 | 0.1278 | 0.1958 | 0.2629 | umami (+0.2232) | 0.9015 |
| vietnamese | 775 | 0.2602 | 0.1446 | 0.1245 | 0.2117 | 0.2591 | umami (+0.1904) | 0.8938 |

### 10.2 Hypothesis Tests & Compositional PERMANOVA
- **Univariate Kruskal–Wallis ($df=19$, Holm-adjusted):** `bitter` ($H=6,048.29, \eta_H^2=0.1611$ [0.1530, 0.1670], large), `umami` ($H=5,039.13, \eta_H^2=0.1341$ [0.1285, 0.1390], medium), `sweet` ($H=3,457.64, \eta_H^2=0.0919$ [0.0870, 0.0985], medium), `sour` ($H=2,575.19, \eta_H^2=0.0683$ [0.0628, 0.0729], medium), `taste_evenness` ($H=2,439.79, \eta_H^2=0.0647$ [0.0594, 0.0701], medium), `salty` ($H=1,626.07, \eta_H^2=0.0429$ [0.0395, 0.0467], small); all Holm $p_{\text{adj}} < 10^{-300}$.
- **Joint Aitchison CLR PERMANOVA:** Pseudo-$F = 79.2489$, $R^2_{\text{Aitchison}} = 3.87\%$, $p = 0.003333$ (299 permutations); all 5 univariate CLR tests remain significant ($p_{\text{adj}} < 10^{-300}$).
- **Centroid PCA:** First 2 PCs capture `76.4%` (`48.9% + 27.5%`) of variance across the 20 standardized cuisine centroids.

### 10.3 Machine Learning Benchmark (`results/model_comparison.csv`)
| Evaluation Setting | Model | 5-Fold CV Macro-F1 | Test Accuracy [95% CI] | Test Macro-F1 [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **Setting 1 (`Full_Ingredients`)** | DummyClassifier | 0.0994 $\pm$ 0.0000 | 0.3308 [0.3228, 0.3401] | 0.0994 [0.0976, 0.1015] |
| **Setting 1 (`Full_Ingredients`)** | **LogisticRegression (L2)** | **0.9242 $\pm$ 0.0016** | **0.9411 [0.9365, 0.9453]** | **0.9345 [0.9290, 0.9397]** |
| **Setting 1 (`Full_Ingredients`)** | RandomForest (150 trees) | 0.7157 $\pm$ 0.0068 | 0.7441 [0.7345, 0.7526] | 0.7207 [0.7106, 0.7292] |
| **Setting 2 (`Masked_Anchors`)** | DummyClassifier | 0.0996 $\pm$ 0.0000 | 0.3360 [0.3271, 0.3466] | 0.1006 [0.0986, 0.1029] |
| **Setting 2 (`Masked_Anchors`)** | **LogisticRegression (L2)** | **0.5918 $\pm$ 0.0062** | **0.6082 [0.5978, 0.6169]** | **0.5942 [0.5830, 0.6038]** |
| **Setting 2 (`Masked_Anchors`)** | RandomForest (150 trees) | 0.5542 $\pm$ 0.0069 | 0.5671 [0.5556, 0.5765] | 0.5479 [0.5368, 0.5579] |

---

## 11. Figures & Generated Tables
- `results/taste_profile_distributions.png` & `results/example_recipe_profiles.png`
- `results/cuisine_taste_heatmap.png`, `results/cuisine_taste_boxplots.png`, `results/cuisine_distinctive_tastes.png`, `results/cuisine_pca_projection.png`
- `results/confusion_matrix.png`, `results/feature_importance.png`
- `results/cuisine_comparison.csv`, `results/model_comparison.csv`

---

## 12. Limitations & Possible Biases
1. **Unvalidated Human Sensory Ground Truth:** Proxy scores and `dominant_taste` are computational proxies; human sensory evaluation has not been performed.
2. **Unweighted Presence:** Missing ingredient mass quantities treat trace herbs/spices identically to bulk staples.
3. **Omitted Cooking Chemistry & Suppression:** Heating, caramelization, fermentation, and cross-modal masking are not modeled.
4. **Corpus & Geographic Bias:** Kaggle/Yummly recipes skew toward English-language online home cooking and collapse broad national labels.

---

## 13. Reproducibility Instructions
```bash
pip install -r requirements.txt
pytest --cov=src --cov=app -v
python run_pipeline.py --validate-only
python run_pipeline.py                       # Requires data/raw/train.json
python run_pipeline.py --use-sample-data --output-dir tmp_smoke
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 14. Conclusion
This framework provides a transparent, leak-free, tested, and deployable pipeline connecting 38,979 real recipes to FlavorDB-verified ingredient taste proxies, compositional cross-cuisine statistics, and circularity-controlled predictive modeling.

---

## 15. References
1. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
2. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall, London. https://doi.org/10.1007/978-94-009-4109-0
3. Garg, N., et al. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
4. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
5. Martín-Fernández, J. A., Barceló-Vidal, C., & Pawlowsky-Glahn, V. (2003). Dealing with zeros and missing values in compositional data sets using nonparametric imputation. *Mathematical Geology*, 35(3), 253–278. https://doi.org/10.1023/A:1023866030544
6. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv:1803.04343*. https://arxiv.org/abs/1803.04343
7. Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19–25.
