# Project Proposal: Computational Analysis and Prediction of Taste Profiles in Recipes

## 1. Abstract
Computational Gastronomy analyzes culinary composition and sensory patterns using data engineering, statistics, and machine learning. While large recipe corpora capture cultural ingredient assemblies and molecular databases such as FlavorDB document volatile aroma molecules, connecting raw ingredient text to interpretable basic taste representations (`sweet`, `sour`, `bitter`, `salty`, `umami`) requires strict separation between gustatory tastants, retronasal aroma, and trigeminal chemesthesis. This project implements a four-deliverable reproducible research pipeline evaluated on 39,774 recipes across 20 cuisines from the Kaggle / Yummly *What's Cooking* corpus, paired with an automated test suite and a deployable FastAPI web explorer.

## 2. Motivation and Problem Statement
Network-based culinary studies (Ahn et al., 2011; Jain et al., 2015) demonstrated cross-cultural structure in volatile flavor compound sharing. However, volatile molecules primarily activate olfactory receptors via retronasal airflow rather than lingual basic-taste receptors. Existing computational pipelines face four recurring pitfalls:
1. Conflating FlavorDB volatile aroma profiles with quantitative basic taste intensities or fabricating database identifiers.
2. Collapsing culinary distinctions (e.g., fresh vs. dried ginger, bell pepper vs. cayenne pepper, beef vs. beef broth) during text normalization, or silently dropping unmatched items.
3. Applying independent univariate tests to simplex-constrained proportions ($\sum p_d = 1$) without accounting for compositional closure.
4. Evaluating supervised classifiers on rule-derived taste targets without controlling vocabulary leakage or isolating direct taste anchors.

## 3. Research Questions & Objectives
- **RQ1 (Data Engineering & Lexicon Provenance):** How cleanly can unstructured recipe ingredients be normalized, deduplicated within cuisines without erasing cross-cuisine parallels, and linked to FlavorDB-verified entity metadata while separating trigeminal pungency and unknown (`NaN`) taste dimensions?
- **RQ2 (Simplex Taste Modeling):** How can matched ingredient proxies be aggregated onto the 5-dimensional probability simplex and summarized via Shannon entropy ($H$) and Pielou's distributional evenness ($J$)?
- **RQ3 (Cross-Cuisine Hypothesis Testing):** Do world cuisines exhibit statistically and practically meaningful differences in modeled taste-share profiles under Kruskal–Wallis tests with Holm–Bonferroni correction and joint Aitchison CLR PERMANOVA?
- **RQ4 (Leak-Free Prediction & Co-Occurrence Structure):** When direct taste-bearing anchor ingredients (salt, sugars, acids, fermented sauces, etc.) and their surface variants are masked, to what extent do companion ingredients predict the rule-derived `dominant_taste` proxy via culinary co-occurrence?

## 4. Four Core Project Deliverables
1. **Deliverable 1 — Dataset Construction & Normalization (`src/data_preprocessing.py`):** Schema validation, deterministic alias resolution (`data/raw/ingredient_aliases.csv`), within-cuisine deduplication, verified 122-entry taste proxy lexicon (`data/raw/ingredient_taste_lexicon.csv`), provenance manifest (`data/dataset_manifest.json`), and `reports/dataset_summary.md`.
2. **Deliverable 2 — Computational Recipe Taste Profile (`src/taste_profile.py`):** `NaN`-aware aggregation, optional quantity-weighted hook (never inventing weights when absent), $L_1$ simplex normalization, Shannon entropy $H$ [nats], Pielou's evenness $J \in [0, 1]$, distribution plots, and `reports/taste_profile_methodology.md`.
3. **Deliverable 3 — Cross-Cuisine Statistical Analysis (`src/cuisine_analysis.py`):** Per-cuisine descriptive summaries and standardized $z$-lifts, Kruskal–Wallis $H$-tests with Holm–Bonferroni adjustment and bootstrap 95% CIs on $\eta_H^2$, multiplicative-zero CLR transformation and multivariate PERMANOVA, standardized centroid PCA biplot, and `reports/cross_cuisine_analysis.md`.
4. **Deliverable 4 — Lightweight Taste Prediction & Retrieval (`src/taste_prediction.py`):** Leak-free pipeline cross-validation and held-out benchmarking (`DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`) across Setting 1 (`Full_Ingredients`) and Setting 2 (`Masked_Anchors`), dual-pipeline serialization (`models/taste_model.joblib`), CLI inference with OOV handling, cosine-similarity retrieval, and `reports/prediction_evaluation.md`.

*(Note: Original repository documentation also included an optional two-student work breakdown in early course drafts; all deliverables are preserved and unified in this verified codebase.)*

## 5. Verifiable References
1. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
2. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall, London. https://doi.org/10.1007/978-94-009-4109-0
3. Garg, N., Sethupathy, A., Tuwani, R., NK, R., Dokania, S., Iyer, A., Gupta, A., Agrawal, S., Singh, N., Shukla, S., Kathuria, K., Badhwar, R., Kanji, R., Jain, A., Kaur, A., Nagpal, R., & Bagler, G. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
4. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
5. Martín-Fernández, J. A., Barceló-Vidal, C., & Pawlowsky-Glahn, V. (2003). Dealing with zeros and missing values in compositional data sets using nonparametric imputation. *Mathematical Geology*, 35(3), 253–278. https://doi.org/10.1023/A:1023866030544
6. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343
7. Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. An overview of some recommended measures of effect size. *Trends in Sport Sciences*, 1(21), 19–25.
