# Project Proposal: Computational Analysis and Prediction of Taste Profiles in Recipes

## 1. Abstract
Computational Gastronomy investigates food, recipes, and sensory perception through data-driven modeling. While public repositories catalog thousands of traditional recipes across world cuisines and molecular databases document volatile flavor compounds, few lightweight, reproducible pipelines connect ingredient composition to interpretable basic taste representations (`sweet`, `sour`, `bitter`, `salty`, `umami`). This two-student undergraduate mini-project constructs a standardized recipe–ingredient–taste dataset, formulates a mathematical representation of recipe-level taste profiles, evaluates statistical variations in taste profiles across cuisines using non-parametric hypothesis testing with multiple-comparison adjustments, and benchmarks lightweight classification and retrieval models while transparently addressing circularity in rule-derived taste targets.

## 2. Motivation and Problem Statement
Traditional culinary analysis relies on qualitative cultural descriptions or expensive sensory panels. Although Ahn et al. (2011) and Jain et al. (2015) pioneered network-based food pairing analysis using volatile flavor compounds, volatile aroma molecules do not map one-to-one onto basic gustatory modalities (sweet, sour, bitter, salty, umami). Students and researchers need a clean, scientifically honest framework that separates olfactory compound metadata from gustatory taste profiles and quantifies how ingredient assemblies shape dish-level taste balance across cuisines.

## 3. Objectives
1. Standardize raw recipe ingredient lists and link them to documented taste and FlavorDB attributes while recording unmatched ingredients.
2. Formulate a quantitative, normalized 5D taste vector and entropy-based taste balance metric for recipes.
3. Statistically compare taste distributions across cuisines using non-parametric omnibus tests and effect-size distinctiveness metrics.
4. Evaluate lightweight machine learning models (`DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`) and cosine-similarity retrieval for estimating recipe taste characteristics.

## 4. Four Mandatory Deliverables
- **Deliverable 1 (Dataset Construction):** Clean recipe corpus, normalize ingredient strings, resolve aliases without collapsing distinct ingredients, match against the taste/FlavorDB lexicon, and log coverage in `reports/dataset_summary.md`.
- **Deliverable 2 (Computational Recipe Taste Profile):** Aggregate ingredient vectors into L1-normalized recipe profiles, Shannon entropy diversity, and Pielou balance scores (`src/taste_profile.py`).
- **Deliverable 3 (Cross-Cuisine Taste Analysis):** Group recipes by cuisine, run Kruskal–Wallis tests with Holm–Bonferroni correction, quantify distinctiveness via standardized $z$-score lift, and visualize via heatmaps, box plots, bar charts, and PCA (`src/cuisine_analysis.py`).
- **Deliverable 4 (Lightweight Taste Prediction Model):** Compare baseline and lightweight classifiers under both full-ingredient (rule-recovery) and masked-anchor (contextual co-occurrence) settings alongside nearest-neighbor recipe retrieval (`src/taste_prediction.py`).

## 5. Supporting Research Papers
1. Garg, N., Sethupathy, A., Tuwani, R., NK, R., Dokania, S., Iyer, A., Gupta, A., Agrawal, S., Singh, N., Shukla, S., Kathuria, K., Badhwar, R., Kanji, R., Jain, A., Kaur, A., Nagpal, R., & Bagler, G. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
2. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
3. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
4. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343

## 6. Expected Outcomes
A reproducible Python repository generating cleaned CSV tables, seven publication-quality visualizations, unit tests, and empirical reports demonstrating how ingredient composition governs recipe taste profiles.

## 7. Work Distribution Between Two Students
| Team Member | Primary Responsibilities & Deliverables | Code & Report Ownership |
| :--- | :--- | :--- |
| **Student A** | Deliverable 1 (Dataset acquisition, normalization, alias resolution, FlavorDB/taste lexicon mapping) & Deliverable 2 (Recipe taste vector formulation, Shannon entropy/balance metrics, unit tests). | `src/data_preprocessing.py`, `src/taste_profile.py`, `tests/test_preprocessing.py`, `reports/dataset_summary.md`, `reports/taste_profile_methodology.md` |
| **Student B** | Literature review, Deliverable 3 (Cross-cuisine statistical testing, Holm–Bonferroni correction, PCA/heatmap/boxplot visualizations) & Deliverable 4 (Predictive modeling, circularity analysis, final report synthesis). | `src/cuisine_analysis.py`, `src/taste_prediction.py`, `reports/literature_review.md`, `reports/cross_cuisine_analysis.md`, `reports/prediction_evaluation.md`, `docs/final_report.md` |