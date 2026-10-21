# Project Proposal: Computational Analysis and Prediction of Taste Profiles in Recipes

## 1. Abstract
Computational Gastronomy investigates food, recipes, and sensory perception through data-driven modeling and complex systems analysis. While culinary repositories catalog tens of thousands of traditional recipes across world cuisines and molecular databases document volatile flavor compounds, few reproducible scientific pipelines connect ingredient composition to interpretable basic taste representations (`sweet`, `sour`, `bitter`, `salty`, `umami`). This project presents an end-to-end computational framework that standardizes raw recipe ingredient lists, maps them to an expert-curated basic taste proxy lexicon anchored against standard reference solutions, formulates a 5-dimensional probability simplex representation with Shannon entropy evenness metrics, statistically characterizes cross-cuisine variations using non-parametric Kruskal–Wallis tests with Holm–Bonferroni correction and compositional log-ratio sensitivity checks, and benchmarks leak-free lightweight machine learning models while transparently confronting the circularity of rule-derived taste targets.

## 2. Motivation and Problem Statement
Culinary traditions have historically been described through qualitative historical texts or sensory evaluations conducted by trained panels. Although network-based food pairing studies (Ahn et al., 2011; Jain et al., 2015) demonstrated that culinary cultures assemble ingredients according to statistical patterns of volatile flavor compound sharing, volatile molecules primarily activate olfactory sensory neurons in the nasal cavity (aroma) rather than basic lingual taste receptors. 

Crucially, existing recipe corpora (such as the Kaggle / Yummly *What's Cooking* corpus) lack:
1. Quantitative, standardized mappings from unstructured ingredient strings to the five basic taste modalities.
2. A formal mathematical distinction separating basic gustatory tastants, volatile aroma compounds, and trigeminal sensations (such as capsaicin heat or piperine pungency).
3. Statistical analyses that account for the compositional geometry of normalized recipe taste proportions (closure on the simplex $\sum p_d = 1$).
4. Supervised machine learning evaluations that acknowledge the circularity of predicting ingredient-derived taste labels from the very ingredients used to generate them.

This project resolves these challenges through a defensible, reproducible, and mathematically rigorous framework.

## 3. Research Objectives and Questions
We formalize four primary research questions:
- **RQ1 (Data Engineering & Provenance):** How effectively can unstructured recipe ingredient strings across diverse world cuisines be standardized, deduplicated, and mapped to documented gustatory taste proxies while rigorously logging unmatched items and separating trigeminal pungency from basic taste?
- **RQ2 (Mathematical Modeling):** How can recipe taste profiles be formulated on the 5-dimensional probability simplex, and how does normalized Shannon entropy quantify taste distributional evenness?
- **RQ3 (Cross-Cuisine Hypothesis Testing):** Do world cuisines display statistically significant differences in their taste proxy distributions under non-parametric omnibus tests (Kruskal–Wallis) with family-wise multiple-testing adjustments (Holm–Bonferroni) and compositional sensitivity checks?
- **RQ4 (Predictive Modeling & Circularity Control):** How accurately can lightweight classifiers predict a recipe's dominant taste modality from remaining companion ingredients when direct taste-bearing anchor ingredients (such as salt, sugar, vinegar, or soy sauce) are systematically masked?

## 4. Four Mandatory Deliverables
- **Deliverable 1 (Dataset Construction & Normalization):**
  - Robust preprocessing pipeline resolving spelling variants, knife cuts, and measurement stopwords while preserving culinary distinctions (e.g., `garlic` vs. `black garlic`; `ginger` vs. `dried ginger`; `bell pepper` vs. `cayenne pepper`).
  - Strict provenance manifest (`data/dataset_manifest.json`) supporting real-world recipe data (Kaggle/Yummly 39,774 recipes) with explicit, non-silent opt-in for synthetic smoke-testing (`--use-sample-data`).
  - Comprehensive ingredient-to-taste mapping table (`ingredient_flavor_mapping.csv`) logging both matched and unmatched ingredients.
- **Deliverable 2 (Computational Recipe Taste Profile):**
  - Formulation of unweighted and quantity-weighted taste proxy intensity vectors $\mathbf{u}(r) \in [0, 1]^5$.
  - $L_1$-normalization onto the probability simplex $\mathbf{p}(r)$ ($\sum_{d} p_d = 1$).
  - Computation of Shannon entropy $H(r)$ [nats] and Pielou's evenness $J(r) = H(r)/\ln(5) \in [0, 1]$.
  - Methodology report (`reports/taste_profile_methodology.md`) and distribution visualizations.
- **Deliverable 3 (Cross-Cuisine Statistical Analysis):**
  - Omnibus Kruskal–Wallis $H$-tests and one-way ANOVA $F$-tests across all represented cuisines with degrees of freedom.
  - Non-parametric effect size estimation (eta-squared $\eta_H^2$).
  - Holm–Bonferroni step-down multiple-comparison adjustment ($\alpha = 0.05$).
  - Compositional Centered Log-Ratio (clr) sensitivity verification (Aitchison, 1986).
  - Standardized PCA projection and publication-quality heatmaps, box plots, and $z$-score distinctiveness charts.
- **Deliverable 4 (Lightweight Prediction & Retrieval):**
  - Rigorous leak-free evaluation protocol fitting feature extractors strictly on training partitions after recipe deduplication.
  - Transparent benchmark comparing `DummyClassifier`, `LogisticRegression` (L2, balanced), and `RandomForestClassifier` (150 trees, balanced) under two distinct experimental settings:
    - *Setting 1 (Full Ingredients):* Circular algebraic rule recovery.
    - *Setting 2 (Masked Anchors):* Contextual companion-ingredient co-occurrence generalization masking 20 direct taste anchors.
  - 5-fold Stratified Cross-Validation on the training partition and held-out test split metrics (accuracy, macro-F1, macro-precision, macro-recall, confusion matrices).
  - Production-grade serializable `Pipeline` artifact and standalone inference API (`predict_recipe_taste`).
  - Unsupervised cosine-similarity recipe retrieval strictly excluding query identity.

## 5. Grounding Literature & Verifiable References
1. **FlavorDB & Molecular Repositories:**  
   Garg, N., Sethupathy, A., Tuwani, R., NK, R., Dokania, S., Iyer, A., Gupta, A., Agrawal, S., Singh, N., Shukla, S., Kathuria, K., Badhwar, R., Kanji, R., Jain, A., Kaur, A., Nagpal, R., & Bagler, G. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
2. **Food Pairing Principles:**  
   Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
3. **Regional Culinary Analysis:**  
   Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
4. **Computational Gastronomy Survey:**  
   Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343
5. **Compositional Data Analysis:**  
   Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall, London.
6. **Non-Parametric Effect Size:**  
   Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. An overview of some recommended measures of effect size. *Trends in Sport Sciences*, 1(21), 19–25.

## 6. Work Distribution Between Two Collaborating Students
| Team Member | Primary Roles & Deliverables | Code & Report Modules |
| :--- | :--- | :--- |
| **Student A** | Data engineering, ingredient normalization, alias resolution, taste lexicon curation, chemesthesis distinction, taste vector aggregation, Shannon entropy / evenness math, unit tests. | `src/data_preprocessing.py`, `src/taste_profile.py`, `tests/test_preprocessing.py`, `tests/test_taste_lexicon.py`, `tests/test_taste_profile.py`, `reports/dataset_summary.md`, `reports/taste_profile_methodology.md` |
| **Student B** | Statistical analysis, Kruskal–Wallis omnibus tests, Holm–Bonferroni correction, clr sensitivity check, PCA visualization, leak-free ML pipelines, masked-anchor evaluation, standalone inference API, final report synthesis. | `src/cuisine_analysis.py`, `src/taste_prediction.py`, `run_pipeline.py`, `tests/test_cuisine_analysis.py`, `tests/test_prediction.py`, `reports/cross_cuisine_analysis.md`, `reports/prediction_evaluation.md`, `docs/final_report.md` |
