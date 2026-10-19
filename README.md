```markdown
# Computational Analysis and Prediction of Taste Profiles in Recipes

A reproducible **Computational Gastronomy** research pipeline that links culinary recipe ingredient lists to standardized 5-dimensional gustatory taste profiles (`sweet`, `sour`, `bitter`, `salty`, `umami`), conducts non-parametric cross-cuisine statistical comparisons, and evaluates lightweight machine learning models for recipe taste prediction.

---

## 1. Project Overview & Scientific Motivation

Public recipe repositories catalog thousands of dishes as unstructured ingredient lists, while molecular databases such as **FlavorDB** (Garg et al., 2018) catalog volatile aroma molecules. A frequent pitfall in computational food analysis is conflating **olfactory flavor molecules** (volatile compounds driving aroma) with **gustatory taste modalities** (`sweet`, `sour`, `bitter`, `salty`, `umami`, mediated by non-volatile tastants binding to lingual receptors).

This project implements a transparent, four-deliverable framework that:
1. **Separates volatile flavor compound metadata from gustatory taste profiles** and logs unmatched ingredients explicitly rather than silently dropping them.
2. **Formulates a mathematical recipe-level taste representation** on the 5-taste L1 simplex alongside Shannon entropy ($H$) and Pielou taste balance ($J$).
3. **Evaluates cross-cuisine differences** using omnibus Kruskal–Wallis $H$-tests with Holm–Bonferroni multiple-testing correction and standardized $z$-score distinctiveness lifts.
4. **Tests lightweight supervised classifiers (`DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`)** under both a full-ingredient setting (rule recovery) and a **masked-anchor setting** (testing whether companion ingredients predict dominant taste when primary taste anchors like salt, sugar, and lemon juice are removed), alongside an unsupervised cosine-similarity recipe retriever.

---

## 2. Repository Structure

```text
computational-gastronomy/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── run_pipeline.py
├── data/
│   ├── raw/
│   │   └── ingredient_taste_lexicon.csv      # Curated 5D taste + FlavorDB compound reference
│   └── processed/
│       ├── recipes_clean.csv                 # Deduplicated, validated recipe records
│       ├── recipe_ingredients.csv            # Normalized recipe-ingredient token pairs
│       ├── ingredient_flavor_mapping.csv     # Matched & unmatched ingredient audit table
│       └── recipe_taste_features.csv         # 5D recipe taste vectors, entropy & balance
├── notebooks/
│   ├── 01_dataset_construction.ipynb
│   ├── 02_taste_profile_analysis.ipynb
│   ├── 03_cross_cuisine_analysis.ipynb
│   └── 04_taste_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py                 # Deliverable 1 implementation
│   ├── taste_profile.py                      # Deliverable 2 implementation
│   ├── cuisine_analysis.py                   # Deliverable 3 implementation
│   └── taste_prediction.py                   # Deliverable 4 implementation
├── tests/
│   └── test_preprocessing.py                 # Unit tests for normalization & Shannon entropy
├── results/
│   ├── taste_profile_distributions.png
│   ├── example_recipe_profiles.png
│   ├── cuisine_taste_heatmap.png
│   ├── cuisine_taste_boxplots.png
│   ├── cuisine_distinctive_tastes.png
│   ├── cuisine_pca_projection.png
│   ├── cuisine_comparison.csv
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── models/
│   └── taste_model.joblib                    # Serialized classifier & vocabulary
├── reports/
│   ├── literature_review.md
│   ├── dataset_summary.md
│   ├── taste_profile_methodology.md
│   ├── cross_cuisine_analysis.md
│   ├── prediction_evaluation.md
│   └── limitations.md
└── docs/
    ├── project_proposal.md
    └── final_report.md

```

---

## 3. Deliverable-to-Artifact Mapping

| Deliverable | Objective | Source Script & Notebook | Generated Outputs & Reports |
| --- | --- | --- | --- |
| **Deliverable 1** | Recipe–Ingredient–Taste Dataset Construction | `src/data_preprocessing.py`<br>

<br>`notebooks/01_dataset_construction.ipynb` | `data/processed/recipes_clean.csv`<br>

<br>`data/processed/recipe_ingredients.csv`<br>

<br>`data/processed/ingredient_flavor_mapping.csv`<br>

<br>`reports/dataset_summary.md` |
| **Deliverable 2** | Computational Recipe Taste Profile | `src/taste_profile.py`<br>

<br>`notebooks/02_taste_profile_analysis.ipynb` | `data/processed/recipe_taste_features.csv`<br>

<br>`results/taste_profile_distributions.png`<br>

<br>`results/example_recipe_profiles.png`<br>

<br>`reports/taste_profile_methodology.md` |
| **Deliverable 3** | Cross-Cuisine Taste Profile Analysis | `src/cuisine_analysis.py`<br>

<br>`notebooks/03_cross_cuisine_analysis.ipynb` | `results/cuisine_comparison.csv`<br>

<br>`results/cuisine_taste_heatmap.png`<br>

<br>`results/cuisine_taste_boxplots.png`<br>

<br>`results/cuisine_distinctive_tastes.png`<br>

<br>`results/cuisine_pca_projection.png`<br>

<br>`reports/cross_cuisine_analysis.md` |
| **Deliverable 4** | Lightweight Recipe Taste Prediction Model | `src/taste_prediction.py`<br>

<br>`notebooks/04_taste_prediction.ipynb` | `results/model_comparison.csv`<br>

<br>`results/confusion_matrix.png`<br>

<br>`results/feature_importance.png`<br>

<br>`models/taste_model.joblib`<br>

<br>`reports/prediction_evaluation.md` |

---

## 4. Installation & Reproducible Execution

### Prerequisites

* Python 3.11 or 3.12
* `pip` package manager

### Step-by-Step Commands (Windows PowerShell)

```powershell
# 1. (Optional) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Run unit tests (verifies string normalization, alias preservation, and Shannon entropy math)
python -m pytest tests/test_preprocessing.py -v

# 4. Run the complete 4-deliverable pipeline
python run_pipeline.py --use-sample-data

```

### Step-by-Step Commands (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_preprocessing.py -v
python run_pipeline.py --use-sample-data

```

> **Using External Kaggle Data (`train.json`):** If you place the Kaggle/Yummly *What's Cooking* `train.json` file inside `data/raw/train.json`, `python run_pipeline.py` automatically ingests the external file and recomputes all four deliverables without any code edits.

---

## 5. Key Empirical Results

### Deliverable 1 & 2: Dataset & Recipe Taste Representation

* **Cleaned Unique Recipes:** `140` (after filtering invalid entries and deduplicating identical normalized ingredient sets)
* **Cuisines Represented:** `6` (`American_Bakery_Comfort`, `Indian`, `Italian`, `Japanese`, `Mexican`, `Thai`)
* **Unique Normalized Ingredients:** `76`
* **Token-Level Ingredient Match Coverage:** `96.4%` (`140 / 140` cleaned recipes satisfy the $\ge 2$ matched ingredients and $\ge 25\%$ match coverage threshold)

### Deliverable 3: Cross-Cuisine Statistical Comparison

* **Omnibus Kruskal–Wallis Tests:** Statistically significant differences across cuisines for all five basic taste modalities after Holm–Bonferroni multiple-testing correction ($p_{\text{adj}} < 0.001$).
* **Distinctive Culinary Signatures:** Standardized $z$-score lifts highlight `umami` and `salty` enrichment in Japanese recipes (soy sauce, miso, shiitake), `bitter` spice enrichment in Indian recipes (turmeric, fenugreek, cumin), `sour` acid enrichment in Mexican and Thai dishes (lime juice, tamarind), and `sweet` dominance in bakery/comfort recipes.

### Deliverable 4: Predictive Model Evaluation & Circularity Analysis

| Evaluation Setting | Model | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
| --- | --- | --- | --- | --- | --- |
| **Full Ingredients** *(Circular Rule Recovery)* | `DummyClassifier (Most Frequent)` | 0.4000 | 0.1143 | 0.0800 | 0.2000 |
| **Full Ingredients** *(Circular Rule Recovery)* | `LogisticRegression (L2, Balanced)` | **0.8857** | **0.8418** | **0.8600** | **0.8357** |
| **Full Ingredients** *(Circular Rule Recovery)* | `RandomForest (150 trees, Balanced)` | **0.8857** | **0.8418** | **0.8600** | **0.8357** |
| **Masked Anchors** *(Non-Circular Context Proxy)* | `DummyClassifier (Most Frequent)` | 0.4000 | 0.1143 | 0.0800 | 0.2000 |
| **Masked Anchors** *(Non-Circular Context Proxy)* | `LogisticRegression (L2, Balanced)` | 0.8286 | 0.7968 | 0.8121 | 0.8071 |
| **Masked Anchors** *(Non-Circular Context Proxy)* | `RandomForest (150 trees, Balanced)` | **0.8571** | **0.8245** | **0.8455** | **0.8214** |

---

## 6. Scientific Limitations & Disclosures

1. **Flavor Molecules vs. Basic Tastes:** Volatile compounds in FlavorDB represent olfactory aroma notes rather than non-volatile lingual tastants; our lexicon explicitly separates `primary_flavor_compounds` from the five numerical taste scores.
2. **Unweighted Ingredient Lists:** Public recipe datasets omit gram quantities, meaning unweighted mean aggregation treats trace spices and bulk ingredients equally.
3. **Rule-Derived Target Circularity:** Because recipe-level `dominant_taste` is computed from ingredient taste vectors rather than human sensory panels, high accuracy in `Full_Ingredients` reflects recovery of the aggregation rule. The `Masked_Anchors` experiment removes 12 primary taste anchors to measure genuine contextual ingredient co-occurrence generalization.

---

## 7. Team Work Distribution

* **Student A:** Deliverable 1 (Dataset acquisition, normalization, alias resolution, FlavorDB/taste lexicon mapping) and Deliverable 2 (Recipe taste vector formulation, Shannon entropy/balance metrics, unit testing).
* **Student B:** Literature review, Deliverable 3 (Cross-cuisine Kruskal–Wallis statistical analysis, Holm–Bonferroni correction, visualizations), and Deliverable 4 (Predictive modeling, masked-anchor circularity evaluation, final report).

---

## 8. References

1. Garg, N., et al. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. https://doi.org/10.1093/nar/gkx957
2. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. https://doi.org/10.1038/srep00196
3. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. https://doi.org/10.1371/journal.pone.0139539
4. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. https://arxiv.org/abs/1803.04343

```

```
