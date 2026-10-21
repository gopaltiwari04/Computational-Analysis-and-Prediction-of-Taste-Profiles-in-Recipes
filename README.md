# Computational Analysis and Prediction of Taste Profiles in Recipes

<<<<<<< HEAD
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
│   ├── **init**.py
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

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/test_preprocessing.py -v
python run_pipeline.py --use-sample-data

### Step-by-Step Commands (Linux / macOS)

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_preprocessing.py -v
python run_pipeline.py --use-sample-data

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

## 7. References

1. Garg, N., et al. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. [https://doi.org/10.1093/nar/gkx957](https://doi.org/10.1093/nar/gkx957)
2. Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. [https://doi.org/10.1038/srep00196](https://doi.org/10.1038/srep00196)
3. Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. [https://doi.org/10.1371/journal.pone.0139539](https://doi.org/10.1371/journal.pone.0139539)
4. Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv preprint arXiv:1803.04343*. [https://arxiv.org/abs/1803.04343](https://arxiv.org/abs/1803.04343)
=======
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test Suite: Pytest](https://img.shields.io/badge/pytest-34%20passed-brightgreen.svg)](tests/)

An end-to-end, scientifically defensible computational gastronomy framework that maps unstructured recipe ingredient lists to standardized 5-dimensional taste simplex profiles (`sweet`, `sour`, `bitter`, `salty`, `umami`), characterizes cross-cuisine variations using non-parametric omnibus tests with multiple-comparison adjustments, and benchmarks leak-free machine learning models that isolate contextual companion-ingredient co-occurrence from circular proxy rule recovery.

---

## 1. System Architecture

```
                                  [ RAW RECIPES ]
                             Kaggle / Yummly train.json
                            (39,774 raw recipe records)
                                         |
                                         v
===================================================================================
DELIVERABLE 1: DATA ENGINEERING & CANONICAL NORMALIZATION (src/data_preprocessing.py)
 - Text cleaning, parenthetical stripping, prep participles removal
 - Canonical ingredient alias resolution (preserving culinary distinctions)
 - Within-cuisine deduplication by canonical ingredient signature (38,985 clean unique)
 - Mapping against 92-entry taste lexicon (61.1% token coverage, 37,033 sufficient)
 - Explicit separation: basic tastants vs. aroma molecules vs. trigeminal chemesthesis
 - Machine-readable manifest output: data/dataset_manifest.json
===================================================================================
                                         |
                                         v
===================================================================================
DELIVERABLE 2: COMPUTATIONAL RECIPE TASTE SIMPLEX (src/taste_profile.py)
 - Unweighted & mass-weighted proxy aggregation: u_d(r) in [0, 1]^5
 - L1 probability simplex projection: p_d(r) = u_d(r) / sum(u), sum(p) == 1
 - Distributional evenness: Shannon entropy H(r) [nats], Pielou's evenness J(r) in [0, 1]
 - Data artifact: data/processed/recipe_taste_features.csv
===================================================================================
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
=========================================== ===========================================
DELIVERABLE 3: CROSS-CUISINE ANALYSIS       DELIVERABLE 4: PREDICTION & RETRIEVAL
(src/cuisine_analysis.py)                   (src/taste_prediction.py)
 - Descriptive stats across 20 cuisines      - Deduplication prior to train/test split
 - Kruskal-Wallis H-tests (df = 19)          - Leak-free preprocessing: CountVectorizer
 - Holm-Bonferroni correction (alpha = 0.05)   fitted strictly on train partition
 - Compositional Centered Log-Ratio (clr)    - Setting 1 (Full Ingredients): Rule recovery
   sensitivity check (Aitchison 1986)        - Setting 2 (Masked Anchors): Context proxy
 - Standardized PCA & Z-score lifts            masking 20 direct sweet/sour/salt/umami anchors
 - Heatmap, boxplots, distinctiveness bar    - 5-fold Stratified Cross-Validation
                                             - Serialized artifact: models/taste_model.joblib
                                             - Standalone CLI inference & cosine retrieval
=========================================== ===========================================
```

---

## 2. Key Empirical Findings (Analyzed on 38,985 Real Recipes)

### A. Cross-Cuisine Taste Distribution (20 Cuisines, $N = 37,033$ Sufficient Recipes)
- **Umami & Salty Dominance in East Asian Cuisines:** Chinese ($z = +0.62$), Korean ($z = +0.50$), and Filipino ($z = +0.29$) cuisines exhibit marked umami enrichment driven by fermented condiments (soy sauce, fish sauce, miso).
- **Sour & Acidic Diversity in Southeast Asian & Latin American Cuisines:** Greek ($z = +0.61$), Mexican ($z = +0.42$), and Thai ($z = +0.30$) cuisines display pronounced sour enrichment via citrus juices, vinegar, and tamarind, accompanied by the highest taste distributional evenness ($J \approx 0.89$).
- **Bitter & Savory Spice Dominance in Indian & Moroccan Traditions:** Moroccan ($z = +0.85$) and Indian ($z = +0.53$) recipes exhibit significant bitter/earthy spice signatures driven by polyphenolic turmeric, cumin, coriander, and fenugreek.
- **Sweet Orientation in Western Bakery & Comfort Cuisines:** British ($z = +0.64$), Russian ($z = +0.50$), Southern US ($z = +0.47$), and Irish ($z = +0.46$) traditions concentrate probability mass on sweet carbohydrates.
- **Omnibus Statistical Significance:** Kruskal–Wallis tests across all 5 dimensions and taste evenness ($J$) yield $p_{adj} < 10^{-300}$ ($df = 19$). Centered Log-Ratio (clr) transformations confirm that significance is not an artifact of simplex closure ($\sum p_d = 1$).

### B. Machine Learning Benchmarks & Target Circularity Disclosure
Because `dominant_taste` is derived by aggregating ingredient scores, full-feature classification (**Setting 1**) tests algebraic rule recovery. To evaluate non-circular contextual pattern recognition, **Setting 2** masks 20 direct taste anchors (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.):

| Evaluation Setting | Model | 5-Fold CV Macro-F1 | Test Accuracy | Test Macro-F1 | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Setting 1 (Full Ingredients)** | DummyClassifier (Most Frequent) | 0.0987 $\pm$ 0.000 | 0.3277 | 0.0987 | Random guessing baseline |
| **Setting 1 (Full Ingredients)** | **LogisticRegression (L2, Balanced)** | **0.9385 $\pm$ 0.003** | **0.9477** | **0.9400** | Linear recovery of proxy rule |
| **Setting 1 (Full Ingredients)** | RandomForest (150 trees, Balanced) | 0.7282 $\pm$ 0.009 | 0.7496 | 0.7292 | Non-linear tree recovery |
| **Setting 2 (Masked Anchors)** | DummyClassifier (Most Frequent) | 0.0988 $\pm$ 0.000 | 0.3280 | 0.0988 | Random guessing baseline |
| **Setting 2 (Masked Anchors)** | **LogisticRegression (L2, Balanced)** | **0.6280 $\pm$ 0.001** | **0.6583** | **0.6310** | **Contextual co-occurrence proxy** |
| **Setting 2 (Masked Anchors)** | RandomForest (150 trees, Balanced) | 0.5724 $\pm$ 0.007 | 0.5964 | 0.5650 | Tree contextual co-occurrence |

*Even when primary taste anchors are masked, companion ingredients predict dominant taste with 0.631 Macro-F1 (vs. 0.099 baseline), demonstrating genuine culinary co-occurrence structure.*

---

## 3. Repository Structure

```
computational-gastronomy/
├── data/
│   ├── raw/
│   │   ├── train.json                       <- Raw Kaggle dataset (39,774 recipes, gitignored)
│   │   └── ingredient_taste_lexicon.csv     <- 92-item taste & chemesthesis reference lexicon
│   ├── processed/
│   │   ├── recipes_clean.csv                <- Deduplicated cleaned recipes
│   │   ├── recipe_ingredients.csv           <- Tokenized recipe-ingredient associations
│   │   ├── ingredient_flavor_mapping.csv    <- Complete audit of matched & unmatched items
│   │   └── recipe_taste_features.csv        <- 5D taste simplex profiles, entropy & balance
│   └── dataset_manifest.json                <- Machine-readable provenance and audit trail
├── docs/
│   ├── quality_audit.md                     <- Comprehensive Phase 1 repository audit
│   ├── validation_report.md                 <- Phase 11 scientific validation report
│   ├── project_proposal.md                  <- Formal research proposal & work distribution
│   └── final_report.md                      <- Full academic report with derivations & tables
├── models/
│   └── taste_model.joblib                   <- Serializable scikit-learn Pipeline bundle
├── reports/
│   ├── dataset_summary.md                   <- Deliverable 1 data engineering report
│   ├── taste_profile_methodology.md         <- Deliverable 2 mathematical formulation report
│   ├── cross_cuisine_analysis.md            <- Deliverable 3 statistical hypothesis test report
│   └── prediction_evaluation.md             <- Deliverable 4 machine learning evaluation report
├── results/
│   ├── cuisine_comparison.csv               <- Per-cuisine descriptive statistics & z-scores
│   ├── model_comparison.csv                 <- Benchmark evaluation metrics & CV scores
│   ├── cuisine_taste_heatmap.png            <- 20-cuisine mean taste heatmap
│   ├── cuisine_taste_boxplots.png           <- Taste distributions across top cuisines
│   ├── cuisine_distinctive_tastes.png       <- Modality z-score lift bar chart
│   ├── cuisine_pca_projection.png           <- Standardized PCA projection of cuisines
│   ├── taste_profile_distributions.png      <- Simplex distributions across corpus
│   ├── example_recipe_profiles.png          <- Representative stacked bar profiles
│   ├── confusion_matrix.png                 <- Test-split classification heatmap
│   └── feature_importance.png               <- Top Random Forest ingredient features
├── src/
│   ├── __init__.py
│   ├── config.py                            <- Central configuration, constants, directory paths
│   ├── data_preprocessing.py                <- Deliverable 1 normalization & manifest pipeline
│   ├── taste_profile.py                     <- Deliverable 2 taste vector & entropy math
│   ├── cuisine_analysis.py                  <- Deliverable 3 Kruskal-Wallis, Holm-Bonf & PCA
│   └── taste_prediction.py                  <- Deliverable 4 leak-free ML, API & retrieval
├── tests/
│   ├── test_preprocessing.py                <- Tests for normalization, aliases, duplicates
│   ├── test_taste_lexicon.py                <- Tests for lexicon bounds, chemesthesis, schema
│   ├── test_taste_profile.py                <- Tests for simplex invariants & entropy math
│   ├── test_cuisine_analysis.py             <- Tests for Holm-Bonferroni & CLR transform
│   ├── test_prediction.py                   <- Tests for leakage prevention, inference & retrieval
│   └── test_pipeline.py                     <- Integration and synthetic smoke tests
├── pytest.ini
├── requirements.txt
├── run_pipeline.py                          <- Master executable CLI
└── README.md
```

---

## 4. Setup & Reproducibility Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/gopaltiwari04/Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes.git
cd Computational-Analysis-and-Prediction-of-Taste-Profiles-in-Recipes

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Reproducible Execution Commands

#### 1. Run Complete Pipeline on Real Dataset (Default)
Requires `data/raw/train.json` (downloaded from [Kaggle](https://www.kaggle.com/c/whats-cooking/data)):
```bash
python run_pipeline.py
```

#### 2. Run Synthetic Smoke Test (Offline Development)
Runs the entire 4-deliverable pipeline deterministically using the built-in synthetic corpus without requiring external downloads:
```bash
python run_pipeline.py --use-sample-data
```

#### 3. Run Automated Pytest Test Suite
Executes 34 unit, invariant, and integration tests:
```bash
pytest -v
```

#### 4. Run Interactive Recipe Taste Prediction
Predicts the dominant taste modality and class probability distribution for any recipe ingredient list:
```bash
python run_pipeline.py --predict "Roma tomatoes, garlic, extra-virgin olive oil, fresh basil"
```
Output:
```
================ RECIPE TASTE PREDICTION ================
Input Raw Ingredients:      ['Roma tomatoes', 'garlic', 'olive oil', 'basil']
Normalized Ingredients:     ['tomato', 'garlic', 'olive oil', 'basil']
Predicted Dominant Taste:   UMAMI
Class Probability Distribution:
  Bitter  : 0.0145 | 
  Salty   : 0.0000 | 
  Sour    : 0.0024 | 
  Sweet   : 0.0090 | 
  Umami   : 0.9741 | #############################
=========================================================
```

#### 5. Retrieve Top-3 Nearest Neighbors in Continuous Taste Space
Finds culinary dishes with the closest taste balance:
```bash
python run_pipeline.py --top-k-similar 10259
```

---

## 5. Scientific Disclaimers & Methodology Scope

1. **Taste Proxy vs. Wet-Lab Sensory Science:** Ingredient taste vectors represent an expert-curated proxy calibrated to sensory reference standards. They do not substitute for gas chromatography-mass spectrometry (GC-MS), electronic tongue assays, or trained sensory panel ratings.
2. **Ingredient Presence vs. Concentration:** In standard recipe corpora lacking gram weights, unweighted averaging treats high-volume staples (e.g. rice) and low-volume spices (e.g. cayenne) equally.
3. **Entropy Evenness vs. Hedonic Balance:** Normalized Shannon entropy ($J$) measures mathematical distributional uniformity across the 5 dimensions, not culinary deliciousness or sensory harmony.
4. **Chemesthesis Distinction:** Pungent ingredients (chilies, black pepper, ginger, mustard) stimulate somatosensory receptors (TRPV1, TRPA1) via the trigeminal nerve and are documented as chemesthetic sensations, separate from lingual taste modalities.

---

## 6. Citations & References

- Ahn, Y.-Y., Ahnert, S. E., Bagrow, J. P., & Barabási, A.-L. (2011). Flavor network and the principles of food pairing. *Scientific Reports*, 1, 196. [doi:10.1038/srep00196](https://doi.org/10.1038/srep00196)
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman and Hall, London.
- Garg, N., Sethupathy, A., Tuwani, R., NK, R., Dokania, S., Iyer, A., Gupta, A., Agrawal, S., Singh, N., Shukla, S., Kathuria, K., Badhwar, R., Kanji, R., Jain, A., Kaur, A., Nagpal, R., & Bagler, G. (2018). FlavorDB: a database of flavor molecules. *Nucleic Acids Research*, 46(D1), D1210–D1216. [doi:10.1093/nar/gkx957](https://doi.org/10.1093/nar/gkx957)
- Jain, A., NK, R., & Bagler, G. (2015). Analysis of food pairing in regional cuisines of India. *PLOS ONE*, 10(10), e0139539. [doi:10.1371/journal.pone.0139539](https://doi.org/10.1371/journal.pone.0139539)
- Singh, N., & Bagler, G. (2018). Data-driven investigations of culinary patterns in traditional recipes across the world. *arXiv:1803.04343*. [arXiv:1803.04343](https://arxiv.org/abs/1803.04343)
- Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19–25.
>>>>>>> d2c6d84 (pipeline upgrade)
