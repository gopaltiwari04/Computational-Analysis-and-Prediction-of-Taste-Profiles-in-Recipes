# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

> **Corpus Notice:** Evaluated on **REAL EMPIRICAL CORPUS** (`Kaggle / Yummly What's Cooking Dataset`).

## 1. Three Distinct Evaluation Tasks & Circularity Disclosure
We explicitly separate three fundamentally different modeling tasks:
1. **Setting 1 — Full-Ingredient Rule Recovery (`Full_Ingredients`):** Classifies the rule-derived `dominant_taste` label from the full ingredient bag. High performance reflects linear/non-linear **recovery of the algebraic lexicon aggregation rule**, not empirical human taste prediction.
2. **Setting 2 — Masked-Anchor Co-Occurrence Prediction (`Masked_Anchors`):** Removes **43 canonical taste-anchor ingredients** (all lexicon entries with intensity $\ge 0.75$ plus direct acid/mineral/umami/bitter anchors) AND regex-matched anchor-family surface variants (`ANCHOR_WORD_REGEX`, preventing alias bypass). Performance reflects how well remaining companion ingredients (e.g., herbs, alliums, flours, fats, base proteins) predict the **proxy label via culinary co-occurrence**, NOT independently measured human sensory perception.
3. **Genuine Empirical Sensory Prediction (Unverified / Absent):** Public recipe corpora (`train.json`) contain **zero** human sensory panel taste ratings; independent empirical validation against human perception remains outside the available data.

## 2. Leak-Free Protocol, Class Balance & Post-Masking Accounting
- **Random Seed:** `42` | **Partition:** 75% Train ($N_{train} = 28,044$) / 25% Held-Out Test ($N_{test} = 9,348$)
- **Pre-Split Deduplication:** Exact normalized ingredient bags deduplicated prior to splitting (37,392 unique recipe bags).
- **Pipeline CV Integrity:** `CountVectorizer` (`min_df=2`) is nested inside the `sklearn.pipeline.Pipeline` during 5-fold Stratified CV so validation folds never influence vocabulary filtering.
- **Post-Masking Retention:** 37,295 / 37,392 recipes (**99.74%**) retained $\ge 2$ non-anchor ingredients (`Setting 2` train = 27,971, test = 9,142 after excluding **182** post-masking duplicate collisions across test/train).

### Class Balance (Setting 1 vs. Post-Masking Setting 2)
| dominant_taste   |   setting1_count |   setting1_pct |   setting2_retained_count |   setting2_pct |
|:-----------------|-----------------:|---------------:|--------------------------:|---------------:|
| bitter           |             3634 |           9.72 |                      3632 |           9.74 |
| salty            |             8642 |          23.11 |                      8623 |          23.12 |
| sour             |             4359 |          11.66 |                      4344 |          11.65 |
| sweet            |             8389 |          22.44 |                      8334 |          22.35 |
| umami            |            12368 |          33.08 |                     12362 |          33.15 |

## 3. Benchmark Model Comparison Results (with 95% Bootstrap CIs)
| evaluation_setting                                   | model                              |   n_train |   n_test |   cv_f1_macro_mean |   cv_f1_macro_std |   test_accuracy | test_acc_ci95    |   test_macro_f1 | test_f1_ci95     |   test_macro_precision |   test_macro_recall |
|:-----------------------------------------------------|:-----------------------------------|----------:|---------:|-------------------:|------------------:|----------------:|:-----------------|----------------:|:-----------------|-----------------------:|--------------------:|
| Setting 1: Full_Ingredients (Rule Recovery)          | DummyClassifier (Most Frequent)    |     28044 |     9348 |             0.0994 |            0      |          0.3308 | [0.3228, 0.3401] |          0.0994 | [0.0976, 0.1015] |                 0.0662 |              0.2    |
| Setting 1: Full_Ingredients (Rule Recovery)          | LogisticRegression (L2, Balanced)  |     28044 |     9348 |             0.9242 |            0.0016 |          0.9411 | [0.9365, 0.9453] |          0.9345 | [0.9290, 0.9397] |                 0.9277 |              0.9423 |
| Setting 1: Full_Ingredients (Rule Recovery)          | RandomForest (150 trees, Balanced) |     28044 |     9348 |             0.7157 |            0.0068 |          0.7441 | [0.7345, 0.7526] |          0.7207 | [0.7106, 0.7292] |                 0.7133 |              0.7419 |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | DummyClassifier (Most Frequent)    |     27971 |     9142 |             0.0996 |            0      |          0.336  | [0.3271, 0.3466] |          0.1006 | [0.0986, 0.1029] |                 0.0672 |              0.2    |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | LogisticRegression (L2, Balanced)  |     27971 |     9142 |             0.5918 |            0.0062 |          0.6082 | [0.5978, 0.6169] |          0.5942 | [0.5830, 0.6038] |                 0.5846 |              0.6165 |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | RandomForest (150 trees, Balanced) |     27971 |     9142 |             0.5542 |            0.0069 |          0.5671 | [0.5556, 0.5765] |          0.5479 | [0.5368, 0.5579] |                 0.5443 |              0.5767 |

### Per-Class Held-Out Test Breakdown — Setting 1 (Logistic Regression)
```
              precision    recall  f1-score   support

      bitter       0.87      0.95      0.91       909
       salty       0.94      0.95      0.94      2160
        sour       0.90      0.94      0.92      1090
       sweet       0.95      0.94      0.95      2097
       umami       0.97      0.93      0.95      3092

    accuracy                           0.94      9348
   macro avg       0.93      0.94      0.93      9348
weighted avg       0.94      0.94      0.94      9348

```

### Per-Class Held-Out Test Breakdown — Setting 2 Masked Anchors (Logistic Regression)
```
              precision    recall  f1-score   support

      bitter       0.55      0.75      0.63       899
       salty       0.57      0.55      0.56      2099
        sour       0.42      0.55      0.47      1070
       sweet       0.64      0.59      0.62      2002
       umami       0.75      0.63      0.69      3072

    accuracy                           0.61      9142
   macro avg       0.58      0.62      0.59      9142
weighted avg       0.63      0.61      0.61      9142

```

*(Note on Feature Importances: Random Forest MDI bars in `results/feature_importance.png` distribute importance across correlated co-occurring ingredients and should be interpreted cautiously as associative rather than causal indicators.)*

## 4. Alternative: Unsupervised Taste Simplex Retrieval
Cosine similarity in the 5-dimensional modeled taste simplex retrieves recipes with matching taste-share geometry (not proof of identical plated sensory experience):
- **Query Recipe ID:** `10259` (greek, Dominant: `umami`)
- **Top-3 Nearest Neighbors:**
|   neighbor_recipe_id | cuisine   | dominant_taste   |   taste_evenness |   cosine_similarity |
|---------------------:|:----------|:-----------------|-----------------:|--------------------:|
|                38210 | mexican   | umami            |           0.9634 |              0.9993 |
|                46611 | mexican   | umami            |           0.9697 |              0.9991 |
|                47864 | thai      | umami            |           0.9537 |              0.9991 |

*(Invariant Verified: Query recipe `10259` is strictly excluded from its own neighbors.)*
