# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

## 1. Prediction Target & Critical Methodological Disclosure on Circularity
- **Prediction Target:** Recipe `dominant_taste` $\in \{\text{sweet}, \text{sour}, \text{bitter}, \text{salty}, \text{umami}\}$.
- **Circularity Disclosure:** Public recipe corpora (e.g. Kaggle/Yummly `train.json`) do not contain sensory panel taste ratings. Because `dominant_taste` is derived by aggregating ingredient-level scores in Deliverable 2, supervised learning on the full ingredient list (**Setting 1**) tests the model's ability to **recover the algebraic aggregation rule**, not independent empirical human taste perception.
- **Contextual Generalization Protocol (**Setting 2: Masked_Anchors**):** To test non-circular culinary structure, we mask 21 primary taste-anchor ingredients (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.). In Setting 2, the classifier must infer the dish's taste orientation entirely from companion ingredients (e.g., vanilla + flour + baking powder predicting `sweet`; cumin + turmeric + ginger predicting `bitter`/savory).
- **Proportion of Recipes Retaining Useful Context:** 36,933 / 36,986 recipes (99.9%) retained $\ge 2$ informative ingredients after masking.

## 2. Leak-Free Evaluation Methodology
- **Deduplication:** Recipes sharing identical normalized ingredient bags were deduplicated prior to partitioning, guaranteeing that identical recipes never cross train/test boundaries.
- **Leak-Free Preprocessing:** `CountVectorizer` was fitted strictly on the training partition ($75\%$, $N_{train} = 27,739$); the held-out test partition ($25\%$, $N_{test} = 9,247$) was transformed using the training vocabulary.
- **Cross-Validation:** 5-fold Stratified Cross-Validation was performed across the training set to evaluate generalization stability (mean $\pm$ standard deviation macro-F1).

## 3. Benchmark Model Comparison Results
| evaluation_setting                                   | model                              |   cv_f1_macro_mean |   cv_f1_macro_std |   test_accuracy |   test_macro_f1 |   test_macro_precision |   test_macro_recall |
|:-----------------------------------------------------|:-----------------------------------|-------------------:|------------------:|----------------:|----------------:|-----------------------:|--------------------:|
| Setting 1: Full_Ingredients (Circular Rule Recovery) | DummyClassifier (Most Frequent)    |             0.0987 |            0      |          0.3277 |          0.0987 |                 0.0655 |              0.2    |
| Setting 1: Full_Ingredients (Circular Rule Recovery) | LogisticRegression (L2, Balanced)  |             0.9385 |            0.0025 |          0.9477 |          0.94   |                 0.9318 |              0.9498 |
| Setting 1: Full_Ingredients (Circular Rule Recovery) | RandomForest (150 trees, Balanced) |             0.7282 |            0.0088 |          0.7496 |          0.7292 |                 0.7203 |              0.7633 |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | DummyClassifier (Most Frequent)    |             0.0988 |            0      |          0.328  |          0.0988 |                 0.0656 |              0.2    |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | LogisticRegression (L2, Balanced)  |             0.628  |            0.0014 |          0.6583 |          0.631  |                 0.6223 |              0.6516 |
| Setting 2: Masked_Anchors (Contextual Co-occurrence) | RandomForest (150 trees, Balanced) |             0.5724 |            0.0069 |          0.5964 |          0.565  |                 0.5655 |              0.6007 |

### Per-Class Test Set Breakdown (Logistic Regression, Setting 1)
```
              precision    recall  f1-score   support

      bitter       0.92      0.94      0.93      1167
       salty       0.94      0.94      0.94      1852
        sour       0.85      0.96      0.91       761
       sweet       0.96      0.96      0.96      2437
       umami       0.97      0.94      0.96      3030

    accuracy                           0.95      9247
   macro avg       0.93      0.95      0.94      9247
weighted avg       0.95      0.95      0.95      9247

```

## 4. Alternative: Unsupervised Taste Simplex Retrieval
When supervised labels are circular or categorical targets are too coarse, unsupervised retrieval in the continuous 5-dimensional taste simplex space finds culinary neighbors with matching taste distribution:
- **Query Recipe ID:** `10259` (greek, Dominant: `umami`)
- **Top-3 Nearest Neighbors:**
|   neighbor_recipe_id | cuisine   | dominant_taste   |   taste_balance |   cosine_similarity |
|---------------------:|:----------|:-----------------|----------------:|--------------------:|
|                 1160 | mexican   | umami            |          0.9734 |              0.9996 |
|                16025 | mexican   | umami            |          0.9804 |              0.9993 |
|                16822 | mexican   | umami            |          0.9735 |              0.9993 |

*(Invariant Verified: Query recipe `10259` is strictly excluded from its own nearest neighbors.)*
