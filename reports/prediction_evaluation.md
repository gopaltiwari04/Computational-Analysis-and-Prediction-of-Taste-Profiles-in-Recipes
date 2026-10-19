# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

## 1. Target Definition & Methodological Disclosure on Circularity
- **Prediction Target:** Recipe `dominant_taste` $\in \{\text{sweet}, \text{sour}, \text{bitter}, \text{salty}, \text{umami}\}$.
- **Critical Scientific Disclosure:** Public culinary datasets (Kaggle/Yummly `train.json` and CulinaryDB) do not contain human sensory panel taste ratings at the recipe level. Because `dominant_taste` is derived by aggregating ingredient-level taste annotations in Deliverable 2, training a supervised classifier on the exact same ingredient list (`Full_Ingredients`) measures how effectively a linear or tree-based model **recovers the aggregation rule**, not independent empirical sensory validity.
- **Mitigation & Non-Circular Evaluation (`Masked_Anchors`):** To test genuine contextual pattern learning, we evaluate a second setting in which 12 primary taste-anchor ingredients (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.) are masked from the input features. High performance in `Masked_Anchors` demonstrates that companion ingredients co-occur in predictable culinary assemblies (e.g., vanilla + cinnamon + wheat flour predicting `sweet`).

## 2. Data Split & Leakage Prevention
- **Total Deduplicated Recipes:** 140 (stratified 75% train / 25% test split, `random_state=42`).
- **Leakage Guard:** Recipes with identical normalized ingredient sets were deduplicated prior to splitting so no identical ingredient bag-of-words appears in both train and test partitions.

## 3. Model Comparison Results
| evaluation_setting                          | model                              |   accuracy |   macro_f1 |   macro_precision |   macro_recall |
|:--------------------------------------------|:-----------------------------------|-----------:|-----------:|------------------:|---------------:|
| Full_Ingredients (Circular Rule Recovery)   | DummyClassifier (Most Frequent)    |     0.4    |     0.1143 |            0.08   |         0.2    |
| Full_Ingredients (Circular Rule Recovery)   | LogisticRegression (L2, Balanced)  |     0.8857 |     0.8418 |            0.86   |         0.8357 |
| Full_Ingredients (Circular Rule Recovery)   | RandomForest (150 trees, Balanced) |     0.8857 |     0.8418 |            0.86   |         0.8357 |
| Masked_Anchors (Non-Circular Context Proxy) | DummyClassifier (Most Frequent)    |     0.4    |     0.1143 |            0.08   |         0.2    |
| Masked_Anchors (Non-Circular Context Proxy) | LogisticRegression (L2, Balanced)  |     0.8286 |     0.7968 |            0.8121 |         0.8071 |
| Masked_Anchors (Non-Circular Context Proxy) | RandomForest (150 trees, Balanced) |     0.8571 |     0.8245 |            0.8455 |         0.8214 |

## 4. Exploratory Alternative: Cosine-Similarity Recipe Retrieval
For applications where supervised labels are circular, unsupervised retrieval in the 5D taste space finds recipes with matching sensory balance. Top-3 nearest neighbors for Recipe ID `1000` (`Italian`, dominant taste `umami`):

|   neighbor_recipe_id | cuisine   | dominant_taste   |   cosine_similarity |
|---------------------:|:----------|:-----------------|--------------------:|
|                 1024 | Italian   | umami            |              0.9966 |
|                 1023 | Italian   | umami            |              0.9866 |
|                 1009 | Italian   | umami            |              0.9865 |
