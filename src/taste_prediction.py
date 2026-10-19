"""
Deliverable 4: Lightweight Recipe Taste Prediction Model.
Implements leak-free train/test splitting, DummyClassifier baseline,
Logistic Regression, Random Forest, non-circular masked-anchor evaluation,
and cosine-similarity recipe retrieval.
"""
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    TASTE_DIMENSIONS,
    ensure_directories,
)

# Direct anchor ingredients masked in Setting 2 to test non-circular contextual generalization
DIRECT_TASTE_ANCHORS = {
    "salt",
    "sugar",
    "brown sugar",
    "honey",
    "maple syrup",
    "lemon juice",
    "lime juice",
    "vinegar",
    "rice vinegar",
    "soy sauce",
    "fish sauce",
    "cocoa powder",
}


def build_feature_matrix(
    df: pd.DataFrame, mask_anchors: bool = False
) -> Tuple[np.ndarray, List[str], CountVectorizer]:
    """Construct binary ingredient indicator matrix from normalized ingredient strings."""
    corpus = []
    for ing_str in df["ingredients_normalized"]:
        tokens = [t.strip() for t in str(ing_str).split(";") if t.strip()]
        if mask_anchors:
            tokens = [t for t in tokens if t not in DIRECT_TASTE_ANCHORS]
        corpus.append(";".join(tokens))

    vectorizer = CountVectorizer(tokenizer=lambda s: s.split(";") if s else [], binary=True, token_pattern=None)
    X = vectorizer.fit_transform(corpus).toarray()
    feature_names = list(vectorizer.get_feature_names_out())
    return X, feature_names, vectorizer


def evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    setting_label: str,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Train DummyClassifier, LogisticRegression, and RandomForestClassifier and compute metrics."""
    models = {
        "DummyClassifier (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "LogisticRegression (L2, Balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "RandomForest (150 trees, Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    rows = []
    fitted = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        rows.append(
            {
                "evaluation_setting": setting_label,
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "macro_f1": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
                "macro_precision": round(float(precision_score(y_test, preds, average="macro", zero_division=0)), 4),
                "macro_recall": round(float(recall_score(y_test, preds, average="macro", zero_division=0)), 4),
            }
        )
        fitted[name] = (clf, preds)

    return rows, fitted


def retrieve_similar_recipes(df: pd.DataFrame, query_recipe_id: int, top_k: int = 3) -> pd.DataFrame:
    """Exploratory similarity-based retrieval using 5D normalized taste profiles."""
    X_taste = df[TASTE_DIMENSIONS].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine")
    nn.fit(X_taste)

    query_idx = df.index[df["recipe_id"] == query_recipe_id].tolist()[0]
    distances, indices = nn.kneighbors(X_taste[query_idx : query_idx + 1])

    results = []
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        row = df.iloc[idx]
        results.append(
            {
                "neighbor_recipe_id": int(row["recipe_id"]),
                "cuisine": row["cuisine"],
                "dominant_taste": row["dominant_taste"],
                "cosine_similarity": round(float(1.0 - dist), 4),
            }
        )
    return pd.DataFrame(results)


def run_deliverable_4() -> pd.DataFrame:
    """Execute Deliverable 4 model training, evaluation, plotting, and reporting."""
    ensure_directories()

    df = pd.read_csv(DATA_PROCESSED_DIR / "recipe_taste_features.csv")
    # Deduplicate by normalized ingredient signature so identical ingredient sets never split across train/test
    df = df.drop_duplicates(subset=["ingredients_normalized"]).reset_index(drop=True)

    y = df["dominant_taste"].to_numpy()
    # Filter rare classes with fewer than 2 instances if any exist
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= 2].index.tolist()
    df = df[df["dominant_taste"].isin(valid_classes)].reset_index(drop=True)
    y = df["dominant_taste"].to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Setting 1: Full Ingredient Features (Rule Recovery)
    X_full, feat_full, vec_full = build_feature_matrix(df, mask_anchors=False)
    rows_full, fitted_full = evaluate_models(
        X_full[train_idx], X_full[test_idx], y[train_idx], y[test_idx], "Full_Ingredients (Circular Rule Recovery)"
    )

    # Setting 2: Masked Direct Taste Anchors (Contextual Co-occurrence Generalization)
    X_mask, feat_mask, vec_mask = build_feature_matrix(df, mask_anchors=True)
    rows_mask, fitted_mask = evaluate_models(
        X_mask[train_idx], X_mask[test_idx], y[train_idx], y[test_idx], "Masked_Anchors (Non-Circular Context Proxy)"
    )

    all_results_df = pd.DataFrame(rows_full + rows_mask)
    all_results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Save best model artifact
    best_clf, best_preds = fitted_full["LogisticRegression (L2, Balanced)"]
    joblib.dump(
        {"model": best_clf, "vectorizer_vocab": feat_full, "classes": list(best_clf.classes_)},
        MODELS_DIR / "taste_model.joblib",
    )

    # Plot Confusion Matrix (Logistic Regression on Test Split)
    labels = sorted(list(set(y)))
    cm = confusion_matrix(y[test_idx], best_preds, labels=labels)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
    )
    plt.title("Test-Set Confusion Matrix (Logistic Regression)")
    plt.xlabel("Predicted Dominant Taste")
    plt.ylabel("Derived Dominant Taste Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Plot Top Feature Importances from Random Forest
    rf_clf, _ = fitted_full["RandomForest (150 trees, Balanced)"]
    importances = rf_clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(8.5, 5.5))
    sns.barplot(
        x=importances[top_indices],
        y=[feat_full[i] for i in top_indices],
        hue=[feat_full[i] for i in top_indices], palette="viridis", legend=False,
    )
    plt.title("Top 15 Ingredient Features Driving Taste Classification (Random Forest)")
    plt.xlabel("Mean Decrease in Impurity (Feature Importance)")
    plt.ylabel("Normalized Ingredient")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # Run exploratory similarity retrieval example
    sample_query_id = int(df.iloc[0]["recipe_id"])
    sim_df = retrieve_similar_recipes(df, sample_query_id, top_k=3)

    report_md = f"""# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

## 1. Target Definition & Methodological Disclosure on Circularity
- **Prediction Target:** Recipe `dominant_taste` $\\in \\{{\\text{{sweet}}, \\text{{sour}}, \\text{{bitter}}, \\text{{salty}}, \\text{{umami}}\\}}$.
- **Critical Scientific Disclosure:** Public culinary datasets (Kaggle/Yummly `train.json` and CulinaryDB) do not contain human sensory panel taste ratings at the recipe level. Because `dominant_taste` is derived by aggregating ingredient-level taste annotations in Deliverable 2, training a supervised classifier on the exact same ingredient list (`Full_Ingredients`) measures how effectively a linear or tree-based model **recovers the aggregation rule**, not independent empirical sensory validity.
- **Mitigation & Non-Circular Evaluation (`Masked_Anchors`):** To test genuine contextual pattern learning, we evaluate a second setting in which 12 primary taste-anchor ingredients (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.) are masked from the input features. High performance in `Masked_Anchors` demonstrates that companion ingredients co-occur in predictable culinary assemblies (e.g., vanilla + cinnamon + wheat flour predicting `sweet`).

## 2. Data Split & Leakage Prevention
- **Total Deduplicated Recipes:** {len(df)} (stratified 75% train / 25% test split, `random_state={RANDOM_SEED}`).
- **Leakage Guard:** Recipes with identical normalized ingredient sets were deduplicated prior to splitting so no identical ingredient bag-of-words appears in both train and test partitions.

## 3. Model Comparison Results
{all_results_df.to_markdown(index=False)}

## 4. Exploratory Alternative: Cosine-Similarity Recipe Retrieval
For applications where supervised labels are circular, unsupervised retrieval in the 5D taste space finds recipes with matching sensory balance. Top-3 nearest neighbors for Recipe ID `{sample_query_id}` (`{df.iloc[0]['cuisine']}`, dominant taste `{df.iloc[0]['dominant_taste']}`):

{sim_df.to_markdown(index=False)}
"""
    (REPORTS_DIR / "prediction_evaluation.md").write_text(report_md, encoding="utf-8")
    return all_results_df
