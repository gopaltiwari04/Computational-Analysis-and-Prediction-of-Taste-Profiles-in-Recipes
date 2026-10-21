"""
Deliverable 4: Lightweight Recipe Taste Prediction Model.
Implements leak-free preprocessing pipelines, DummyClassifier baseline,
Logistic Regression, Random Forest, non-circular masked-anchor evaluation,
model serialization with end-to-end inference, and similarity-based retrieval.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_LEXICON_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    TASTE_DIMENSIONS,
    ensure_directories,
)
from src.data_preprocessing import normalize_ingredient_name

# Primary direct taste anchors systematically masked in Setting 2 (Contextual Co-occurrence).
# Covers pure tastants and high-intensity single/double-modality markers across all 5 dimensions.
DIRECT_TASTE_ANCHORS = {
    # Sweet anchors
    "sugar",
    "brown sugar",
    "honey",
    "maple syrup",
    # Sour / acid anchors
    "lemon juice",
    "lime juice",
    "vinegar",
    "rice vinegar",
    "tamarind",
    # Salty / mineral anchors
    "salt",
    "baking powder",
    # Umami / fermented anchors
    "soy sauce",
    "fish sauce",
    "miso paste",
    "oyster sauce",
    "anchovy",
    "parmesan cheese",
    "tomato paste",
    # Bitter anchors
    "cocoa powder",
    "coffee",
    "fenugreek",
}


def tokenize_ingredients(ingredient_str: str) -> List[str]:
    """Tokenize a semicolon-separated normalized ingredient string for CountVectorizer."""
    if not isinstance(ingredient_str, str):
        return []
    return [t.strip() for t in ingredient_str.split(";") if t.strip()]


def mask_ingredient_string(ingredient_str: str, anchors_to_mask: set) -> str:
    """Mask specified direct taste anchor ingredients from a semicolon-separated ingredient string."""
    tokens = tokenize_ingredients(ingredient_str)
    remaining = [t for t in tokens if t not in anchors_to_mask]
    return ";".join(remaining)


def train_and_evaluate_models(
    train_corpus: List[str],
    test_corpus: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    setting_label: str,
) -> Tuple[List[Dict], Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """
    Leak-free model training and evaluation:
    1. Fits CountVectorizer ONLY on train_corpus.
    2. Builds end-to-end Pipelines [('vectorizer', CountVectorizer), ('classifier', clf)].
    3. Computes 5-fold Stratified Cross-Validation on the training partition.
    4. Evaluates held-out test split (Accuracy, Macro-F1, Macro-Precision, Macro-Recall).
    Returns result rows, fitted pipelines dictionary, and test predictions dictionary.
    """
    models = {
        "DummyClassifier (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "LogisticRegression (L2, Balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "RandomForest (150 trees, Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=14, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
    }

    # Vectorizer fitted ONLY on train data using top-level picklable tokenizer
    vectorizer = CountVectorizer(
        tokenizer=tokenize_ingredients,
        binary=True,
        token_pattern=None,
        min_df=2,
    )
    vectorizer.fit(train_corpus)

    X_train = vectorizer.transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)

    # Determine if dataset is large enough for 5-fold cross validation
    class_series = pd.Series(y_train).value_counts()
    min_class_samples = int(class_series.min()) if len(class_series) > 0 else 0
    can_cv = len(y_train) >= 15 and min_class_samples >= 2 and len(class_series) > 1
    n_splits = min(5, min_class_samples) if can_cv else 0

    results = []
    fitted_pipelines = {}
    test_predictions = {}

    for name, clf in models.items():
        if can_cv and n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        else:
            cv_mean = 0.0
            cv_std = 0.0

        # Fit on full training split
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1_mac = float(f1_score(y_test, preds, average="macro", zero_division=0))
        prec_mac = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec_mac = float(recall_score(y_test, preds, average="macro", zero_division=0))

        results.append(
            {
                "evaluation_setting": setting_label,
                "model": name,
                "cv_f1_macro_mean": round(cv_mean, 4),
                "cv_f1_macro_std": round(cv_std, 4),
                "test_accuracy": round(acc, 4),
                "test_macro_f1": round(f1_mac, 4),
                "test_macro_precision": round(prec_mac, 4),
                "test_macro_recall": round(rec_mac, 4),
            }
        )

        # Store complete leak-free pipeline
        pipe = Pipeline([("vectorizer", vectorizer), ("classifier", clf)])
        fitted_pipelines[name] = pipe
        test_predictions[name] = preds

    return results, fitted_pipelines, test_predictions


def predict_recipe_taste(
    ingredients: List[str],
    model_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Standalone, production-grade inference API:
    Accepts arbitrary list of raw ingredient strings, normalizes them,
    transforms via the fitted pipeline, and returns predicted dominant taste,
    per-class probability distribution, and matched ingredient breakdown.
    """
    path = model_path or (MODELS_DIR / "taste_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {path}. Run run_deliverable_4() first.")

    artifact = joblib.load(path)
    pipeline: Pipeline = artifact["pipeline"]
    classes = list(pipeline.named_steps["classifier"].classes_)

    # Normalize input strings
    normalized = []
    for raw_ing in ingredients:
        norm = normalize_ingredient_name(raw_ing)
        if norm:
            normalized.append(norm)

    if not normalized:
        return {
            "predicted_dominant_taste": "unspecified",
            "probabilities": {c: 0.0 for c in classes},
            "normalized_ingredients": [],
            "status": "empty_input",
        }

    input_text = ";".join(sorted(set(normalized)))
    pred_label = str(pipeline.predict([input_text])[0])

    probs = {}
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        prob_vec = pipeline.predict_proba([input_text])[0]
        for c, p in zip(classes, prob_vec):
            probs[c] = round(float(p), 4)

    return {
        "predicted_dominant_taste": pred_label,
        "probabilities": probs,
        "normalized_ingredients": normalized,
        "model_setting": artifact.get("evaluation_setting", "unknown"),
        "status": "success",
    }


def retrieve_similar_recipes(
    df: pd.DataFrame,
    query_recipe_id: int,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Unsupervised retrieval in 5D normalized taste simplex space.
    Strictly verifies that query recipe is NEVER returned as its own neighbor.
    Handles missing IDs and small corpora safely.
    """
    if query_recipe_id not in df["recipe_id"].values:
        raise ValueError(f"Recipe ID {query_recipe_id} not found in recipe dataset.")

    X_taste = df[TASTE_DIMENSIONS].to_numpy(dtype=float)
    # Fit cosine distance nearest neighbors
    n_neighbors = min(top_k + 5, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X_taste)

    query_mask = df["recipe_id"] == query_recipe_id
    query_idx = int(df.index[query_mask][0])
    query_vector = X_taste[query_idx : query_idx + 1]

    distances, indices = nn.kneighbors(query_vector)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        cand_id = int(df.iloc[idx]["recipe_id"])
        # Invariant: Never return the query recipe itself
        if cand_id == query_recipe_id:
            continue

        row = df.iloc[idx]
        sim = float(1.0 - dist)
        results.append(
            {
                "neighbor_recipe_id": cand_id,
                "cuisine": row["cuisine"],
                "dominant_taste": row["dominant_taste"],
                "taste_balance": round(float(row["taste_balance"]), 4),
                "cosine_similarity": round(sim, 4),
            }
        )
        if len(results) >= top_k:
            break

    return pd.DataFrame(results)


def run_deliverable_4() -> pd.DataFrame:
    """
    Execute Deliverable 4:
    1. Loads recipe taste features.
    2. Deduplicates recipes by normalized ingredient bag-of-words to prevent train/test leakage.
    3. Evaluates Setting 1: Full_Ingredients (Rule Recovery baseline).
    4. Evaluates Setting 2: Masked_Anchors (Contextual Co-occurrence Generalization).
    5. Serializes complete Pipeline to models/taste_model.joblib.
    6. Generates confusion matrix and feature importance plots.
    7. Performs cosine-similarity recipe retrieval.
    8. Writes results/model_comparison.csv and reports/prediction_evaluation.md.
    """
    ensure_directories()

    df = pd.read_csv(DATA_PROCESSED_DIR / "recipe_taste_features.csv")

    # 1. Deduplicate by normalized ingredient signature so no identical ingredient bag splits across train/test
    df = df.drop_duplicates(subset=["ingredients_normalized"]).reset_index(drop=True)

    # 2. Filter rare classes with fewer than 5 instances
    y_raw = df["dominant_taste"].to_numpy()
    class_counts = pd.Series(y_raw).value_counts()
    valid_classes = class_counts[class_counts >= 5].index.tolist()
    df = df[df["dominant_taste"].isin(valid_classes)].reset_index(drop=True)

    y = df["dominant_taste"].to_numpy()
    corpus_full = df["ingredients_normalized"].tolist()

    # Stratified train/test partition (75% train, 25% test)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_full = [corpus_full[i] for i in train_idx]
    test_full = [corpus_full[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # SETTING 1: Full Ingredient Features (Circular Rule Recovery)
    rows_full, pipelines_full, preds_full = train_and_evaluate_models(
        train_corpus=train_full,
        test_corpus=test_full,
        y_train=y_train,
        y_test=y_test,
        setting_label="Setting 1: Full_Ingredients (Circular Rule Recovery)",
    )

    # SETTING 2: Masked Direct Taste Anchors (Contextual Generalization)
    corpus_masked = [mask_ingredient_string(s, DIRECT_TASTE_ANCHORS) for s in corpus_full]
    # Retain recipes that maintain at least 2 non-masked ingredients
    valid_mask_filter = np.array([len([t for t in s.split(";") if t.strip()]) >= 2 for s in corpus_masked])

    train_mask_idx = [i for i in train_idx if valid_mask_filter[i]]
    test_mask_idx = [i for i in test_idx if valid_mask_filter[i]]

    train_masked = [corpus_masked[i] for i in train_mask_idx]
    test_masked = [corpus_masked[i] for i in test_mask_idx]
    y_train_mask = y[train_mask_idx]
    y_test_mask = y[test_mask_idx]

    rows_masked, pipelines_masked, preds_masked = train_and_evaluate_models(
        train_corpus=train_masked,
        test_corpus=test_masked,
        y_train=y_train_mask,
        y_test=y_test_mask,
        setting_label="Setting 2: Masked_Anchors (Contextual Co-occurrence)",
    )

    all_results_df = pd.DataFrame(rows_full + rows_masked)
    all_results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Serialize best production model pipeline (Logistic Regression, Full Setting)
    best_pipe = pipelines_full["LogisticRegression (L2, Balanced)"]
    joblib.dump(
        {
            "pipeline": best_pipe,
            "classes": list(best_pipe.named_steps["classifier"].classes_),
            "evaluation_setting": "Full_Ingredients",
            "random_seed": RANDOM_SEED,
            "vocabulary_size": len(best_pipe.named_steps["vectorizer"].vocabulary_),
        },
        MODELS_DIR / "taste_model.joblib",
    )

    # Visualization 1: Confusion Matrix on Test Split (Logistic Regression, Full Setting)
    classes_ordered = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, preds_full["LogisticRegression (L2, Balanced)"], labels=classes_ordered)
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in classes_ordered],
        yticklabels=[c.capitalize() for c in classes_ordered],
    )
    plt.title("Held-Out Test Split Confusion Matrix (Logistic Regression)")
    plt.xlabel("Predicted Dominant Taste Modality")
    plt.ylabel("Derived Dominant Taste Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Visualization 2: Top Feature Importances from Random Forest (Full Setting)
    rf_pipe = pipelines_full["RandomForest (150 trees, Balanced)"]
    rf_clf: RandomForestClassifier = rf_pipe.named_steps["classifier"]
    vec: CountVectorizer = rf_pipe.named_steps["vectorizer"]
    feat_names = vec.get_feature_names_out()
    importances = rf_clf.feature_importances_
    top_k = min(15, len(feat_names))
    top_indices = np.argsort(importances)[::-1][:top_k]

    plt.figure(figsize=(8.5, 6))
    sns.barplot(
        x=importances[top_indices],
        y=[feat_names[i] for i in top_indices],
        hue=[feat_names[i] for i in top_indices],
        palette="viridis",
        legend=False,
    )
    plt.title("Top 15 Ingredient Features Driving Taste Classification (Random Forest)")
    plt.xlabel("Mean Decrease in Impurity (Feature Importance)")
    plt.ylabel("Normalized Ingredient")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # Similarity Retrieval Demonstration
    sample_id = int(df.iloc[0]["recipe_id"])
    sim_df = retrieve_similar_recipes(df, query_recipe_id=sample_id, top_k=3)

    # Detailed Classification Report for Logistic Regression on Test Set
    clf_report = classification_report(
        y_test,
        preds_full["LogisticRegression (L2, Balanced)"],
        labels=classes_ordered,
        zero_division=0,
    )

    report_md = f"""# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

## 1. Prediction Target & Critical Methodological Disclosure on Circularity
- **Prediction Target:** Recipe `dominant_taste` $\\in \\{{\\text{{sweet}}, \\text{{sour}}, \\text{{bitter}}, \\text{{salty}}, \\text{{umami}}\\}}$.
- **Circularity Disclosure:** Public recipe corpora (e.g. Kaggle/Yummly `train.json`) do not contain sensory panel taste ratings. Because `dominant_taste` is derived by aggregating ingredient-level scores in Deliverable 2, supervised learning on the full ingredient list (**Setting 1**) tests the model's ability to **recover the algebraic aggregation rule**, not independent empirical human taste perception.
- **Contextual Generalization Protocol (**Setting 2: Masked_Anchors**):** To test non-circular culinary structure, we mask {len(DIRECT_TASTE_ANCHORS)} primary taste-anchor ingredients (`salt`, `sugar`, `lemon juice`, `soy sauce`, `fish sauce`, `cocoa powder`, etc.). In Setting 2, the classifier must infer the dish's taste orientation entirely from companion ingredients (e.g., vanilla + flour + baking powder predicting `sweet`; cumin + turmeric + ginger predicting `bitter`/savory).
- **Proportion of Recipes Retaining Useful Context:** {int(valid_mask_filter.sum()):,} / {len(df):,} recipes ({valid_mask_filter.mean()*100:.1f}%) retained $\\ge 2$ informative ingredients after masking.

## 2. Leak-Free Evaluation Methodology
- **Deduplication:** Recipes sharing identical normalized ingredient bags were deduplicated prior to partitioning, guaranteeing that identical recipes never cross train/test boundaries.
- **Leak-Free Preprocessing:** `CountVectorizer` was fitted strictly on the training partition ($75\\%$, $N_{{train}} = {len(train_full):,}$); the held-out test partition ($25\\%$, $N_{{test}} = {len(test_full):,}$) was transformed using the training vocabulary.
- **Cross-Validation:** 5-fold Stratified Cross-Validation was performed across the training set to evaluate generalization stability (mean $\\pm$ standard deviation macro-F1).

## 3. Benchmark Model Comparison Results
{all_results_df.to_markdown(index=False)}

### Per-Class Test Set Breakdown (Logistic Regression, Setting 1)
```
{clf_report}
```

## 4. Alternative: Unsupervised Taste Simplex Retrieval
When supervised labels are circular or categorical targets are too coarse, unsupervised retrieval in the continuous 5-dimensional taste simplex space finds culinary neighbors with matching taste distribution:
- **Query Recipe ID:** `{sample_id}` ({df.iloc[0]['cuisine']}, Dominant: `{df.iloc[0]['dominant_taste']}`)
- **Top-3 Nearest Neighbors:**
{sim_df.to_markdown(index=False)}

*(Invariant Verified: Query recipe `{sample_id}` is strictly excluded from its own nearest neighbors.)*
"""
    (REPORTS_DIR / "prediction_evaluation.md").write_text(report_md, encoding="utf-8")
    return all_results_df
