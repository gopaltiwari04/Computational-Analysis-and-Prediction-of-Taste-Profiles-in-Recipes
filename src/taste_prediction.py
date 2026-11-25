"""
Deliverable 4: Lightweight Recipe Taste Prediction Model & Retrieval.
Implements strictly leak-free CV and test evaluation, alias-proof anchor masking,
bootstrap 95% confidence intervals, dual-setting serialization (Full & Masked),
safe inference with unknown-token detection, and cosine similarity retrieval.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    ANCHOR_INTENSITY_THRESHOLD,
    MODELS_DIR,
    RANDOM_SEED,
    RunContext,
    TASTE_DIMENSIONS,
    default_context,
)
from src.data_preprocessing import load_aliases, load_lexicon, normalize_ingredient_name

# Base anchor set matching primary pure/dominant tastants (>= 0.75 intensity or key seasonings)
DIRECT_TASTE_ANCHORS: Set[str] = {
    # Sweeteners
    "sugar",
    "powdered sugar",
    "brown sugar",
    "honey",
    "maple syrup",
    "mango",
    "apple",
    "sweet potato",
    "mirin",
    # Acids & Citrus
    "lemon juice",
    "lime juice",
    "lemon",
    "lime",
    "vinegar",
    "rice vinegar",
    "red wine vinegar",
    "balsamic vinegar",
    "tamarind",
    # Salty / Mineral / Brined
    "salt",
    "salted butter",
    "baking powder",
    "baking soda",
    "feta cheese",
    "capers",
    "olives",
    "bacon",
    # Umami / Fermented / Stocks
    "soy sauce",
    "fish sauce",
    "miso paste",
    "oyster sauce",
    "worcestershire sauce",
    "anchovy",
    "parmesan cheese",
    "tomato paste",
    "chicken broth",
    "beef broth",
    "beef",
    "mushroom",
    "shiitake mushroom",
    "black garlic",
    # Bitter
    "cocoa powder",
    "coffee",
    "fenugreek",
}

# Unmatched surface word patterns belonging to anchor families so unmapped variants cannot bypass masking
ANCHOR_WORD_REGEX = re.compile(
    r"\b("
    r"sugar|honey|syrup|molasses|agave|"
    r"lemon|lime|vinegar|tamarind|"
    r"salt|salted|brine|"
    r"soy|tamari|miso|worcestershire|anchov\w*|parmesan|parmigiano|feta|bacon|"
    r"tomato paste|broth|stock|"
    r"cocoa|chocolate|coffee|espresso|fenugreek|caper|capers|olives|mirin"
    r")\b"
)


def build_anchor_set(lexicon_path: Optional[Path] = None) -> Set[str]:
    """Derive complete canonical anchor set from DIRECT_TASTE_ANCHORS and lexicon intensity threshold."""
    anchors = set(DIRECT_TASTE_ANCHORS)
    path = lexicon_path
    if path is not None and path.exists():
        lex_df = load_lexicon(path)
        for _, row in lex_df.iterrows():
            vals = [row[d] for d in TASTE_DIMENSIONS if pd.notna(row[d])]
            if vals and max(vals) >= ANCHOR_INTENSITY_THRESHOLD:
                anchors.add(str(row["ingredient_norm"]))
    return anchors


def tokenize_ingredients(ingredient_str: str) -> List[str]:
    """Tokenize a semicolon-separated normalized ingredient string for CountVectorizer."""
    if not isinstance(ingredient_str, str):
        return []
    return [t.strip() for t in ingredient_str.split(";") if t.strip()]


def is_anchor_token(token: str, anchors_to_mask: Set[str], aliases: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a normalized token is a direct taste anchor or an alias/surface variant of one,
    preventing unmapped surface strings (e.g. 'dark chocolate', 'coarse kosher salt') from bypassing masking.
    """
    t = token.strip().lower()
    if not t:
        return False
    if t in anchors_to_mask:
        return True
    if aliases is not None:
        canon = normalize_ingredient_name(t, aliases=aliases)
        if canon in anchors_to_mask:
            return True
    # Preserve 'olive oil' (non-brined lipid) while masking brined olives / anchor phrases
    if t == "olive oil":
        return False
    return bool(ANCHOR_WORD_REGEX.search(t))


def mask_ingredient_string(
    ingredient_str: str,
    anchors_to_mask: Set[str],
    aliases: Optional[Dict[str, str]] = None,
) -> str:
    """Remove all direct taste anchors and anchor-family surface variants from a semicolon-separated list."""
    tokens = tokenize_ingredients(ingredient_str)
    remaining = [t for t in tokens if not is_anchor_token(t, anchors_to_mask, aliases=aliases)]
    return ";".join(remaining)


def _bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 200,
    seed: int = RANDOM_SEED,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Percentile 95% bootstrap CIs for test accuracy and macro-F1."""
    n = len(y_true)
    if n < 5:
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return (acc, acc), (f1m, f1m)

    rng = np.random.default_rng(seed)
    accs = []
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs.append(float(accuracy_score(yt, yp)))
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
    return (
        (round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)),
        (round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)),
    )


def train_and_evaluate_models(
    train_corpus: List[str],
    test_corpus: List[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    setting_label: str,
) -> Tuple[List[Dict], Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """
    Strictly leak-free model training and evaluation:
    1. Wraps CountVectorizer + Classifier inside an sklearn Pipeline BEFORE cross-validation
       so CV validation folds never leak into vocabulary construction (min_df).
    2. Fits full training pipeline strictly on train_corpus.
    3. Evaluates held-out test_corpus with 95% bootstrap confidence intervals.
    """
    classifiers = {
        "DummyClassifier (Most Frequent)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "LogisticRegression (L2, Balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "RandomForest (150 trees, Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=14, class_weight="balanced", n_jobs=1, random_state=RANDOM_SEED
        ),
    }

    min_df_val = 2 if len(train_corpus) >= 10 else 1
    class_series = pd.Series(y_train).value_counts()
    min_class_samples = int(class_series.min()) if len(class_series) > 0 else 0
    can_cv = len(y_train) >= 15 and min_class_samples >= 2 and len(class_series) > 1
    n_splits = min(5, min_class_samples) if can_cv else 0

    results = []
    fitted_pipelines: Dict[str, Pipeline] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for name, clf in classifiers.items():
        pipe = Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        tokenizer=tokenize_ingredients,
                        binary=True,
                        token_pattern=None,
                        min_df=min_df_val,
                    ),
                ),
                ("classifier", clf),
            ]
        )

        if can_cv and n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(pipe, train_corpus, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        else:
            cv_mean = 0.0
            cv_std = 0.0

        pipe.fit(train_corpus, y_train)
        preds = pipe.predict(test_corpus)

        acc = float(accuracy_score(y_test, preds))
        f1_mac = float(f1_score(y_test, preds, average="macro", zero_division=0))
        prec_mac = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec_mac = float(recall_score(y_test, preds, average="macro", zero_division=0))
        acc_ci, f1_ci = _bootstrap_classification_ci(y_test, preds, n_boot=200)

        results.append(
            {
                "evaluation_setting": setting_label,
                "model": name,
                "n_train": len(train_corpus),
                "n_test": len(test_corpus),
                "cv_f1_macro_mean": round(cv_mean, 4),
                "cv_f1_macro_std": round(cv_std, 4),
                "test_accuracy": round(acc, 4),
                "test_acc_ci95": f"[{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]",
                "test_macro_f1": round(f1_mac, 4),
                "test_f1_ci95": f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
                "test_macro_precision": round(prec_mac, 4),
                "test_macro_recall": round(rec_mac, 4),
            }
        )

        fitted_pipelines[name] = pipe
        test_predictions[name] = preds

    return results, fitted_pipelines, test_predictions


def predict_recipe_taste(
    ingredients: List[str],
    model_path: Optional[Path] = None,
    setting: str = "full",
) -> Dict[str, object]:
    """
    Standalone inference API on saved end-to-end pipeline bundle.
    Validates input tokens against the fitted training vocabulary and refuses
    to emit a confident prediction when zero normalized tokens match the vocabulary.
    Supports setting='full' (Setting 1) or setting='masked' (Setting 2 contextual model).
    """
    path = model_path or (MODELS_DIR / "taste_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {path}. Run run_deliverable_4() first.")

    artifact = joblib.load(path)
    if setting == "masked" and "masked_pipeline" in artifact:
        pipeline: Pipeline = artifact["masked_pipeline"]
        used_setting = "Setting 2: Masked_Anchors (Contextual Co-occurrence)"
    else:
        pipeline = artifact["pipeline"]
        used_setting = artifact.get("evaluation_setting", "Setting 1: Full_Ingredients")

    classes = [str(c) for c in pipeline.named_steps["classifier"].classes_]
    vocab = pipeline.named_steps["vectorizer"].vocabulary_

    normalized = []
    for raw_ing in ingredients:
        norm = normalize_ingredient_name(raw_ing)
        if norm and norm not in normalized:
            normalized.append(norm)

    if not normalized:
        return {
            "predicted_dominant_taste": "unspecified",
            "probabilities": {c: 0.0 for c in classes},
            "normalized_ingredients": [],
            "known_ingredients": [],
            "unknown_ingredients": [],
            "model_setting": used_setting,
            "is_synthetic_model": bool(artifact.get("is_synthetic", False)),
            "status": "empty_input",
        }

    known = [tok for tok in normalized if tok in vocab]
    unknown = [tok for tok in normalized if tok not in vocab]

    if not known:
        return {
            "predicted_dominant_taste": "unspecified",
            "probabilities": {c: 0.0 for c in classes},
            "normalized_ingredients": normalized,
            "known_ingredients": [],
            "unknown_ingredients": unknown,
            "model_setting": used_setting,
            "is_synthetic_model": bool(artifact.get("is_synthetic", False)),
            "status": "no_known_ingredients",
        }

    input_text = ";".join(sorted(known))
    pred_label = str(pipeline.predict([input_text])[0])

    probs: Dict[str, float] = {}
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        prob_vec = pipeline.predict_proba([input_text])[0]
        for c, p in zip(classes, prob_vec):
            probs[c] = round(float(p), 4)

    return {
        "predicted_dominant_taste": pred_label,
        "probabilities": probs,
        "normalized_ingredients": normalized,
        "known_ingredients": known,
        "unknown_ingredients": unknown,
        "model_setting": used_setting,
        "is_synthetic_model": bool(artifact.get("is_synthetic", False)),
        "status": "success",
    }


def retrieve_similar_recipes(
    df: pd.DataFrame,
    query_recipe_id: int,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Unsupervised cosine-similarity retrieval in 5D normalized taste simplex space.
    - Resets positional index safely so filtered DataFrames never mismatch iloc vs .index.
    - Guarantees query_recipe_id is NEVER returned as its own nearest neighbor.
    - Note: Proximity in modeled proxy taste space does not imply sensory equivalence.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    if df.empty or len(df) < 2:
        raise ValueError("Dataset must contain at least 2 recipes for similarity retrieval.")

    work_df = df.reset_index(drop=True)
    valid_df = work_df[work_df["dominant_taste"] != "unspecified"].reset_index(drop=True)
    if valid_df.empty or len(valid_df) < 2:
        raise ValueError("Dataset contains fewer than 2 non-zero taste profiles for cosine similarity.")

    matches = valid_df.index[valid_df["recipe_id"] == query_recipe_id].tolist()
    if not matches:
        raise ValueError(f"Recipe ID {query_recipe_id} not found (or has unspecified zero taste profile).")

    query_pos = int(matches[0])
    X_taste = valid_df[TASTE_DIMENSIONS].to_numpy(dtype=float)

    n_neighbors = min(top_k + 5, len(valid_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X_taste)

    distances, indices = nn.kneighbors(X_taste[query_pos : query_pos + 1])
    evenness_col = "taste_evenness" if "taste_evenness" in valid_df.columns else "taste_balance"

    results = []
    for dist, pos in zip(distances[0], indices[0]):
        row = valid_df.iloc[int(pos)]
        cand_id = int(row["recipe_id"])
        if cand_id == query_recipe_id:
            continue

        sim = max(-1.0, min(1.0, float(1.0 - dist)))
        results.append(
            {
                "neighbor_recipe_id": cand_id,
                "cuisine": row["cuisine"],
                "dominant_taste": row["dominant_taste"],
                "taste_evenness": round(float(row[evenness_col]), 4) if evenness_col in row else np.nan,
                "cosine_similarity": round(sim, 4),
            }
        )
        if len(results) >= top_k:
            break

    return pd.DataFrame(results)


def _serialize_linear_model(pipe: Pipeline) -> Dict[str, object]:
    """Extract compact JSON-serializable weights from a CountVectorizer + LogisticRegression pipeline."""
    vec: CountVectorizer = pipe.named_steps["vectorizer"]
    clf: LogisticRegression = pipe.named_steps["classifier"]
    inv_vocab = {idx: tok for tok, idx in vec.vocabulary_.items()}
    coef = clf.coef_
    weights = {
        inv_vocab[idx]: [round(float(v), 8) for v in coef[:, idx]]
        for idx in range(len(inv_vocab))
    }
    return {
        "classes": [str(c) for c in clf.classes_],
        "intercept": [round(float(v), 8) for v in clf.intercept_],
        "weights": weights,
    }


def export_model_weights_json(bundle_path: Path, output_json_path: Path) -> Dict[str, object]:
    """Export lightweight pure-JSON logistic regression weights for serverless runtimes."""
    bundle = joblib.load(bundle_path)
    payload: Dict[str, object] = {
        "is_synthetic": bool(bundle.get("is_synthetic", False)),
        "dataset_name": bundle.get("dataset_name", "Unknown"),
        "trained_at": bundle.get("trained_at", "unknown"),
        "anchor_tokens": bundle.get("anchor_tokens", []),
        "full": _serialize_linear_model(bundle["pipeline"]),
    }
    if "masked_pipeline" in bundle:
        payload["masked"] = _serialize_linear_model(bundle["masked_pipeline"])
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def _load_manifest(context: RunContext) -> Dict[str, object]:
    if context.manifest_path.exists():
        with open(context.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"is_synthetic": False, "dataset_name": "Unknown Recipe Corpus"}


def run_deliverable_4(ctx: Optional[RunContext] = None) -> pd.DataFrame:
    """
    Execute Deliverable 4:
    1. Loads recipe taste features and excludes zero-vector 'unspecified' profiles.
    2. Deduplicates recipes by normalized ingredient bag prior to partitioning.
    3. Splits into stratified 75% train / 25% test (or unstratified fallback on tiny fixtures).
    4. Evaluates Setting 1 (Full_Ingredients — algebraic rule recovery) and Setting 2
       (Masked_Anchors — contextual co-occurrence after removing anchor tokens & post-masking collisions).
    5. Serializes both Setting 1 and Setting 2 Pipelines into models/taste_model.joblib.
    6. Saves dual confusion matrices, feature importances, model_comparison.csv, and prediction_evaluation.md.
    """
    context = ctx or default_context()
    context.ensure()
    manifest = _load_manifest(context)
    is_synthetic = bool(manifest.get("is_synthetic", False))
    corpus_tag = "SYNTHETIC SMOKE TEST" if is_synthetic else "REAL CORPUS"

    raw_df = pd.read_csv(context.processed_dir / "recipe_taste_features.csv")
    df = raw_df[raw_df["dominant_taste"] != "unspecified"].copy()
    df = df.drop_duplicates(subset=["ingredients_normalized"]).reset_index(drop=True)

    # Filter rare classes with fewer than 5 instances
    class_counts_raw = df["dominant_taste"].value_counts()
    valid_classes = class_counts_raw[class_counts_raw >= 5].index.tolist()
    df = df[df["dominant_taste"].isin(valid_classes)].reset_index(drop=True)
    if len(df) < 10 or df["dominant_taste"].nunique() < 2:
        raise ValueError("Insufficient recipes or classes (< 2 classes with >= 5 samples) for Deliverable 4.")

    y = df["dominant_taste"].to_numpy()
    corpus_full = df["ingredients_normalized"].tolist()

    min_class_count = int(pd.Series(y).value_counts().min())
    stratify_arg = y if min_class_count >= 2 else None
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=stratify_arg,
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
        setting_label="Setting 1: Full_Ingredients (Rule Recovery)",
    )

    # SETTING 2: Masked Direct Taste Anchors (Contextual Co-occurrence)
    anchors = build_anchor_set(context.lexicon_path)
    aliases = load_aliases(context.aliases_path)
    corpus_masked = [mask_ingredient_string(s, anchors, aliases=aliases) for s in corpus_full]
    valid_mask_filter = np.array([len(tokenize_ingredients(s)) >= 2 for s in corpus_masked])

    train_mask_idx = [i for i in train_idx if valid_mask_filter[i]]
    raw_test_mask_idx = [i for i in test_idx if valid_mask_filter[i]]

    # Prevent post-masking collisions from crossing train/test boundary
    train_masked_bags = {corpus_masked[i] for i in train_mask_idx}
    test_mask_idx = []
    seen_test_bags = set()
    post_mask_collisions_removed = 0
    for i in raw_test_mask_idx:
        bag = corpus_masked[i]
        if bag in train_masked_bags or bag in seen_test_bags:
            post_mask_collisions_removed += 1
            continue
        seen_test_bags.add(bag)
        test_mask_idx.append(i)

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
    all_results_df.to_csv(context.results_dir / "model_comparison.csv", index=False)

    # Serialize both Full and Masked pipelines together with provenance
    best_pipe = pipelines_full["LogisticRegression (L2, Balanced)"]
    masked_pipe = pipelines_masked["LogisticRegression (L2, Balanced)"]
    model_bundle_path = context.models_dir / "taste_model.joblib"
    joblib.dump(
        {
            "pipeline": best_pipe,
            "masked_pipeline": masked_pipe,
            "classes": list(best_pipe.named_steps["classifier"].classes_),
            "evaluation_setting": "Setting 1: Full_Ingredients",
            "random_seed": RANDOM_SEED,
            "vocabulary_size": len(best_pipe.named_steps["vectorizer"].vocabulary_),
            "masked_vocabulary_size": len(masked_pipe.named_steps["vectorizer"].vocabulary_),
            "anchor_count": len(anchors),
            "anchor_tokens": sorted(anchors),
            "is_synthetic": is_synthetic,
            "dataset_name": manifest.get("dataset_name", "Unknown"),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        model_bundle_path,
    )
    export_model_weights_json(model_bundle_path, context.models_dir / "taste_model_weights.json")

    # Visualization 1: Side-by-side Confusion Matrices (Setting 1 vs Setting 2)
    classes_ordered = sorted(list(set(y_test) | set(y_test_mask)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    cm1 = confusion_matrix(y_test, preds_full["LogisticRegression (L2, Balanced)"], labels=classes_ordered)
    sns.heatmap(
        cm1,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in classes_ordered],
        yticklabels=[c.capitalize() for c in classes_ordered],
        ax=axes[0],
    )
    axes[0].set_title(f"Setting 1: Full Ingredients\n(Rule Recovery, N_test={len(y_test):,})")
    axes[0].set_xlabel("Predicted Modality")
    axes[0].set_ylabel("Rule-Derived Label")

    cm2 = confusion_matrix(y_test_mask, preds_masked["LogisticRegression (L2, Balanced)"], labels=classes_ordered)
    sns.heatmap(
        cm2,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[c.capitalize() for c in classes_ordered],
        yticklabels=[c.capitalize() for c in classes_ordered],
        ax=axes[1],
    )
    axes[1].set_title(f"Setting 2: Masked Anchors\n(Co-occurrence Context, N_test={len(y_test_mask):,})")
    axes[1].set_xlabel("Predicted Modality")
    axes[1].set_ylabel("Rule-Derived Label")

    fig.suptitle(f"Held-Out Test Confusion Matrices — Logistic Regression [{corpus_tag}]", fontsize=12)
    plt.tight_layout()
    plt.savefig(context.results_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Visualization 2: Top Random Forest Feature Importances (Full vs Masked)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, pipe_dict, title_sub in [
        (axes[0], pipelines_full, "Setting 1: Full Ingredients"),
        (axes[1], pipelines_masked, "Setting 2: Masked Anchors"),
    ]:
        rf_pipe = pipe_dict["RandomForest (150 trees, Balanced)"]
        rf_clf: RandomForestClassifier = rf_pipe.named_steps["classifier"]
        vec: CountVectorizer = rf_pipe.named_steps["vectorizer"]
        feat_names = vec.get_feature_names_out()
        importances = rf_clf.feature_importances_
        top_k = min(15, len(feat_names))
        top_indices = np.argsort(importances)[::-1][:top_k]
        sns.barplot(
            x=importances[top_indices],
            y=[feat_names[i] for i in top_indices],
            hue=[feat_names[i] for i in top_indices],
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title(title_sub)
        ax.set_xlabel("Mean Decrease in Impurity (MDI)")
        ax.set_ylabel("Normalized Ingredient")

    fig.suptitle(
        f"Top 15 Ingredient Features (Random Forest MDI — Interpret Cautiously Under Collinearity) [{corpus_tag}]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(context.results_dir / "feature_importance.png", dpi=200)
    plt.close()

    # Similarity retrieval demonstration
    sample_id = int(df.iloc[0]["recipe_id"])
    sim_df = retrieve_similar_recipes(df, query_recipe_id=sample_id, top_k=3)

    clf_report_full = classification_report(
        y_test, preds_full["LogisticRegression (L2, Balanced)"], labels=classes_ordered, zero_division=0
    )
    clf_report_masked = classification_report(
        y_test_mask, preds_masked["LogisticRegression (L2, Balanced)"], labels=classes_ordered, zero_division=0
    )

    class_dist_full = pd.Series(y).value_counts().to_dict()
    class_dist_masked = pd.Series(y[valid_mask_filter]).value_counts().to_dict()
    class_dist_df = pd.DataFrame(
        [
            {
                "dominant_taste": cls,
                "setting1_count": int(class_dist_full.get(cls, 0)),
                "setting1_pct": round(class_dist_full.get(cls, 0) / len(y) * 100.0, 2),
                "setting2_retained_count": int(class_dist_masked.get(cls, 0)),
                "setting2_pct": round(
                    class_dist_masked.get(cls, 0) / max(1, int(valid_mask_filter.sum())) * 100.0, 2
                ),
            }
            for cls in sorted(valid_classes)
        ]
    )

    corpus_banner = (
        "> **Corpus Notice:** Evaluated on **SYNTHETIC SMOKE TEST (`--use-sample-data`)** — "
        "results validate pipeline execution only, NOT real-world culinary predictability."
        if is_synthetic
        else "> **Corpus Notice:** Evaluated on **REAL EMPIRICAL CORPUS** (`"
        + str(manifest.get("dataset_name", "External Corpus"))
        + "`)."
    )

    report_md = f"""# Deliverable 4: Lightweight Recipe Taste Prediction Evaluation

{corpus_banner}

## 1. Three Distinct Evaluation Tasks & Circularity Disclosure
We explicitly separate three fundamentally different modeling tasks:
1. **Setting 1 — Full-Ingredient Rule Recovery (`Full_Ingredients`):** Classifies the rule-derived `dominant_taste` label from the full ingredient bag. High performance reflects linear/non-linear **recovery of the algebraic lexicon aggregation rule**, not empirical human taste prediction.
2. **Setting 2 — Masked-Anchor Co-Occurrence Prediction (`Masked_Anchors`):** Removes **{len(anchors)} canonical taste-anchor ingredients** (all lexicon entries with intensity $\\ge {ANCHOR_INTENSITY_THRESHOLD}$ plus direct acid/mineral/umami/bitter anchors) AND regex-matched anchor-family surface variants (`ANCHOR_WORD_REGEX`, preventing alias bypass). Performance reflects how well remaining companion ingredients (e.g., herbs, alliums, flours, fats, base proteins) predict the **proxy label via culinary co-occurrence**, NOT independently measured human sensory perception.
3. **Genuine Empirical Sensory Prediction (Unverified / Absent):** Public recipe corpora (`train.json`) contain **zero** human sensory panel taste ratings; independent empirical validation against human perception remains outside the available data.

## 2. Leak-Free Protocol, Class Balance & Post-Masking Accounting
- **Random Seed:** `{RANDOM_SEED}` | **Partition:** 75% Train ($N_{{train}} = {len(train_full):,}$) / 25% Held-Out Test ($N_{{test}} = {len(test_full):,}$)
- **Pre-Split Deduplication:** Exact normalized ingredient bags deduplicated prior to splitting ({len(df):,} unique recipe bags).
- **Pipeline CV Integrity:** `CountVectorizer` (`min_df={2 if len(train_full) >= 10 else 1}`) is nested inside the `sklearn.pipeline.Pipeline` during 5-fold Stratified CV so validation folds never influence vocabulary filtering.
- **Post-Masking Retention:** {int(valid_mask_filter.sum()):,} / {len(df):,} recipes (**{valid_mask_filter.mean()*100:.2f}%**) retained $\\ge 2$ non-anchor ingredients (`Setting 2` train = {len(train_masked):,}, test = {len(test_masked):,} after excluding **{post_mask_collisions_removed:,}** post-masking duplicate collisions across test/train).

### Class Balance (Setting 1 vs. Post-Masking Setting 2)
{class_dist_df.to_markdown(index=False)}

## 3. Benchmark Model Comparison Results (with 95% Bootstrap CIs)
{all_results_df.to_markdown(index=False)}

### Per-Class Held-Out Test Breakdown — Setting 1 (Logistic Regression)
```
{clf_report_full}
```

### Per-Class Held-Out Test Breakdown — Setting 2 Masked Anchors (Logistic Regression)
```
{clf_report_masked}
```

*(Note on Feature Importances: Random Forest MDI bars in `results/feature_importance.png` distribute importance across correlated co-occurring ingredients and should be interpreted cautiously as associative rather than causal indicators.)*

## 4. Alternative: Unsupervised Taste Simplex Retrieval
Cosine similarity in the 5-dimensional modeled taste simplex retrieves recipes with matching taste-share geometry (not proof of identical plated sensory experience):
- **Query Recipe ID:** `{sample_id}` ({df.iloc[0]['cuisine']}, Dominant: `{df.iloc[0]['dominant_taste']}`)
- **Top-3 Nearest Neighbors:**
{sim_df.to_markdown(index=False)}

*(Invariant Verified: Query recipe `{sample_id}` is strictly excluded from its own neighbors.)*
"""
    (context.reports_dir / "prediction_evaluation.md").write_text(report_md, encoding="utf-8")
    return all_results_df
