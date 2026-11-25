"""
Zero-heavy-dependency pure-Python inference, normalization, simplex math, and
similarity engine for lightweight serverless deployment (e.g. Vercel).
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import ANCHOR_INTENSITY_THRESHOLD, TASTE_DIMENSIONS

PREP_STOPWORDS = {
    "chopped", "diced", "sliced", "minced", "crushed", "peeled", "seeded",
    "halved", "quartered", "finely", "coarsely", "freshly", "thinly", "thickly",
    "organic", "large", "medium", "small", "fresh", "grated", "shredded",
    "cooked", "uncooked", "prepared", "optional", "canned", "drained", "rinsed",
    "cup", "cups", "tbsp", "tsp", "tablespoon", "tablespoons", "teaspoon",
    "teaspoons", "ounce", "ounces", "oz", "lb", "lbs", "pound", "pounds",
    "pinch", "dash", "bunch", "slice", "slices", "can", "cans",
}

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

BASE_ANCHORS: Set[str] = {
    "sugar", "powdered sugar", "brown sugar", "honey", "maple syrup",
    "mango", "apple", "sweet potato", "mirin", "lemon juice", "lime juice",
    "lemon", "lime", "vinegar", "rice vinegar", "red wine vinegar",
    "balsamic vinegar", "tamarind", "salt", "salted butter", "baking powder",
    "baking soda", "feta cheese", "capers", "olives", "bacon", "soy sauce",
    "fish sauce", "miso paste", "oyster sauce", "worcestershire sauce",
    "anchovy", "parmesan cheese", "tomato paste", "chicken broth",
    "beef broth", "beef", "mushroom", "shiitake mushroom", "black garlic",
    "cocoa powder", "coffee", "fenugreek",
}


def _parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "na", "none", "null", "unknown"}:
        return None
    try:
        num = float(s)
        return None if (math.isnan(num) or math.isinf(num)) else num
    except ValueError:
        return None


def load_aliases_csv(path: Path) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if not path.exists():
        return aliases
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            surf = str(row.get("surface_form", "")).strip().lower()
            canon = str(row.get("canonical", "")).strip().lower()
            if surf and canon:
                aliases[surf] = canon
    return aliases


def load_lexicon_csv(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: Dict[str, Any] = dict(row)
            for dim in TASTE_DIMENSIONS:
                parsed[dim] = _parse_float(row.get(dim))
            entries.append(parsed)
    return entries


def load_csv_records(path: Path, numeric_cols: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    num_set = numeric_cols or set()
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            item: Dict[str, Any] = {}
            for k, v in row.items():
                if k in num_set:
                    item[k] = _parse_float(v)
                else:
                    item[k] = v
            rows.append(item)
    return rows


def normalize_ingredient(raw_name: str, aliases: Dict[str, str]) -> str:
    if not isinstance(raw_name, str):
        return ""
    text = raw_name.lower().strip()
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in aliases:
        return aliases[text]
    tokens = [tok for tok in text.split() if tok not in PREP_STOPWORDS]
    cleaned = " ".join(tokens) if tokens else text
    return aliases.get(cleaned, cleaned)


def derive_anchors(lex_rows: List[Dict[str, Any]]) -> Set[str]:
    anchors = set(BASE_ANCHORS)
    for row in lex_rows:
        vals = [row[d] for d in TASTE_DIMENSIONS if row[d] is not None]
        if vals and max(vals) >= ANCHOR_INTENSITY_THRESHOLD:
            anchors.add(str(row["ingredient_norm"]))
    return anchors


def is_anchor(token: str, anchors: Set[str], aliases: Dict[str, str]) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    if t in anchors:
        return True
    canon = normalize_ingredient(t, aliases)
    if canon in anchors:
        return True
    if t == "olive oil":
        return False
    return bool(ANCHOR_WORD_REGEX.search(t))


def compute_lexicon_profile(taste_rows: List[List[Optional[float]]]) -> Dict[str, Any]:
    if not taste_rows:
        return {
            "n_taste_matched": 0,
            "n_known_dimensions": 0,
            "raw_mean_vector": {d: None for d in TASTE_DIMENSIONS},
            "normalized_simplex": {d: 0.0 for d in TASTE_DIMENSIONS},
            "dominant_taste": "unspecified",
            "shannon_entropy_nats": 0.0,
            "taste_evenness_j": 0.0,
        }

    raw_vec: List[Optional[float]] = []
    for col in range(len(TASTE_DIMENSIONS)):
        vals = [row[col] for row in taste_rows if row[col] is not None]
        raw_vec.append(sum(vals) / len(vals) if vals else None)

    n_known_dims = sum(1 for v in raw_vec if v is not None)
    zeroed = [0.0 if v is None else max(0.0, v) for v in raw_vec]
    total = sum(zeroed)
    if total > 1e-9:
        simplex = [v / total for v in zeroed]
        dom_idx = max(range(len(simplex)), key=lambda i: simplex[i])
        dom_taste = TASTE_DIMENSIONS[dom_idx]
    else:
        simplex = [0.0] * len(TASTE_DIMENSIONS)
        dom_taste = "unspecified"

    positives = [p for p in simplex if p > 1e-12]
    if len(positives) <= 1:
        entropy, evenness = 0.0, 0.0
    else:
        entropy = -sum(p * math.log(p) for p in positives)
        evenness = min(1.0, max(0.0, entropy / math.log(len(TASTE_DIMENSIONS))))

    return {
        "n_taste_matched": len(taste_rows),
        "n_known_dimensions": n_known_dims,
        "raw_mean_vector": {
            d: (None if raw_vec[i] is None else round(raw_vec[i], 6))
            for i, d in enumerate(TASTE_DIMENSIONS)
        },
        "normalized_simplex": {d: round(simplex[i], 4) for i, d in enumerate(TASTE_DIMENSIONS)},
        "dominant_taste": dom_taste,
        "shannon_entropy_nats": round(entropy, 4),
        "taste_evenness_j": round(evenness, 4),
    }


def predict_from_weights(
    weights_bundle: Dict[str, Any],
    raw_ingredients: List[str],
    aliases: Dict[str, str],
    setting: str = "full",
) -> Dict[str, Any]:
    key = "masked" if (setting == "masked" and "masked" in weights_bundle) else "full"
    model = weights_bundle[key]
    used_setting = (
        "Setting 2: Masked_Anchors (Contextual Co-occurrence)"
        if key == "masked"
        else "Setting 1: Full_Ingredients"
    )
    classes: List[str] = model["classes"]
    intercept: List[float] = model["intercept"]
    weights: Dict[str, List[float]] = model["weights"]

    normalized: List[str] = []
    for raw in raw_ingredients:
        norm = normalize_ingredient(raw, aliases)
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
            "is_synthetic_model": bool(weights_bundle.get("is_synthetic", False)),
            "status": "empty_input",
        }

    known = [tok for tok in normalized if tok in weights]
    unknown = [tok for tok in normalized if tok not in weights]

    if not known:
        return {
            "predicted_dominant_taste": "unspecified",
            "probabilities": {c: 0.0 for c in classes},
            "normalized_ingredients": normalized,
            "known_ingredients": [],
            "unknown_ingredients": unknown,
            "model_setting": used_setting,
            "is_synthetic_model": bool(weights_bundle.get("is_synthetic", False)),
            "status": "no_known_ingredients",
        }

    logits = list(intercept)
    for tok in known:
        tok_w = weights[tok]
        for idx in range(len(classes)):
            logits[idx] += tok_w[idx]

    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    exp_sum = sum(exps)
    probs = [e / exp_sum for e in exps]
    best_idx = max(range(len(classes)), key=lambda i: probs[i])

    return {
        "predicted_dominant_taste": classes[best_idx],
        "probabilities": {c: round(probs[i], 4) for i, c in enumerate(classes)},
        "normalized_ingredients": normalized,
        "known_ingredients": known,
        "unknown_ingredients": unknown,
        "model_setting": used_setting,
        "is_synthetic_model": bool(weights_bundle.get("is_synthetic", False)),
        "status": "success",
    }


def retrieve_similar_from_rows(
    feature_rows: List[Dict[str, Any]],
    query_recipe_id: int,
    top_k: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    valid_rows = [r for r in feature_rows if r.get("dominant_taste") != "unspecified"]
    query_row = next((r for r in valid_rows if int(r["recipe_id"]) == query_recipe_id), None)
    if query_row is None:
        raise KeyError(f"Recipe ID {query_recipe_id} not found.")
    if len(valid_rows) < 2:
        raise ValueError("At least 2 valid recipes are required for similarity retrieval.")

    q_vec = [float(query_row[d] or 0.0) for d in TASTE_DIMENSIONS]
    q_norm = math.sqrt(sum(x * x for x in q_vec))
    if q_norm <= 1e-12:
        raise ValueError("Query recipe has zero norm taste vector.")

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for row in valid_rows:
        cid = int(row["recipe_id"])
        if cid == query_recipe_id:
            continue
        c_vec = [float(row[d] or 0.0) for d in TASTE_DIMENSIONS]
        c_norm = math.sqrt(sum(x * x for x in c_vec))
        if c_norm <= 1e-12:
            sim = 0.0
        else:
            dot = sum(a * b for a, b in zip(q_vec, c_vec))
            sim = max(-1.0, min(1.0, dot / (q_norm * c_norm)))
        scored.append((sim, cid, row))

    scored.sort(key=lambda item: (-item[0], item[1]))
    neighbors = []
    for sim, cid, row in scored[:top_k]:
        evenness = row.get("taste_evenness")
        if evenness is None:
            evenness = row.get("taste_balance")
        neighbors.append(
            {
                "neighbor_recipe_id": cid,
                "cuisine": str(row["cuisine"]),
                "dominant_taste": str(row["dominant_taste"]),
                "taste_evenness": round(float(evenness), 4) if evenness is not None else None,
                "cosine_similarity": round(sim, 4),
                "ingredients": str(row.get("ingredients_normalized", "")).split(";"),
            }
        )

    return query_row, neighbors
