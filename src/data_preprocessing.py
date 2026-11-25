"""
Deliverable 1: Recipe-Ingredient-Taste Dataset Construction.
Handles ingredient normalization, alias resolution, schema validation,
within-cuisine deduplication, verified lexicon matching, and manifest/report generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ANNOTATION_STATUSES,
    CHEMESTHETIC_TAGS,
    CONFIDENCE_LEVELS,
    DEFAULT_ALIASES_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_RAW_RECIPES_PATH,
    MIN_MATCH_COVERAGE_RATIO,
    MIN_MATCHED_INGREDIENTS,
    RANDOM_SEED,
    RunContext,
    TASTE_DIMENSIONS,
    default_context,
    display_path,
    ensure_directories,
)

logger = logging.getLogger(__name__)

KNOWN_KAGGLE_TRAIN_SHA256 = "f18e7e568d1c606774bfb134ece51ef0a143cd5069e6f9241bb6036b45b7d2b0"

# Non-informative preparation and unit stopwords stripped during token normalization
PREP_STOPWORDS = {
    "chopped",
    "diced",
    "sliced",
    "minced",
    "crushed",
    "peeled",
    "seeded",
    "halved",
    "quartered",
    "finely",
    "coarsely",
    "freshly",
    "thinly",
    "thickly",
    "organic",
    "large",
    "medium",
    "small",
    "fresh",
    "grated",
    "shredded",
    "cooked",
    "uncooked",
    "prepared",
    "optional",
    "canned",
    "drained",
    "rinsed",
    # Measurement and unit stopwords
    "cup",
    "cups",
    "tbsp",
    "tsp",
    "tablespoon",
    "tablespoons",
    "teaspoon",
    "teaspoons",
    "ounce",
    "ounces",
    "oz",
    "lb",
    "lbs",
    "pound",
    "pounds",
    "pinch",
    "dash",
    "bunch",
    "slice",
    "slices",
    "can",
    "cans",
}

# Minimal fallback aliases used only if an external aliases CSV is not provided
FALLBACK_ALIASES: Dict[str, str] = {
    "extra virgin olive oil": "olive oil",
    "kosher salt": "salt",
    "sea salt": "salt",
    "garlic cloves": "garlic",
    "yellow onion": "onion",
    "onions": "onion",
    "carrots": "carrot",
    "tomatoes": "tomato",
    "eggs": "egg",
    "scallions": "green onion",
    "green onions": "green onion",
    "shallots": "shallot",
    "purple onion": "red onion",
    "parmigiano reggiano": "parmesan cheese",
    "fresh ginger": "ginger",
    "ground ginger": "dried ginger",
    "red bell pepper": "bell pepper",
    "red chili powder": "cayenne pepper",
    "unsalted butter": "butter",
    "granulated sugar": "sugar",
    "confectioners sugar": "powdered sugar",
}


@lru_cache(maxsize=8)
def _load_aliases_cached(resolved_path: str, mtime_ns: int) -> Dict[str, str]:
    df = pd.read_csv(resolved_path)
    required = {"surface_form", "canonical"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Aliases CSV missing columns: {sorted(missing)}")
    aliases: Dict[str, str] = {}
    for _, row in df.iterrows():
        surf = str(row["surface_form"]).strip().lower()
        canon = str(row["canonical"]).strip().lower()
        if surf and canon:
            aliases[surf] = canon
    return aliases


def load_aliases(aliases_path: Optional[Path] = None) -> Dict[str, str]:
    """Load canonical ingredient alias mappings from CSV (or fallback dictionary if absent)."""
    path = aliases_path or DEFAULT_ALIASES_PATH
    if path.exists():
        resolved = path.resolve()
        return _load_aliases_cached(str(resolved), resolved.stat().st_mtime_ns)
    logger.warning("Aliases CSV not found at %s; using minimal built-in fallback aliases.", path)
    return dict(FALLBACK_ALIASES)


def normalize_ingredient_name(
    raw_name: str,
    aliases: Optional[Dict[str, str]] = None,
    aliases_path: Optional[Path] = None,
) -> str:
    """
    Normalize raw ingredient string deterministically:
    1. Lowercase and strip whitespace.
    2. Strip parenthetical volume/instructional notes (e.g. '(optional)', '(10 oz)').
    3. Replace hyphens with spaces and remove digits/punctuation.
    4. Collapse multiple spaces.
    5. Check direct alias mapping before stripping preparation stopwords.
    6. Strip preparation/unit stopwords (only if informative tokens remain) and check alias again.
    """
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

    alias_map = aliases if aliases is not None else load_aliases(aliases_path)
    if text in alias_map:
        return alias_map[text]

    tokens = [tok for tok in text.split() if tok not in PREP_STOPWORDS]
    cleaned = " ".join(tokens) if tokens else text
    return alias_map.get(cleaned, cleaned)


def validate_lexicon_df(lexicon_df: pd.DataFrame) -> pd.DataFrame:
    """Validate taste lexicon schema, numeric ranges [0, 1] (or NaN), tags, and uniqueness."""
    required_cols = [
        "ingredient_norm",
        "category",
        "sweet",
        "sour",
        "bitter",
        "salty",
        "umami",
        "chemesthetic_pungency",
        "confidence_level",
        "annotation_status",
        "flavordb_entity_id",
        "flavordb_entity_name",
        "primary_aroma_compounds",
        "known_tastant_or_pungent_compounds",
        "annotation_provenance",
    ]
    missing = [c for c in required_cols if c not in lexicon_df.columns]
    if missing:
        raise ValueError(f"Taste lexicon missing required columns: {missing}")

    if not lexicon_df["ingredient_norm"].is_unique:
        dups = lexicon_df[lexicon_df["ingredient_norm"].duplicated()]["ingredient_norm"].tolist()
        raise ValueError(f"Duplicate ingredient_norm entries in lexicon: {dups}")

    for dim in TASTE_DIMENSIONS:
        vals = pd.to_numeric(lexicon_df[dim], errors="coerce")
        non_na = vals.dropna()
        if ((non_na < 0.0) | (non_na > 1.0)).any():
            raise ValueError(f"Lexicon taste scores in '{dim}' must lie in [0.0, 1.0] or be NaN.")

    bad_chem = set(lexicon_df["chemesthetic_pungency"].dropna()) - set(CHEMESTHETIC_TAGS)
    if bad_chem:
        raise ValueError(f"Invalid chemesthetic_pungency tags in lexicon: {sorted(bad_chem)}")

    bad_conf = set(lexicon_df["confidence_level"].dropna()) - set(CONFIDENCE_LEVELS)
    if bad_conf:
        raise ValueError(f"Invalid confidence_level entries in lexicon: {sorted(bad_conf)}")

    bad_status = set(lexicon_df["annotation_status"].dropna()) - set(ANNOTATION_STATUSES)
    if bad_status:
        raise ValueError(f"Invalid annotation_status entries in lexicon: {sorted(bad_status)}")

    for _, row in lexicon_df.iterrows():
        eid = str(row["flavordb_entity_id"]).strip()
        if eid != "NONE" and not eid.isdigit():
            raise ValueError(
                f"Invalid flavordb_entity_id '{eid}' for '{row['ingredient_norm']}'; "
                "must be verified integer ID or 'NONE'."
            )
        taste_vals = [row[d] for d in TASTE_DIMENSIONS]
        all_nan = all(pd.isna(v) for v in taste_vals)
        if row["annotation_status"] == "taste_unknown" and not all_nan:
            raise ValueError(f"'{row['ingredient_norm']}' marked taste_unknown but has numeric taste values.")
        if row["annotation_status"] in {"reference_standard", "curated_proxy"} and all_nan:
            raise ValueError(f"'{row['ingredient_norm']}' marked {row['annotation_status']} but all taste values are NaN.")

    return lexicon_df


def load_lexicon(lexicon_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the ingredient taste lexicon CSV."""
    path = lexicon_path or DEFAULT_LEXICON_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Taste lexicon file not found at '{path}'. Ensure data/raw/ingredient_taste_lexicon.csv is present."
        )
    df = pd.read_csv(path, keep_default_na=False)
    for dim in TASTE_DIMENSIONS:
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
    return validate_lexicon_df(df)


def generate_sample_recipes_dataset(seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Generate a deterministic, multi-cuisine synthetic sample recipe dataset
    across 6 cuisines with intentional edge cases (duplicates, empty lists)
    specifically designed for unit tests and offline pipeline smoke tests.
    """
    rng = np.random.default_rng(seed)

    cuisine_templates = {
        "Italian": [
            ["Roma tomatoes", "fresh garlic", "extra-virgin olive oil", "fresh basil", "parmigiano-reggiano", "kosher salt", "wheat flour"],
            ["tomato paste", "garlic cloves", "yellow onion", "olive oil", "oregano", "parmesan cheese", "beef", "black pepper", "salt"],
            ["anchovy", "capers", "olives", "roma tomatoes", "garlic", "olive oil", "oregano", "red chili powder"],
            ["mushroom", "parmesan cheese", "unsalted butter", "onion", "garlic", "rice", "olive oil", "salt"],
            ["feta cheese", "olives", "tomato", "olive oil", "oregano", "lemon juice", "garlic"],
            ["heavy cream", "butter", "parmesan cheese", "garlic", "black pepper", "salt", "wheat flour", "nutmeg"],
        ],
        "Indian": [
            ["ground turmeric", "ground cumin", "coriander seeds", "fresh ginger", "garlic", "onion", "tomato", "green chilies", "salt"],
            ["chickpeas", "onion", "tomato", "ginger", "garlic", "turmeric", "cumin", "cayenne pepper", "lemon juice", "cilantro"],
            ["yogurt", "chicken", "ginger", "garlic", "turmeric", "cumin", "cayenne pepper", "lemon juice", "salt", "garam_masala_unmatched"],
            ["spinach", "fenugreek", "turmeric", "cumin", "ginger", "garlic", "onion", "green chili", "butter", "salt"],
            ["tamarind", "brown sugar", "cumin", "dried ginger", "cayenne pepper", "salt", "mint", "cilantro"],
            ["eggplant", "onion", "tomato", "turmeric", "cumin", "coriander seed", "garlic", "ginger", "green chili", "salt"],
        ],
        "Mexican": [
            ["avocado", "fresh lime juice", "cilantro", "red onion", "jalapeno pepper", "roma tomatoes", "kosher salt"],
            ["black beans", "onion", "garlic", "ground cumin", "jalapeno", "tomato", "lime juice", "cilantro", "salt"],
            ["pork", "lime juice", "onion", "garlic", "oregano", "cumin", "jalapeno", "cilantro", "salt", "corn_tortilla_unmatched"],
            ["chicken", "tomato", "jalapeno", "onion", "garlic", "cumin", "lime juice", "cilantro", "cheddar cheese"],
            ["cocoa powder", "cinnamon", "cayenne pepper", "tomato", "onion", "garlic", "cumin", "sugar", "chicken", "salt"],
            ["mango", "lime juice", "red onion", "jalapeno", "cilantro", "bell pepper", "salt"],
        ],
        "Japanese": [
            ["soy sauce", "miso paste", "shiitake mushroom", "green onions", "ginger", "rice vinegar", "sesame oil", "tofu"],
            ["soy sauce", "sugar", "rice vinegar", "ginger", "garlic", "chicken", "sesame oil", "green onion"],
            ["rice", "rice vinegar", "sugar", "salt", "shrimp", "avocado", "soy sauce", "nori_unmatched"],
            ["miso paste", "shiitake mushroom", "green onion", "soy sauce", "ginger", "spinach"],
            ["beef", "soy sauce", "sugar", "onion", "ginger", "garlic", "sesame oil", "green onion", "rice"],
            ["pork", "miso paste", "soy sauce", "ginger", "garlic", "green onion", "sesame oil", "shiitake mushroom"],
        ],
        "Thai": [
            ["fish sauce", "fresh lime juice", "coconut milk", "brown sugar", "green chilies", "garlic", "ginger", "chicken", "cilantro"],
            ["tamarind", "fish sauce", "brown sugar", "lime juice", "shrimp", "garlic", "green onion", "cayenne pepper", "rice"],
            ["mango", "coconut milk", "sugar", "salt", "rice"],
            ["coconut milk", "fish sauce", "lime juice", "shiitake mushroom", "ginger", "green chili", "cilantro", "sugar", "chicken"],
            ["beef", "fish sauce", "lime juice", "red onion", "mint", "cilantro", "green chili", "brown sugar"],
            ["oyster sauce", "soy sauce", "fish sauce", "sugar", "garlic", "green chili", "basil", "pork"],
        ],
        "American_Bakery_Comfort": [
            ["granulated sugar", "brown sugar", "unsalted butter", "all-purpose flour", "cocoa powder", "vanilla", "milk", "salt"],
            ["apple", "cinnamon", "sugar", "brown sugar", "butter", "wheat flour", "lemon juice", "salt"],
            ["sweet potato", "brown sugar", "maple syrup", "butter", "cinnamon", "milk", "vanilla", "salt"],
            ["coffee", "cocoa powder", "sugar", "heavy cream", "butter", "vanilla", "wheat flour"],
            ["cheddar cheese", "milk", "salted butter", "wheat flour", "onion", "black pepper", "salt"],
            ["beef", "tomato paste", "vinegar", "brown sugar", "honey", "onion", "garlic", "salt", "black pepper"],
        ],
    }

    optional_extras = [
        "black pepper",
        "salt",
        "garlic",
        "onion",
        "olive oil",
        "water",
        "baking_powder_unmatched",
    ]

    recipes = []
    recipe_id = 1000
    for cuisine, templates in cuisine_templates.items():
        for idx in range(30):
            base = list(templates[idx % len(templates)])
            if len(base) > 5 and rng.random() < 0.35:
                drop_idx = int(rng.integers(0, len(base)))
                base.pop(drop_idx)
            if rng.random() < 0.5:
                extra = optional_extras[int(rng.integers(0, len(optional_extras)))]
                if extra not in base:
                    base.append(extra)
            recipes.append({"id": recipe_id, "cuisine": cuisine, "ingredients": base})
            recipe_id += 1

    # Intentional edge cases for validation tests
    recipes.append(recipes[0].copy())  # exact duplicate ID and content
    recipes.append({"id": 9991, "cuisine": "Italian", "ingredients": recipes[1]["ingredients"]})  # duplicate signature
    recipes.append({"id": 9992, "cuisine": "", "ingredients": ["salt"]})  # missing cuisine
    recipes.append({"id": 9993, "cuisine": "Indian", "ingredients": []})  # empty ingredient list
    return recipes


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for provenance manifest."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_recipes(
    raw_json_path: Path = DEFAULT_RAW_RECIPES_PATH,
    use_sample_data: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load raw recipes strictly requiring real data by default.
    Raises FileNotFoundError when raw_json_path is absent unless use_sample_data=True.
    Never silently substitutes synthetic data for real data.
    """
    if use_sample_data:
        sample_data = generate_sample_recipes_dataset()
        meta = {
            "dataset_name": "Synthetic Multi-Cuisine Development Corpus (--use-sample-data)",
            "source_url": "Internal deterministic generator (generate_sample_recipes_dataset)",
            "source_file": "internal::synthetic_corpus",
            "sha256": "deterministic_seed_42",
            "access_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "license": "CC0 / Internal Synthetic Testing Only (Not Empirical Data)",
            "is_synthetic": True,
            "raw_record_count": len(sample_data),
        }
        return pd.DataFrame(sample_data), meta

    if not raw_json_path.exists():
        raise FileNotFoundError(
            f"External raw recipe dataset not found at '{raw_json_path}'.\n"
            "To run on real data:\n"
            "  1. Obtain 'train.json' from Kaggle What's Cooking: https://www.kaggle.com/c/whats-cooking/data\n"
            f"  2. Place it at '{DEFAULT_RAW_RECIPES_PATH}' (or pass '--data-path path/to/recipes.json').\n"
            "For offline unit testing or smoke testing only, explicitly pass '--use-sample-data'."
        )

    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected JSON array of recipe objects in {raw_json_path}")

    sha256 = compute_file_sha256(raw_json_path)
    mtime = datetime.fromtimestamp(raw_json_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    is_kaggle_whats_cooking = (
        sha256 == KNOWN_KAGGLE_TRAIN_SHA256
        or raw_json_path.resolve() == DEFAULT_RAW_RECIPES_PATH.resolve()
    )

    if is_kaggle_whats_cooking:
        dataset_name = "Kaggle / Yummly What's Cooking Dataset"
        source_url = "https://www.kaggle.com/c/whats-cooking/data"
        license_note = (
            "Unknown / Kaggle Playground Competition Rules (Yummly Inc.; "
            "educational/non-commercial research only, raw data not redistributed)"
        )
    else:
        dataset_name = f"External Recipe Corpus ({raw_json_path.name})"
        source_url = "User-supplied external dataset path"
        license_note = "User-supplied dataset (verify upstream license)"

    meta = {
        "dataset_name": dataset_name,
        "source_url": source_url,
        "source_file": display_path(raw_json_path),
        "sha256": sha256,
        "access_date": mtime,
        "license": license_note,
        "is_synthetic": False,
        "raw_record_count": len(data),
    }
    return pd.DataFrame(data), meta


def validate_raw_recipes_df(
    raw_df: pd.DataFrame,
    aliases: Optional[Dict[str, str]] = None,
    aliases_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate raw recipe DataFrame schema and filter invalid records with explicit accounting.
    Checks required columns ('id', 'cuisine', 'ingredients'), integer-coercible IDs,
    non-empty cuisines, list-typed ingredients, and non-empty normalized ingredient sets.
    """
    required_cols = {"id", "cuisine", "ingredients"}
    missing = sorted(required_cols - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw recipe dataset missing required columns: {missing}")

    df = raw_df.copy()
    breakdown = {
        "missing_or_invalid_id": 0,
        "missing_cuisine": 0,
        "invalid_or_empty_ingredients": 0,
        "empty_after_normalization": 0,
    }

    parsed_ids = pd.to_numeric(df["id"], errors="coerce")
    valid_id_mask = parsed_ids.notna() & (parsed_ids % 1 == 0)
    breakdown["missing_or_invalid_id"] = int((~valid_id_mask).sum())

    cuisines = df["cuisine"].fillna("").astype(str).str.strip()
    valid_cuisine_mask = cuisines != ""
    breakdown["missing_cuisine"] = int((valid_id_mask & ~valid_cuisine_mask).sum())

    valid_ing_list_mask = df["ingredients"].apply(
        lambda x: isinstance(x, list) and any(isinstance(item, str) and item.strip() for item in x)
    )
    breakdown["invalid_or_empty_ingredients"] = int(
        (valid_id_mask & valid_cuisine_mask & ~valid_ing_list_mask).sum()
    )

    pre_mask = valid_id_mask & valid_cuisine_mask & valid_ing_list_mask
    filtered = df[pre_mask].copy()
    filtered["id"] = parsed_ids[pre_mask].astype(int)
    filtered["cuisine"] = cuisines[pre_mask]

    alias_map = aliases if aliases is not None else load_aliases(aliases_path)
    filtered["norm_list"] = filtered["ingredients"].apply(
        lambda lst: sorted(
            {
                n
                for item in lst
                for n in [normalize_ingredient_name(item, aliases=alias_map)]
                if n
            }
        )
    )
    non_empty_norm = filtered["norm_list"].apply(len) > 0
    breakdown["empty_after_normalization"] = int((~non_empty_norm).sum())

    valid_df = filtered[non_empty_norm].copy().reset_index(drop=True)
    return valid_df, breakdown


def run_deliverable_1(
    raw_data_path: Optional[Path] = None,
    use_sample_data: bool = False,
    ctx: Optional[RunContext] = None,
) -> Dict[str, object]:
    """
    Execute Deliverable 1 end-to-end:
    1. Loads raw recipes (real by default; synthetic only with explicit opt-in).
    2. Validates schema and filters invalid/empty records.
    3. Normalizes ingredients and resolves aliases without collapsing culinary distinctions.
    4. Deduplicates by recipe ID and within-cuisine canonical signature (preserving cross-cuisine matches).
    5. Maps ingredients to the verified taste lexicon and logs taste-unknown and unmatched items.
    6. Writes clean CSVs, dataset_manifest.json, and reports/dataset_summary.md.
    """
    context = ctx or default_context()
    context.ensure()

    lexicon_df = load_lexicon(context.lexicon_path)
    aliases = load_aliases(context.aliases_path)

    raw_json_path = raw_data_path or DEFAULT_RAW_RECIPES_PATH
    raw_df, meta = load_raw_recipes(raw_json_path, use_sample_data=use_sample_data)
    initial_count = len(raw_df)

    valid_df, invalid_breakdown = validate_raw_recipes_df(raw_df, aliases=aliases)
    invalid_removed = initial_count - len(valid_df)

    # Deduplicate exact IDs first, then identical ingredient signatures WITHIN each cuisine
    valid_df["signature_within_cuisine"] = (
        valid_df["cuisine"] + "::" + valid_df["norm_list"].apply(lambda x: "|".join(x))
    )
    valid_df["ingredient_bag_key"] = valid_df["norm_list"].apply(lambda x: "|".join(x))

    dedup_id_df = valid_df.drop_duplicates(subset=["id"]).copy()
    duplicate_ids_removed = len(valid_df) - len(dedup_id_df)

    dedup_df = dedup_id_df.drop_duplicates(subset=["signature_within_cuisine"]).copy()
    within_cuisine_dups_removed = len(dedup_id_df) - len(dedup_df)
    duplicates_removed = duplicate_ids_removed + within_cuisine_dups_removed

    # Count legitimate cross-cuisine shared ingredient sets preserved by within-cuisine dedup
    bag_cuisine_counts = dedup_df.groupby("ingredient_bag_key")["cuisine"].nunique()
    cross_cuisine_shared_signatures = int((bag_cuisine_counts > 1).sum())

    lexicon_lookup = lexicon_df.set_index("ingredient_norm").to_dict(orient="index")
    taste_known_set = {
        ing
        for ing, info in lexicon_lookup.items()
        if any(pd.notna(info[d]) for d in TASTE_DIMENSIONS)
    }

    recipe_rows = []
    ingredient_rows = []

    for _, row in dedup_df.iterrows():
        r_id = int(row["id"])
        cuisine = str(row["cuisine"])
        raw_ings = row["ingredients"]

        seen_norm_in_recipe = set()
        lexicon_matched_count = 0
        taste_known_count = 0
        total_norm_count = 0

        for pos_idx, raw_ing in enumerate(raw_ings):
            norm_ing = normalize_ingredient_name(raw_ing, aliases=aliases)
            if not norm_ing or norm_ing in seen_norm_in_recipe:
                continue
            seen_norm_in_recipe.add(norm_ing)
            total_norm_count += 1

            in_lexicon = norm_ing in lexicon_lookup
            has_known_taste = norm_ing in taste_known_set
            if in_lexicon:
                lexicon_matched_count += 1
            if has_known_taste:
                taste_known_count += 1

            ingredient_rows.append(
                {
                    "recipe_id": r_id,
                    "cuisine": cuisine,
                    "position_index": pos_idx,
                    "ingredient_raw": str(raw_ing),
                    "ingredient_norm": norm_ing,
                    "is_matched": has_known_taste,
                    "in_lexicon": in_lexicon,
                }
            )

        match_ratio = taste_known_count / total_norm_count if total_norm_count > 0 else 0.0
        sufficient_for_analysis = (
            taste_known_count >= MIN_MATCHED_INGREDIENTS
            and match_ratio >= MIN_MATCH_COVERAGE_RATIO
        )

        recipe_rows.append(
            {
                "recipe_id": r_id,
                "cuisine": cuisine,
                "n_ingredients_total": total_norm_count,
                "n_ingredients_matched": taste_known_count,
                "n_ingredients_in_lexicon": lexicon_matched_count,
                "match_coverage_ratio": round(match_ratio, 4),
                "sufficient_for_analysis": sufficient_for_analysis,
                "ingredients_normalized": ";".join(sorted(seen_norm_in_recipe)),
            }
        )

    recipes_clean_df = pd.DataFrame(recipe_rows)
    recipe_ingredients_df = pd.DataFrame(ingredient_rows)

    norm_freq = (
        recipe_ingredients_df.groupby("ingredient_norm")
        .agg(
            recipe_frequency=("recipe_id", "nunique"),
            example_raw_surface=("ingredient_raw", "first"),
        )
        .reset_index()
    )

    mapping_rows = []
    for _, nrow in norm_freq.iterrows():
        ing_norm = nrow["ingredient_norm"]
        if ing_norm in lexicon_lookup:
            info = lexicon_lookup[ing_norm]
            has_taste = ing_norm in taste_known_set
            mapping_rows.append(
                {
                    "ingredient_norm": ing_norm,
                    "example_raw_surface": nrow["example_raw_surface"],
                    "recipe_frequency": int(nrow["recipe_frequency"]),
                    "match_status": "matched" if has_taste else "taste_unknown",
                    "category": info.get("category", "unclassified"),
                    "sweet": info.get("sweet", np.nan),
                    "sour": info.get("sour", np.nan),
                    "bitter": info.get("bitter", np.nan),
                    "salty": info.get("salty", np.nan),
                    "umami": info.get("umami", np.nan),
                    "chemesthetic_pungency": info.get("chemesthetic_pungency", "none"),
                    "confidence_level": info.get("confidence_level", "medium"),
                    "annotation_status": info.get("annotation_status", "curated_proxy"),
                    "flavordb_entity_id": info.get("flavordb_entity_id", "NONE"),
                    "flavordb_entity_name": info.get("flavordb_entity_name", "NONE"),
                    "primary_aroma_compounds": info.get("primary_aroma_compounds", "NONE"),
                    "known_tastant_or_pungent_compounds": info.get("known_tastant_or_pungent_compounds", "NONE"),
                    "annotation_provenance": info.get("annotation_provenance", "curated"),
                }
            )
        else:
            mapping_rows.append(
                {
                    "ingredient_norm": ing_norm,
                    "example_raw_surface": nrow["example_raw_surface"],
                    "recipe_frequency": int(nrow["recipe_frequency"]),
                    "match_status": "unmatched",
                    "category": "unmatched",
                    "sweet": np.nan,
                    "sour": np.nan,
                    "bitter": np.nan,
                    "salty": np.nan,
                    "umami": np.nan,
                    "chemesthetic_pungency": "none",
                    "confidence_level": "unknown",
                    "annotation_status": "unmatched",
                    "flavordb_entity_id": "NONE",
                    "flavordb_entity_name": "NONE",
                    "primary_aroma_compounds": "NONE",
                    "known_tastant_or_pungent_compounds": "NONE",
                    "annotation_provenance": "unmatched_logged",
                }
            )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        by=["match_status", "recipe_frequency"], ascending=[True, False]
    ).reset_index(drop=True)

    recipes_clean_df.to_csv(context.processed_dir / "recipes_clean.csv", index=False)
    recipe_ingredients_df.to_csv(context.processed_dir / "recipe_ingredients.csv", index=False)
    mapping_df.to_csv(context.processed_dir / "ingredient_flavor_mapping.csv", index=False)

    n_recipes_clean = len(recipes_clean_df)
    n_unique_ings = len(mapping_df)
    n_matched_taste = int((mapping_df["match_status"] == "matched").sum())
    n_taste_unknown = int((mapping_df["match_status"] == "taste_unknown").sum())
    n_unmatched = int((mapping_df["match_status"] == "unmatched").sum())
    pct_matched_unique = (n_matched_taste / n_unique_ings * 100.0) if n_unique_ings else 0.0

    total_ing_tokens = len(recipe_ingredients_df)
    matched_ing_tokens = int(recipe_ingredients_df["is_matched"].sum())
    in_lexicon_tokens = int(recipe_ingredients_df["in_lexicon"].sum())
    pct_matched_tokens = (matched_ing_tokens / total_ing_tokens * 100.0) if total_ing_tokens else 0.0

    n_cuisines = recipes_clean_df["cuisine"].nunique()
    n_sufficient = int(recipes_clean_df["sufficient_for_analysis"].sum())
    pct_sufficient = (n_sufficient / n_recipes_clean * 100.0) if n_recipes_clean else 0.0

    cuisine_counts = (
        recipes_clean_df.groupby("cuisine")
        .agg(
            total_recipes=("recipe_id", "count"),
            sufficient_recipes=("sufficient_for_analysis", "sum"),
            mean_match_ratio=("match_coverage_ratio", "mean"),
        )
        .reset_index()
        .sort_values("cuisine")
    )
    cuisine_counts["mean_match_ratio"] = cuisine_counts["mean_match_ratio"].round(4)

    top_unmatched_df = mapping_df[mapping_df["match_status"] == "unmatched"][
        ["ingredient_norm", "recipe_frequency", "example_raw_surface"]
    ].head(15)

    manifest = {
        "dataset_name": meta["dataset_name"],
        "source_url": meta["source_url"],
        "source_file": meta["source_file"],
        "sha256": meta["sha256"],
        "access_date": meta["access_date"],
        "license": meta["license"],
        "is_synthetic": meta["is_synthetic"],
        "lexicon_path": display_path(context.lexicon_path),
        "lexicon_entries_count": len(lexicon_df),
        "aliases_path": display_path(context.aliases_path),
        "aliases_count": len(aliases),
        "raw_record_count": initial_count,
        "invalid_records_removed": invalid_removed,
        "invalid_breakdown": invalid_breakdown,
        "duplicate_ids_removed": duplicate_ids_removed,
        "within_cuisine_duplicates_removed": within_cuisine_dups_removed,
        "duplicates_removed": duplicates_removed,
        "cross_cuisine_shared_signatures_preserved": cross_cuisine_shared_signatures,
        "clean_recipe_count": n_recipes_clean,
        "unique_ingredients_count": n_unique_ings,
        "matched_ingredients_count": n_matched_taste,
        "taste_unknown_ingredients_count": n_taste_unknown,
        "unmatched_ingredients_count": n_unmatched,
        "pct_unique_matched": round(pct_matched_unique, 2),
        "total_ingredient_tokens": total_ing_tokens,
        "matched_ingredient_tokens": matched_ing_tokens,
        "in_lexicon_tokens": in_lexicon_tokens,
        "pct_tokens_matched": round(pct_matched_tokens, 2),
        "sufficient_recipes_count": n_sufficient,
        "pct_recipes_sufficient": round(pct_sufficient, 2),
        "cuisines_represented": n_cuisines,
        "cuisine_distribution": {
            str(k): int(v)
            for k, v in cuisine_counts.set_index("cuisine")["total_recipes"].items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(context.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    corpus_badge = (
        "**SYNTHETIC SMOKE TEST (`--use-sample-data`) — NOT EMPIRICAL EVIDENCE**"
        if meta["is_synthetic"]
        else "**REAL EMPIRICAL CORPUS**"
    )
    top_unmatched_md = (
        top_unmatched_df.to_markdown(index=False)
        if not top_unmatched_df.empty
        else "_No unmatched ingredients in this corpus._"
    )

    summary_md = f"""# Deliverable 1: Dataset Construction & Provenance Summary

> **Corpus Classification:** {corpus_badge}

## 1. Data Source & Provenance Manifest
- **Dataset Name:** {meta['dataset_name']}
- **Source URL:** {meta['source_url']}
- **Source File:** `{meta['source_file']}`
- **SHA-256 Checksum:** `{meta['sha256']}`
- **Access / File Date:** `{meta['access_date']}`
- **License / Terms:** {meta['license']}
- **Taste Lexicon:** `{display_path(context.lexicon_path)}` ({len(lexicon_df)} entries; `reference_standard`, `curated_proxy`, and `taste_unknown` explicitly labeled; FlavorDB entity IDs verified against Garg et al., 2018)
- **Alias Dictionary:** `{display_path(context.aliases_path)}` ({len(aliases)} deterministic surface-to-canonical rules)

## 2. Corpus Cleaning, Deduplication & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | {initial_count:,} |
| Invalid / Incomplete Records Removed | {invalid_removed:,} |
| Duplicate Recipe IDs Removed | {duplicate_ids_removed:,} |
| Within-Cuisine Canonical Duplicates Removed | {within_cuisine_dups_removed:,} |
| Cross-Cuisine Identical Ingredient Sets Preserved | {cross_cuisine_shared_signatures:,} |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **{n_recipes_clean:,}** |
| **Cuisines Represented** | **{n_cuisines}** |
| **Unique Normalized Ingredients** | **{n_unique_ings:,}** |
| Lexicon Matched with Taste Proxy (`match_status='matched'`) | {n_matched_taste:,} ({pct_matched_unique:.2f}%) |
| Cataloged as Taste-Unknown (`match_status='taste_unknown'`) | {n_taste_unknown:,} |
| Unmatched Unique Ingredients Logged (`match_status='unmatched'`) | {n_unmatched:,} |
| Total Recipe-Ingredient Occurrences (Tokens) | {total_ing_tokens:,} |
| Taste-Known Matched Occurrences | {matched_ing_tokens:,} ({pct_matched_tokens:.2f}%) |
| **Sufficient Recipes for Analysis ($M_r \\ge {MIN_MATCHED_INGREDIENTS}$, coverage $\\ge {int(MIN_MATCH_COVERAGE_RATIO * 100)}\\%$)** | **{n_sufficient:,} ({pct_sufficient:.2f}%)** |

## 3. Breakdown by Cuisine
{cuisine_counts.to_markdown(index=False)}

## 4. Top Unmatched Unique Ingredients (Logged for Transparency)
{top_unmatched_md}

## 5. Scientific Distinction: Gustatory Tastants, Retronasal Aroma, & Chemesthesis
1. **Basic Gustatory Tastants:** Sweet, sour, bitter, salty, and umami represent non-volatile, water-soluble tastants acting on lingual taste receptor cells (T1R, T2R, ENaC, OTOP1).
2. **Retronasal Aroma Compounds:** Volatile organic molecules verified in FlavorDB (e.g., *linalool*, *citral*, *2,3,5,6-tetramethylpyrazine*) activate olfactory neurons via retronasal airflow and do **not** validate numerical basic-taste intensities.
3. **Trigeminal Chemesthesis:** Capsaicin, piperine, gingerol, allicin, and allyl isothiocyanate stimulate somatosensory TRP channels (TRPV1/TRPA1) via the trigeminal nerve. Chili heat is not a basic taste modality and is tracked separately under `chemesthetic_pungency`.
4. **Unknown Taste Values (`NaN`):** Refined lipids (`cooking oil`, `peanut oil`) and unestablished minor dimensions on low-confidence items are recorded as `NaN` rather than fabricated numbers.
"""
    (context.reports_dir / "dataset_summary.md").write_text(summary_md, encoding="utf-8")

    return {
        "n_recipes_clean": n_recipes_clean,
        "n_unique_ingredients": n_unique_ings,
        "n_cuisines": n_cuisines,
        "pct_matched_unique": pct_matched_unique,
        "pct_matched_tokens": pct_matched_tokens,
        "n_sufficient": n_sufficient,
        "is_synthetic": meta["is_synthetic"],
        "duplicate_ids_removed": duplicate_ids_removed,
        "within_cuisine_duplicates_removed": within_cuisine_dups_removed,
    }
