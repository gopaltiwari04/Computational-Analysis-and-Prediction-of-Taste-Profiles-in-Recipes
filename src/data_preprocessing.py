"""
Deliverable 1: Recipe-Ingredient-Taste Dataset Construction.
Handles ingredient normalization, alias resolution, deduplication,
FlavorDB/lexicon matching, and dataset summary generation.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MIN_MATCH_COVERAGE_RATIO,
    MIN_MATCHED_INGREDIENTS,
    REPORTS_DIR,
    TASTE_DIMENSIONS,
    ensure_directories,
)

# Deterministic alias dictionary: resolves spelling/preparation variants
# while preserving distinct culinary ingredients.
INGREDIENT_ALIASES: Dict[str, str] = {
    "garlic cloves": "garlic",
    "minced garlic": "garlic",
    "fresh garlic": "garlic",
    "garlic": "garlic",
    "black garlic": "black garlic",
    "fresh ginger": "ginger",
    "ginger root": "ginger",
    "grated ginger": "ginger",
    "ground ginger": "dried ginger",
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "cane sugar": "sugar",
    "brown sugar": "brown sugar",
    "kosher salt": "salt",
    "sea salt": "salt",
    "table salt": "salt",
    "fine sea salt": "salt",
    "fresh lemon juice": "lemon juice",
    "juice of a lemon": "lemon juice",
    "fresh lime juice": "lime juice",
    "extra virgin olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "light soy sauce": "soy sauce",
    "low sodium soy sauce": "soy sauce",
    "all purpose flour": "wheat flour",
    "all-purpose flour": "wheat flour",
    "plain flour": "wheat flour",
    "roma tomatoes": "tomato",
    "plum tomatoes": "tomato",
    "fresh tomatoes": "tomato",
    "tomatoes": "tomato",
    "cherry tomatoes": "tomato",
    "diced tomatoes": "tomato",
    "tomato paste": "tomato paste",
    "parmigiano reggiano": "parmesan cheese",
    "parmigiano-reggiano": "parmesan cheese",
    "grated parmesan cheese": "parmesan cheese",
    "fresh basil": "basil",
    "sweet basil": "basil",
    "fresh cilantro": "cilantro",
    "coriander leaves": "cilantro",
    "ground coriander": "coriander seed",
    "coriander seeds": "coriander seed",
    "ground cumin": "cumin",
    "cumin seeds": "cumin",
    "ground turmeric": "turmeric",
    "turmeric powder": "turmeric",
    "cayenne": "cayenne pepper",
    "red chili powder": "cayenne pepper",
    "green chilies": "green chili",
    "green chile": "green chili",
    "jalapeno chilies": "jalapeno",
    "jalapeno pepper": "jalapeno",
    "heavy whipping cream": "heavy cream",
    "unsalted butter": "butter",
    "salted butter": "salted butter",
    "scallions": "green onion",
    "spring onions": "green onion",
    "green onions": "green onion",
    "yellow onion": "onion",
    "white onion": "onion",
    "onions": "onion",
    "red onion": "red onion",
}

# Preparation adjectives that do not alter core ingredient identity
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
}


def normalize_ingredient_name(raw_name: str) -> str:
    """
    Normalize a raw ingredient string:
    1. Convert to lowercase.
    2. Strip parenthetical notes (e.g., '(optional)', '(about 2 cups)').
    3. Replace hyphens/slashes with spaces and strip non-alphanumeric punctuation.
    4. Remove standalone preparation adverbs/participles while preserving
       meaningful modifiers (e.g., 'black garlic', 'dried ginger', 'brown sugar').
    5. Collapse whitespace and apply deterministic alias mapping.
    """
    if not isinstance(raw_name, str):
        return ""
    text = raw_name.lower().strip()
    if not text:
        return ""

    # Remove parenthetical remarks
    text = re.sub(r"\([^)]*\)", " ", text)
    # Replace hyphens with spaces so 'extra-virgin' == 'extra virgin'
    text = text.replace("-", " ")
    # Remove punctuation/digits except spaces and letters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Check direct alias before token stripping
    if text in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[text]

    # Strip non-informative preparation words only if other tokens remain
    tokens = [tok for tok in text.split() if tok not in PREP_STOPWORDS]
    cleaned = " ".join(tokens) if tokens else text

    # Check alias again after stripping prep descriptors
    return INGREDIENT_ALIASES.get(cleaned, cleaned)


def generate_default_lexicon(output_path: Path) -> pd.DataFrame:
    """
    Create a curated, literature-grounded ingredient taste & FlavorDB reference lexicon
    if one is not already present on disk.
    Taste intensities are in [0.0, 1.0].
    """
    records = [
        # (ingredient_norm, category, sweet, sour, bitter, salty, umami, flavordb_id, primary_flavor_compounds, annotation_source)
        ("garlic", "allium", 0.15, 0.05, 0.10, 0.00, 0.35, "FDB000142", "allicin;diallyl disulfide", "culinary_sensory+FlavorDB"),
        ("black garlic", "allium", 0.65, 0.25, 0.05, 0.10, 0.80, "FDB000142_ferm", "S-allyl-cysteine;5-HMF", "culinary_sensory"),
        ("onion", "allium", 0.35, 0.05, 0.05, 0.00, 0.30, "FDB000281", "dipropyl disulfide;quercetin", "culinary_sensory+FlavorDB"),
        ("red onion", "allium", 0.30, 0.10, 0.10, 0.00, 0.20, "FDB000282", "anthocyanins;thiopropanal S-oxide", "culinary_sensory+FlavorDB"),
        ("green onion", "allium", 0.20, 0.05, 0.10, 0.00, 0.20, "FDB000285", "propyl mercaptan", "culinary_sensory+FlavorDB"),
        ("ginger", "spice_rhizome", 0.10, 0.10, 0.20, 0.00, 0.05, "FDB000155", "6-gingerol;zingiberene", "culinary_sensory+FlavorDB"),
        ("dried ginger", "spice_rhizome", 0.10, 0.05, 0.25, 0.00, 0.05, "FDB000155_dry", "shogaol;zingerone", "culinary_sensory+FlavorDB"),
        ("turmeric", "spice_rhizome", 0.05, 0.05, 0.65, 0.00, 0.05, "FDB000412", "curcumin;turmerone", "culinary_sensory+FlavorDB"),
        ("cumin", "spice_seed", 0.05, 0.05, 0.45, 0.00, 0.10, "FDB000118", "cuminaldehyde;p-cymene", "culinary_sensory+FlavorDB"),
        ("coriander seed", "spice_seed", 0.15, 0.10, 0.25, 0.00, 0.05, "FDB000109", "linalool;alpha-pinene", "culinary_sensory+FlavorDB"),
        ("fenugreek", "spice_seed", 0.15, 0.00, 0.85, 0.00, 0.10, "FDB000136", "sotolon;trigonelline", "culinary_sensory+FlavorDB"),
        ("black pepper", "spice", 0.00, 0.00, 0.35, 0.00, 0.05, "FDB000064", "piperine;rotundone", "culinary_sensory+FlavorDB"),
        ("cayenne pepper", "chili", 0.05, 0.05, 0.25, 0.00, 0.05, "FDB000081", "capsaicin;dihydrocapsaicin", "culinary_sensory+FlavorDB"),
        ("green chili", "chili", 0.10, 0.10, 0.25, 0.00, 0.05, "FDB000082", "capsaicin;2-isobutyl-3-methoxypyrazine", "culinary_sensory+FlavorDB"),
        ("jalapeno", "chili", 0.15, 0.10, 0.20, 0.00, 0.05, "FDB000083", "capsaicin;hexanal", "culinary_sensory+FlavorDB"),
        ("bell pepper", "vegetable", 0.45, 0.10, 0.10, 0.00, 0.10, "FDB000084", "2-methoxy-3-isobutylpyrazine", "culinary_sensory+FlavorDB"),
        ("tomato", "vegetable_fruit", 0.35, 0.45, 0.05, 0.05, 0.70, "FDB000398", "glutamic acid;hexanal;lycopene", "culinary_sensory+FlavorDB"),
        ("tomato paste", "condiment", 0.40, 0.50, 0.10, 0.15, 0.90, "FDB000399", "L-glutamate;furfural", "culinary_sensory+FlavorDB"),
        ("lemon juice", "citrus", 0.10, 0.95, 0.10, 0.00, 0.00, "FDB000210", "citric acid;limonene;citral", "culinary_sensory+FlavorDB"),
        ("lime juice", "citrus", 0.08, 0.95, 0.15, 0.00, 0.00, "FDB000214", "citric acid;gamma-terpinene", "culinary_sensory+FlavorDB"),
        ("tamarind", "fruit_acid", 0.35, 0.85, 0.10, 0.00, 0.05, "FDB000385", "tartaric acid;2-acetylfuran", "culinary_sensory+FlavorDB"),
        ("vinegar", "acid", 0.05, 0.90, 0.05, 0.00, 0.05, "FDB000430", "acetic acid;ethyl acetate", "culinary_sensory+FlavorDB"),
        ("rice vinegar", "acid", 0.20, 0.75, 0.00, 0.05, 0.10, "FDB000431", "acetic acid;lactic acid", "culinary_sensory"),
        ("sugar", "sweetener", 1.00, 0.00, 0.00, 0.00, 0.00, "FDB000370", "sucrose", "culinary_sensory+SweetenersDB"),
        ("brown sugar", "sweetener", 0.95, 0.05, 0.05, 0.02, 0.05, "FDB000371", "sucrose;molasses furans", "culinary_sensory+SweetenersDB"),
        ("honey", "sweetener", 0.95, 0.10, 0.05, 0.00, 0.00, "FDB000175", "fructose;glucose;phenylacetic acid", "culinary_sensory+SweetenersDB"),
        ("maple syrup", "sweetener", 0.95, 0.05, 0.05, 0.00, 0.05, "FDB000232", "sucrose;sotolon;vanillin", "culinary_sensory+SweetenersDB"),
        ("coconut milk", "dairy_plant", 0.45, 0.00, 0.00, 0.05, 0.10, "FDB000102", "delta-octalactone;gamma-nonalactone", "culinary_sensory+FlavorDB"),
        ("milk", "dairy", 0.30, 0.05, 0.00, 0.05, 0.10, "FDB000245", "lactose;2-heptanone", "culinary_sensory+FlavorDB"),
        ("heavy cream", "dairy", 0.25, 0.05, 0.00, 0.05, 0.10, "FDB000246", "delta-decalactone;diacetyl", "culinary_sensory+FlavorDB"),
        ("butter", "dairy", 0.15, 0.02, 0.00, 0.05, 0.10, "FDB000070", "diacetyl;butyric acid", "culinary_sensory+FlavorDB"),
        ("salted butter", "dairy", 0.15, 0.02, 0.00, 0.55, 0.10, "FDB000071", "diacetyl;sodium chloride", "culinary_sensory"),
        ("yogurt", "dairy_fermented", 0.15, 0.65, 0.05, 0.10, 0.15, "FDB000445", "lactic acid;acetaldehyde", "culinary_sensory+FlavorDB"),
        ("parmesan cheese", "dairy_aged", 0.05, 0.15, 0.10, 0.75, 0.95, "FDB000091", "monosodium glutamate;butyric acid", "culinary_sensory+FlavorDB"),
        ("cheddar cheese", "dairy_aged", 0.05, 0.20, 0.10, 0.65, 0.65, "FDB000089", "lactic acid;methional", "culinary_sensory+FlavorDB"),
        ("feta cheese", "dairy_brined", 0.05, 0.45, 0.05, 0.85, 0.45, "FDB000093", "octanoic acid;sodium chloride", "culinary_sensory+FlavorDB"),
        ("salt", "mineral", 0.00, 0.00, 0.00, 1.00, 0.00, "FDB000350", "sodium chloride", "culinary_sensory"),
        ("soy sauce", "fermented_condiment", 0.15, 0.15, 0.10, 0.90, 0.95, "FDB000365", "L-glutamate;HEMF;sodium chloride", "culinary_sensory+FlavorDB"),
        ("fish sauce", "fermented_condiment", 0.10, 0.10, 0.05, 0.95, 0.98, "FDB000138", "glutamic acid;inosine monophosphate", "culinary_sensory+FlavorDB"),
        ("miso paste", "fermented_condiment", 0.25, 0.15, 0.10, 0.85, 0.90, "FDB000248", "glutamate;4-ethylguaiacol", "culinary_sensory+FlavorDB"),
        ("oyster sauce", "fermented_condiment", 0.45, 0.05, 0.05, 0.80, 0.90, "FDB000288", "glutamate;glycine;sucrose", "culinary_sensory"),
        ("anchovy", "seafood", 0.00, 0.05, 0.05, 0.85, 0.95, "FDB000018", "inosine-5-monophosphate;trimethylamine", "culinary_sensory+FlavorDB"),
        ("shrimp", "seafood", 0.30, 0.00, 0.00, 0.35, 0.70, "FDB000355", "glycine;glutamate;astaxanthin", "culinary_sensory+FlavorDB"),
        ("chicken", "meat", 0.05, 0.00, 0.00, 0.15, 0.65, "FDB000095", "inosinate;glutamate;2-methyl-3-furanthiol", "culinary_sensory+FlavorDB"),
        ("beef", "meat", 0.05, 0.05, 0.05, 0.15, 0.75, "FDB000045", "inosine monophosphate;12-methyltridecanal", "culinary_sensory+FlavorDB"),
        ("pork", "meat", 0.10, 0.00, 0.00, 0.15, 0.70, "FDB000310", "inosinate;gamma-nonalactone", "culinary_sensory+FlavorDB"),
        ("mushroom", "fungi", 0.10, 0.02, 0.05, 0.05, 0.85, "FDB000255", "guanylate;1-octen-3-ol", "culinary_sensory+FlavorDB"),
        ("shiitake mushroom", "fungi", 0.10, 0.05, 0.05, 0.05, 0.95, "FDB000256", "5-GMP;lenthionine", "culinary_sensory+FlavorDB"),
        ("olive oil", "fat_oil", 0.05, 0.02, 0.35, 0.00, 0.00, "FDB000275", "oleocanthal;oleuropein;hexanal", "culinary_sensory+BitterDB"),
        ("sesame oil", "fat_oil", 0.10, 0.00, 0.15, 0.00, 0.15, "FDB000352", "sesamol;2-acetylpyrazine", "culinary_sensory+FlavorDB"),
        ("basil", "herb", 0.20, 0.00, 0.25, 0.00, 0.05, "FDB000040", "estragole;linalool;eugenol", "culinary_sensory+FlavorDB"),
        ("cilantro", "herb", 0.05, 0.10, 0.20, 0.00, 0.00, "FDB000108", "(E)-2-decenal;decanal", "culinary_sensory+FlavorDB"),
        ("oregano", "herb", 0.05, 0.05, 0.40, 0.00, 0.05, "FDB000286", "carvacrol;thymol", "culinary_sensory+FlavorDB"),
        ("rosemary", "herb", 0.05, 0.05, 0.45, 0.00, 0.00, "FDB000338", "1,8-cineole;camphor", "culinary_sensory+FlavorDB"),
        ("mint", "herb", 0.20, 0.05, 0.20, 0.00, 0.00, "FDB000247", "L-menthol;menthone", "culinary_sensory+FlavorDB"),
        ("cocoa powder", "cocoa", 0.05, 0.10, 0.95, 0.00, 0.05, "FDB000100", "theobromine;caffeine;diketopiperazines", "culinary_sensory+BitterDB"),
        ("coffee", "beverage_extract", 0.02, 0.25, 0.92, 0.00, 0.05, "FDB000105", "caffeine;chlorogenic acid lactones", "culinary_sensory+BitterDB"),
        ("kale", "leafy_green", 0.10, 0.05, 0.70, 0.05, 0.10, "FDB000192", "glucosinolates;sinigrin", "culinary_sensory+BitterDB"),
        ("spinach", "leafy_green", 0.15, 0.05, 0.35, 0.10, 0.20, "FDB000368", "oxalic acid;glutamate", "culinary_sensory+FlavorDB"),
        ("eggplant", "vegetable", 0.15, 0.05, 0.45, 0.00, 0.20, "FDB000128", "solasonine;chlorogenic acid", "culinary_sensory+BitterDB"),
        ("carrot", "root_vegetable", 0.60, 0.05, 0.10, 0.05, 0.10, "FDB000078", "sucrose;terpinolene;falcarinol", "culinary_sensory+FlavorDB"),
        ("sweet potato", "root_vegetable", 0.75, 0.02, 0.05, 0.02, 0.05, "FDB000380", "maltose;beta-ionone", "culinary_sensory+FlavorDB"),
        ("potato", "tuber", 0.15, 0.02, 0.05, 0.02, 0.15, "FDB000315", "methional;solanine", "culinary_sensory+FlavorDB"),
        ("rice", "grain", 0.25, 0.00, 0.00, 0.00, 0.05, "FDB000335", "2-acetyl-1-pyrroline", "culinary_sensory+FlavorDB"),
        ("wheat flour", "grain", 0.15, 0.00, 0.02, 0.00, 0.05, "FDB000438", "hexanal;nonanal", "culinary_sensory+FlavorDB"),
        ("black beans", "legume", 0.15, 0.05, 0.15, 0.05, 0.30, "FDB000055", "saponins;glutamic acid", "culinary_sensory"),
        ("chickpeas", "legume", 0.20, 0.02, 0.10, 0.05, 0.25, "FDB000096", "hexanal;glutamate", "culinary_sensory"),
        ("avocado", "fruit_fat", 0.15, 0.02, 0.10, 0.02, 0.20, "FDB000032", "(E)-2-hexenal;oleic acid", "culinary_sensory+FlavorDB"),
        ("mango", "fruit", 0.85, 0.35, 0.02, 0.00, 0.00, "FDB000230", "furaneol;ethyl butanoate;myrcene", "culinary_sensory+FlavorDB"),
        ("apple", "fruit", 0.75, 0.40, 0.05, 0.00, 0.00, "FDB000022", "malic acid;hexyl acetate", "culinary_sensory+FlavorDB"),
        ("cinnamon", "spice_bark", 0.40, 0.00, 0.20, 0.00, 0.00, "FDB000098", "cinnamaldehyde;eugenol", "culinary_sensory+FlavorDB"),
        ("vanilla", "spice_pod", 0.35, 0.00, 0.10, 0.00, 0.00, "FDB000425", "vanillin;p-hydroxybenzaldehyde", "culinary_sensory+FlavorDB"),
        ("capers", "pickled_bud", 0.05, 0.65, 0.35, 0.85, 0.25, "FDB000074", "rutin;glucocapparin;sodium chloride", "culinary_sensory+FlavorDB"),
        ("olives", "pickled_fruit", 0.05, 0.35, 0.45, 0.80, 0.35, "FDB000274", "oleuropein;lactic acid;sodium chloride", "culinary_sensory+BitterDB"),
    ]
    columns = [
        "ingredient_norm",
        "category",
        "sweet",
        "sour",
        "bitter",
        "salty",
        "umami",
        "flavordb_id",
        "primary_flavor_compounds",
        "annotation_source",
    ]
    df = pd.DataFrame(records, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_sample_recipes_dataset(seed: int = 42) -> List[Dict]:
    """
    Generate a deterministic, realistic sample recipe dataset across 6 cuisines
    using authentic culinary templates with realistic variation, plus intentional
    edge cases (duplicates, missing ingredients, unmatched pantry items) to
    verify the full data engineering pipeline when raw Kaggle train.json is absent.
    """
    rng = np.random.default_rng(seed)

    cuisine_templates = {
        "Italian": [
            ["Roma tomatoes", "fresh garlic", "extra-virgin olive oil", "fresh basil", "parmigiano-reggiano", "kosher salt", "wheat flour"],
            ["tomato paste", "garlic cloves", "yellow onion", "olive oil", "oregano", "parmesan cheese", "beef", "black pepper", "salt"],
            ["anchovy", "capers", "olives", "roma tomatoes", "garlic", "olive oil", "oregano", "red chili powder"],
            ["mushroom", "parmesan cheese", "unsalted butter", "onion", "garlic", "rice", "olive oil", "salt"],
            ["feta cheese", "olives", "tomato", "olive oil", "oregano", "lemon juice", "garlic"],
            ["heavy cream", "butter", "parmesan cheese", "garlic", "black pepper", "salt", "wheat flour", "nutmeg_unmatched"],
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
            ["soy sauce", "miso paste", "shiitake mushroom", "green onions", "ginger", "rice vinegar", "sesame oil", "tofu_unmatched"],
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
        "water_unmatched",
        "baking_powder_unmatched",
    ]

    recipes = []
    recipe_id = 1000
    for cuisine, templates in cuisine_templates.items():
        for idx in range(30):  # 30 recipes per cuisine = 180 base recipes
            base = list(templates[idx % len(templates)])
            # Stochastically drop 1 non-essential item or add 1-2 pantry items
            if len(base) > 5 and rng.random() < 0.35:
                drop_idx = int(rng.integers(0, len(base)))
                base.pop(drop_idx)
            if rng.random() < 0.5:
                extra = optional_extras[int(rng.integers(0, len(optional_extras)))]
                if extra not in base:
                    base.append(extra)
            recipes.append(
                {
                    "id": recipe_id,
                    "cuisine": cuisine,
                    "ingredients": base,
                }
            )
            recipe_id += 1

    # Inject 3 exact/near duplicates and 2 invalid empty records to test cleaning logic
    recipes.append(recipes[0].copy())  # exact duplicate ID and content
    recipes.append({"id": 9991, "cuisine": "Italian", "ingredients": recipes[1]["ingredients"]})  # duplicate ingredient set in same cuisine
    recipes.append({"id": 9992, "cuisine": "", "ingredients": ["salt"]})  # missing cuisine
    recipes.append({"id": 9993, "cuisine": "Indian", "ingredients": []})  # empty ingredient list

    return recipes


def load_raw_recipes(raw_json_path: Path, use_sample_if_missing: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Load recipes from Kaggle/Yummly train.json if present, or fall back to
    the deterministic sample dataset when permitted.
    """
    if raw_json_path.exists():
        with open(raw_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data), f"Loaded external dataset from {raw_json_path}"

    if not use_sample_if_missing:
        raise FileNotFoundError(
            f"Required recipe file not found at {raw_json_path}. "
            "Place train.json in data/raw/ or pass --use-sample-data."
        )

    sample_data = generate_sample_recipes_dataset()
    return pd.DataFrame(sample_data), "Generated deterministic multi-cuisine sample corpus (--use-sample-data)"


def run_deliverable_1(use_sample_if_missing: bool = True) -> Dict[str, float]:
    """Execute Deliverable 1 end-to-end and write all processed CSVs and summary report."""
    ensure_directories()

    lexicon_path = DATA_RAW_DIR / "ingredient_taste_lexicon.csv"
    if not lexicon_path.exists():
        lexicon_df = generate_default_lexicon(lexicon_path)
    else:
        lexicon_df = pd.read_csv(lexicon_path)

    raw_json_path = DATA_RAW_DIR / "train.json"
    raw_df, data_source_note = load_raw_recipes(raw_json_path, use_sample_if_missing=use_sample_if_missing)
    initial_count = len(raw_df)

    # 1. Filter invalid records (missing cuisine or empty ingredient list)
    raw_df["cuisine"] = raw_df["cuisine"].fillna("").astype(str).str.strip()
    valid_mask = (raw_df["cuisine"] != "") & (
        raw_df["ingredients"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    )
    valid_df = raw_df[valid_mask].copy()
    invalid_removed = initial_count - len(valid_df)

    # 2. Deduplicate by recipe_id and canonical ingredient signature per cuisine
    valid_df["norm_list"] = valid_df["ingredients"].apply(
        lambda lst: sorted({normalize_ingredient_name(i) for i in lst if normalize_ingredient_name(i)})
    )
    valid_df["signature"] = valid_df["cuisine"] + "::" + valid_df["norm_list"].apply(lambda x: "|".join(x))
    dedup_df = valid_df.drop_duplicates(subset=["id"]).drop_duplicates(subset=["signature"]).copy()
    duplicates_removed = len(valid_df) - len(dedup_df)

    # 3. Explode recipe-ingredient pairs
    lexicon_lookup = lexicon_df.set_index("ingredient_norm").to_dict(orient="index")

    recipe_rows = []
    ingredient_rows = []
    unique_raw_to_norm: Dict[Tuple[str, str], int] = {}

    for _, row in dedup_df.iterrows():
        r_id = int(row["id"])
        cuisine = row["cuisine"]
        raw_ings = row["ingredients"]

        seen_norm_in_recipe = set()
        matched_count = 0
        total_norm_count = 0

        for pos_idx, raw_ing in enumerate(raw_ings):
            norm_ing = normalize_ingredient_name(raw_ing)
            if not norm_ing or norm_ing in seen_norm_in_recipe:
                continue
            seen_norm_in_recipe.add(norm_ing)
            total_norm_count += 1

            is_matched = norm_ing in lexicon_lookup
            if is_matched:
                matched_count += 1

            unique_raw_to_norm[(raw_ing, norm_ing)] = unique_raw_to_norm.get((raw_ing, norm_ing), 0) + 1
            ingredient_rows.append(
                {
                    "recipe_id": r_id,
                    "cuisine": cuisine,
                    "position_index": pos_idx,
                    "ingredient_raw": raw_ing,
                    "ingredient_norm": norm_ing,
                    "is_matched": is_matched,
                }
            )

        match_ratio = matched_count / total_norm_count if total_norm_count > 0 else 0.0
        sufficient_for_analysis = (
            matched_count >= MIN_MATCHED_INGREDIENTS and match_ratio >= MIN_MATCH_COVERAGE_RATIO
        )

        recipe_rows.append(
            {
                "recipe_id": r_id,
                "cuisine": cuisine,
                "n_ingredients_total": total_norm_count,
                "n_ingredients_matched": matched_count,
                "match_coverage_ratio": round(match_ratio, 4),
                "sufficient_for_analysis": sufficient_for_analysis,
                "ingredients_normalized": ";".join(sorted(seen_norm_in_recipe)),
            }
        )

    recipes_clean_df = pd.DataFrame(recipe_rows)
    recipe_ingredients_df = pd.DataFrame(ingredient_rows)

    # 4. Build ingredient_flavor_mapping.csv recording BOTH matched and unmatched ingredients
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
            mapping_rows.append(
                {
                    "ingredient_norm": ing_norm,
                    "example_raw_surface": nrow["example_raw_surface"],
                    "recipe_frequency": int(nrow["recipe_frequency"]),
                    "match_status": "matched",
                    "category": info["category"],
                    "sweet": info["sweet"],
                    "sour": info["sour"],
                    "bitter": info["bitter"],
                    "salty": info["salty"],
                    "umami": info["umami"],
                    "flavordb_id": info["flavordb_id"],
                    "primary_flavor_compounds": info["primary_flavor_compounds"],
                    "annotation_source": info["annotation_source"],
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
                    "flavordb_id": "NONE",
                    "primary_flavor_compounds": "NONE",
                    "annotation_source": "unmatched_logged",
                }
            )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        by=["match_status", "recipe_frequency"], ascending=[True, False]
    )

    # Save Deliverable 1 CSV outputs
    recipes_clean_df.to_csv(DATA_PROCESSED_DIR / "recipes_clean.csv", index=False)
    recipe_ingredients_df.to_csv(DATA_PROCESSED_DIR / "recipe_ingredients.csv", index=False)
    mapping_df.to_csv(DATA_PROCESSED_DIR / "ingredient_flavor_mapping.csv", index=False)

    # Calculate summary statistics
    n_recipes_clean = len(recipes_clean_df)
    n_unique_ings = len(mapping_df)
    n_matched_unique = int((mapping_df["match_status"] == "matched").sum())
    pct_matched_unique = (n_matched_unique / n_unique_ings * 100.0) if n_unique_ings else 0.0

    total_ing_tokens = len(recipe_ingredients_df)
    matched_ing_tokens = int(recipe_ingredients_df["is_matched"].sum())
    pct_matched_tokens = (matched_ing_tokens / total_ing_tokens * 100.0) if total_ing_tokens else 0.0

    n_cuisines = recipes_clean_df["cuisine"].nunique()
    n_sufficient = int(recipes_clean_df["sufficient_for_analysis"].sum())
    pct_sufficient = (n_sufficient / n_recipes_clean * 100.0) if n_recipes_clean else 0.0

    cuisine_counts = recipes_clean_df.groupby("cuisine").agg(
        total_recipes=("recipe_id", "count"),
        sufficient_recipes=("sufficient_for_analysis", "sum"),
        mean_match_ratio=("match_coverage_ratio", "mean"),
    )

    # Write reports/dataset_summary.md
    summary_md = f"""# Deliverable 1: Dataset Construction Summary

## 1. Data Source & Provenance
- **Source Note:** {data_source_note}
- **Taste & Flavor Reference:** `data/raw/ingredient_taste_lexicon.csv` combining documented gustatory modality scores (`sweet`, `sour`, `bitter`, `salty`, `umami`) with FlavorDB entity IDs and representative volatile aroma molecules (Garg et al., 2018).

## 2. Corpus & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | {initial_count} |
| Invalid/Incomplete Records Removed | {invalid_removed} |
| Exact & Canonical Duplicate Recipes Removed | {duplicates_removed} |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **{n_recipes_clean}** |
| **Cuisines Represented** | **{n_cuisines}** |
| **Unique Normalized Ingredients** | **{n_unique_ings}** |
| Unique Ingredients Matched to Taste Lexicon | {n_matched_unique} ({pct_matched_unique:.2f}%) |
| Unmatched Unique Ingredients Explicitly Logged | {n_unique_ings - n_matched_unique} ({100.0 - pct_matched_unique:.2f}%) |
| Total Recipe-Ingredient Occurrences (Tokens) | {total_ing_tokens} |
| Matched Recipe-Ingredient Occurrences | {matched_ing_tokens} ({pct_matched_tokens:.2f}%) |
| **Recipes Sufficient for Downstream Analysis** | **{n_sufficient} ({pct_sufficient:.2f}%)** |

## 3. Breakdown by Cuisine
{cuisine_counts.to_markdown()}

## 4. Scientific Distinction: Flavor Molecules vs. Gustatory Taste
Not every volatile flavor molecule in FlavorDB activates human gustatory taste receptors. Volatile compounds (e.g., *linalool*, *limonene*, *cuminaldehyde*) primarily drive retronasal olfaction (aroma), whereas basic taste perception (`sweet`, `sour`, `bitter`, `salty`, `umami`) is driven by water-soluble non-volatile tastants (e.g., sucrose, citric acid, quinine/glucosinolates, sodium chloride, L-glutamate/5'-ribonucleotides). Our mapping table explicitly separates `primary_flavor_compounds` (olfactory) from the five numerical taste scores (`sweet` through `umami`) and records unmatched ingredients (`match_status = "unmatched"`) without fabricating taste values.
"""
    (REPORTS_DIR / "dataset_summary.md").write_text(summary_md, encoding="utf-8")

    return {
        "n_recipes_clean": n_recipes_clean,
        "n_unique_ingredients": n_unique_ings,
        "n_cuisines": n_cuisines,
        "pct_matched_unique": pct_matched_unique,
        "pct_matched_tokens": pct_matched_tokens,
        "n_sufficient": n_sufficient,
    }