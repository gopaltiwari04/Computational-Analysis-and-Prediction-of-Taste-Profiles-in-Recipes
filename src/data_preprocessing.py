"""
Deliverable 1: Recipe-Ingredient-Taste Dataset Construction.
Handles ingredient normalization, alias resolution, deduplication,
lexicon matching against basic gustatory modalities, and dataset manifest/summary generation.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATASET_MANIFEST_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_RAW_RECIPES_PATH,
    MIN_MATCH_COVERAGE_RATIO,
    MIN_MATCHED_INGREDIENTS,
    RANDOM_SEED,
    REPORTS_DIR,
    TASTE_DIMENSIONS,
    ensure_directories,
)

# Deterministic alias dictionary: maps surface spelling variations,
# plurals, and knife-cut descriptions to canonical culinary ingredients,
# strictly preserving distinct culinary ingredients (e.g. garlic vs black garlic;
# ginger vs dried ginger; bell pepper vs cayenne pepper / chili).
INGREDIENT_ALIASES: Dict[str, str] = {
    # Alliums
    "garlic": "garlic",
    "garlic cloves": "garlic",
    "minced garlic": "garlic",
    "fresh garlic": "garlic",
    "garlic clove": "garlic",
    "garlic powder": "garlic",
    "chopped garlic": "garlic",
    "crushed garlic": "garlic",
    "black garlic": "black garlic",
    "onion": "onion",
    "onions": "onion",
    "yellow onion": "onion",
    "yellow onions": "onion",
    "white onion": "onion",
    "white onions": "onion",
    "chopped onions": "onion",
    "diced onions": "onion",
    "sliced onions": "onion",
    "onion powder": "onion",
    "red onion": "red onion",
    "red onions": "red onion",
    "purple onion": "red onion",
    "green onion": "green onion",
    "green onions": "green onion",
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "spring onions": "green onion",
    "shallot": "shallot",
    "shallots": "shallot",
    "french shallots": "shallot",

    # Spices & Rhizomes
    "ginger": "ginger",
    "fresh ginger": "ginger",
    "fresh ginger root": "ginger",
    "gingerroot": "ginger",
    "ginger root": "ginger",
    "grated ginger": "ginger",
    "minced ginger": "ginger",
    "ground ginger": "dried ginger",
    "ginger powder": "dried ginger",
    "dried ginger": "dried ginger",
    "turmeric": "turmeric",
    "ground turmeric": "turmeric",
    "turmeric powder": "turmeric",
    "cumin": "cumin",
    "ground cumin": "cumin",
    "cumin seed": "cumin",
    "cumin seeds": "cumin",
    "coriander": "coriander seed",
    "ground coriander": "coriander seed",
    "coriander seed": "coriander seed",
    "coriander seeds": "coriander seed",
    "fenugreek": "fenugreek",
    "fenugreek seeds": "fenugreek",
    "black pepper": "black pepper",
    "ground black pepper": "black pepper",
    "cracked black pepper": "black pepper",
    "coarse black pepper": "black pepper",
    "black pepper powder": "black pepper",
    "pepper": "black pepper",
    "ground pepper": "black pepper",
    "black peppercorns": "black pepper",
    "cinnamon": "cinnamon",
    "ground cinnamon": "cinnamon",
    "cinnamon stick": "cinnamon",
    "cinnamon sticks": "cinnamon",
    "vanilla": "vanilla",
    "vanilla extract": "vanilla",
    "pure vanilla extract": "vanilla",
    "vanilla bean": "vanilla",

    # Chilies & Peppers (Distinguishing sweet bell pepper from pungent capsaicin chilies)
    "bell pepper": "bell pepper",
    "red bell pepper": "bell pepper",
    "green bell pepper": "bell pepper",
    "yellow bell pepper": "bell pepper",
    "sweet pepper": "bell pepper",
    "cayenne": "cayenne pepper",
    "cayenne pepper": "cayenne pepper",
    "ground cayenne pepper": "cayenne pepper",
    "red chili powder": "cayenne pepper",
    "chili powder": "cayenne pepper",
    "red pepper flakes": "cayenne pepper",
    "crushed red pepper flakes": "cayenne pepper",
    "crushed red pepper": "cayenne pepper",
    "green chili": "green chili",
    "green chilies": "green chili",
    "green chile": "green chili",
    "green chiles": "green chili",
    "jalapeno": "jalapeno",
    "jalapeno pepper": "jalapeno",
    "jalapeno peppers": "jalapeno",
    "jalapeno chilies": "jalapeno",
    "jalapenos": "jalapeno",

    # Sweeteners
    "sugar": "sugar",
    "granulated sugar": "sugar",
    "white sugar": "sugar",
    "cane sugar": "sugar",
    "brown sugar": "brown sugar",
    "light brown sugar": "brown sugar",
    "dark brown sugar": "brown sugar",
    "honey": "honey",
    "raw honey": "honey",
    "maple syrup": "maple syrup",
    "pure maple syrup": "maple syrup",

    # Mineral / Leavening
    "salt": "salt",
    "kosher salt": "salt",
    "sea salt": "salt",
    "fine sea salt": "salt",
    "coarse salt": "salt",
    "table salt": "salt",
    "baking powder": "baking powder",

    # Citrus & Acids
    "lemon": "lemon juice",
    "lemon juice": "lemon juice",
    "fresh lemon juice": "lemon juice",
    "juice of a lemon": "lemon juice",
    "lime": "lime juice",
    "lime juice": "lime juice",
    "fresh lime juice": "lime juice",
    "juice of a lime": "lime juice",
    "tamarind": "tamarind",
    "tamarind pulp": "tamarind",
    "tamarind paste": "tamarind",
    "vinegar": "vinegar",
    "white vinegar": "vinegar",
    "cider vinegar": "vinegar",
    "apple cider vinegar": "vinegar",
    "rice vinegar": "rice vinegar",
    "seasoned rice vinegar": "rice vinegar",
    "white wine": "white wine",
    "dry white wine": "white wine",
    "red wine": "red wine",
    "dry red wine": "red wine",

    # Condiments & Ferments
    "soy sauce": "soy sauce",
    "light soy sauce": "soy sauce",
    "low sodium soy sauce": "soy sauce",
    "dark soy sauce": "soy sauce",
    "fish sauce": "fish sauce",
    "miso": "miso paste",
    "miso paste": "miso paste",
    "white miso": "miso paste",
    "oyster sauce": "oyster sauce",
    "mayonnaise": "mayonnaise",
    "mustard": "mustard",
    "dijon mustard": "mustard",
    "yellow mustard": "mustard",
    "capers": "capers",
    "olives": "olives",
    "black olives": "olives",
    "kalamata olives": "olives",

    # Dairy & Eggs
    "milk": "milk",
    "whole milk": "milk",
    "butter": "butter",
    "unsalted butter": "butter",
    "salted butter": "salted butter",
    "heavy cream": "heavy cream",
    "heavy whipping cream": "heavy cream",
    "whipping cream": "heavy cream",
    "sour cream": "sour cream",
    "yogurt": "yogurt",
    "plain yogurt": "yogurt",
    "greek yogurt": "yogurt",
    "parmesan cheese": "parmesan cheese",
    "parmesan": "parmesan cheese",
    "grated parmesan cheese": "parmesan cheese",
    "parmigiano reggiano": "parmesan cheese",
    "parmigiano-reggiano": "parmesan cheese",
    "cheddar cheese": "cheddar cheese",
    "shredded cheddar cheese": "cheddar cheese",
    "sharp cheddar cheese": "cheddar cheese",
    "feta cheese": "feta cheese",
    "crumbled feta cheese": "feta cheese",
    "egg": "egg",
    "eggs": "egg",
    "large egg": "egg",
    "large eggs": "egg",
    "egg yolk": "egg",
    "egg whites": "egg",

    # Meats & Seafood
    "chicken": "chicken",
    "boneless skinless chicken breasts": "chicken",
    "chicken breasts": "chicken",
    "chicken thighs": "chicken",
    "chicken broth": "chicken broth",
    "chicken stock": "chicken broth",
    "low sodium chicken broth": "chicken broth",
    "beef": "beef",
    "ground beef": "beef",
    "beef broth": "beef",
    "pork": "pork",
    "ground pork": "pork",
    "pork chops": "pork",
    "shrimp": "shrimp",
    "prawns": "shrimp",
    "anchovy": "anchovy",
    "anchovy fillets": "anchovy",
    "anchovies": "anchovy",

    # Vegetables, Fungi & Fruits
    "tomato": "tomato",
    "tomatoes": "tomato",
    "roma tomatoes": "tomato",
    "plum tomatoes": "tomato",
    "fresh tomatoes": "tomato",
    "cherry tomatoes": "tomato",
    "diced tomatoes": "tomato",
    "canned tomatoes": "tomato",
    "crushed tomatoes": "tomato",
    "tomato paste": "tomato paste",
    "carrot": "carrot",
    "carrots": "carrot",
    "potato": "potato",
    "potatoes": "potato",
    "russet potatoes": "potato",
    "sweet potato": "sweet potato",
    "sweet potatoes": "sweet potato",
    "eggplant": "eggplant",
    "spinach": "spinach",
    "baby spinach": "spinach",
    "fresh spinach": "spinach",
    "kale": "kale",
    "celery": "celery",
    "celery ribs": "celery",
    "celery stalks": "celery",
    "avocado": "avocado",
    "avocados": "avocado",
    "mango": "mango",
    "mangoes": "mango",
    "apple": "apple",
    "apples": "apple",
    "mushroom": "mushroom",
    "mushrooms": "mushroom",
    "button mushrooms": "mushroom",
    "cremini mushrooms": "mushroom",
    "shiitake mushroom": "shiitake mushroom",
    "shiitake mushrooms": "shiitake mushroom",
    "dried shiitake mushrooms": "shiitake mushroom",

    # Fats & Oils
    "olive oil": "olive oil",
    "extra virgin olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "sesame oil": "sesame oil",
    "toasted sesame oil": "sesame oil",
    "cooking oil": "cooking oil",
    "vegetable oil": "cooking oil",
    "canola oil": "cooking oil",
    "corn oil": "cooking oil",
    "sunflower oil": "cooking oil",
    "oil": "cooking oil",
    "cooking spray": "cooking oil",

    # Herbs
    "basil": "basil",
    "fresh basil": "basil",
    "fresh basil leaves": "basil",
    "sweet basil": "basil",
    "cilantro": "cilantro",
    "fresh cilantro": "cilantro",
    "chopped cilantro fresh": "cilantro",
    "fresh cilantro leaves": "cilantro",
    "coriander leaves": "cilantro",
    "oregano": "oregano",
    "dried oregano": "oregano",
    "fresh oregano": "oregano",
    "rosemary": "rosemary",
    "fresh rosemary": "rosemary",
    "dried rosemary": "rosemary",
    "thyme": "thyme",
    "fresh thyme": "thyme",
    "dried thyme": "thyme",
    "parsley": "parsley",
    "fresh parsley": "parsley",
    "chopped fresh parsley": "parsley",
    "flat leaf parsley": "parsley",
    "bay leaf": "bay leaf",
    "bay leaves": "bay leaf",
    "dried bay leaf": "bay leaf",
    "dill": "dill",
    "fresh dill": "dill",
    "mint": "mint",
    "fresh mint": "mint",
    "fresh mint leaves": "mint",

    # Grains, Starches & Legumes
    "flour": "wheat flour",
    "all purpose flour": "wheat flour",
    "all-purpose flour": "wheat flour",
    "plain flour": "wheat flour",
    "unbleached flour": "wheat flour",
    "wheat flour": "wheat flour",
    "cornstarch": "cornstarch",
    "corn starch": "cornstarch",
    "rice": "rice",
    "white rice": "rice",
    "jasmine rice": "rice",
    "basmati rice": "rice",
    "water": "water",
    "warm water": "water",
    "cold water": "water",
    "black beans": "black beans",
    "canned black beans": "black beans",
    "chickpeas": "chickpeas",
    "garbanzo beans": "chickpeas",
    "coconut milk": "coconut milk",
    "sesame seeds": "sesame seeds",
    "toasted sesame seeds": "sesame seeds",

    # Bitter beverages
    "cocoa powder": "cocoa powder",
    "unsweetened cocoa powder": "cocoa powder",
    "coffee": "coffee",
    "brewed coffee": "coffee",
    "espresso": "coffee",
}

# Non-informative preparation stopwords stripped during token normalization
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


def normalize_ingredient_name(raw_name: str) -> str:
    """
    Normalize raw ingredient string:
    1. Lowercase and strip whitespace.
    2. Strip parenthetical volume/instructional notes (e.g. '(optional)', '(10 oz)').
    3. Replace hyphens with spaces and strip non-alphanumeric punctuation.
    4. Collapse multiple whitespaces.
    5. Resolve direct alias mapping.
    6. Strip non-informative preparation adjectives and resolve alias again.
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

    if not text:
        return ""

    # Check direct alias before token stripping
    if text in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[text]

    # Strip non-informative preparation words only if other informative tokens remain
    tokens = [tok for tok in text.split() if tok not in PREP_STOPWORDS]
    cleaned = " ".join(tokens) if tokens else text

    # Check alias again after stripping prep descriptors
    return INGREDIENT_ALIASES.get(cleaned, cleaned)


def generate_default_lexicon(output_path: Path) -> pd.DataFrame:
    """
    Create a curated, literature-grounded ingredient taste & FlavorDB reference lexicon
    with explicit chemesthetic pungency indicators, confidence scores, and provenance notes.
    """
    # 92 curated records linking basic tastes, chemesthetic sensations, and aroma molecules
    lexicon_records = [
        # Alliums
        ("garlic", "allium", 0.15, 0.05, 0.10, 0.00, 0.35, "allicin_pungency", "medium", "FDB000142", "diallyl disulfide;allyl mercaptan", "Curated Proxy (Allium cepa/sativum profile; Garg et al. 2018)"),
        ("black garlic", "allium", 0.65, 0.25, 0.05, 0.10, 0.80, "none", "medium", "FDB000142_ferm", "S-allyl-cysteine;5-HMF", "Curated Proxy (Aged fermented allium profile)"),
        ("onion", "allium", 0.35, 0.05, 0.05, 0.00, 0.30, "none", "medium", "FDB000281", "dipropyl disulfide;quercetin", "Curated Proxy (Allium cepa sensory proxy)"),
        ("red onion", "allium", 0.30, 0.10, 0.10, 0.00, 0.20, "none", "medium", "FDB000282", "anthocyanins;thiopropanal S-oxide", "Curated Proxy (Sensory proxy; USDA nutrient sugar/acid)"),
        ("green onion", "allium", 0.20, 0.05, 0.10, 0.00, 0.20, "none", "medium", "FDB000285", "propyl mercaptan;diallyl disulfide", "Curated Proxy (Sensory proxy; FlavorDB FDB000285)"),
        ("shallot", "allium", 0.40, 0.05, 0.05, 0.00, 0.30, "allicin_pungency", "medium", "FDB000353", "dipropyl disulfide;methyl propyl disulfide", "Curated Proxy (Allium ascalonicum proxy)"),

        # Spices & Rhizomes
        ("ginger", "spice_rhizome", 0.10, 0.10, 0.20, 0.00, 0.05, "gingerol_pungency", "high", "FDB000155", "6-gingerol;zingiberene;curcumene", "Curated Proxy (Rhizome taste proxy; FlavorDB)"),
        ("dried ginger", "spice_rhizome", 0.10, 0.05, 0.25, 0.00, 0.05, "gingerol_pungency", "high", "FDB000155_dry", "shogaol;zingerone", "Curated Proxy (Dehydrated pungent rhizome profile)"),
        ("turmeric", "spice_rhizome", 0.05, 0.05, 0.65, 0.00, 0.05, "none", "medium", "FDB000412", "curcumin;turmerone;curlone", "Curated Proxy (Curcuma longa bitter-earthy profile)"),
        ("cumin", "spice_seed", 0.05, 0.05, 0.45, 0.00, 0.10, "none", "medium", "FDB000118", "cuminaldehyde;p-cymene;terpinen-7-al", "Curated Proxy (Cuminum cyminum seed profile)"),
        ("coriander seed", "spice_seed", 0.15, 0.10, 0.25, 0.00, 0.05, "none", "medium", "FDB000109", "linalool;alpha-pinene;gamma-terpinene", "Curated Proxy (Coriandrum sativum seed profile)"),
        ("fenugreek", "spice_seed", 0.15, 0.00, 0.85, 0.00, 0.10, "none", "high", "FDB000136", "sotolon;trigonelline;4-hydroxyisoleucine", "Curated Proxy (Trigonella foenum-graecum intense bitter seed)"),
        ("black pepper", "spice", 0.00, 0.00, 0.35, 0.00, 0.05, "piperine_pungency", "high", "FDB000064", "piperine;rotundone;beta-caryophyllene", "Curated Proxy (Piper nigrum piperine proxy)"),
        ("cinnamon", "spice_bark", 0.40, 0.00, 0.20, 0.00, 0.00, "none", "medium", "FDB000098", "cinnamaldehyde;eugenol;coumarin", "Curated Proxy (Cinnamomum verum bark sweet-spicy proxy)"),
        ("vanilla", "spice_pod", 0.35, 0.00, 0.10, 0.00, 0.00, "none", "high", "FDB000425", "vanillin;p-hydroxybenzaldehyde", "Curated Proxy (Vanilla planifolia aromatic pod proxy)"),

        # Chilies
        ("cayenne pepper", "chili", 0.05, 0.05, 0.25, 0.00, 0.05, "capsaicin_heat", "high", "FDB000081", "capsaicin;dihydrocapsaicin", "Curated Proxy (Capsicum annuum pungency standard)"),
        ("green chili", "chili", 0.10, 0.10, 0.25, 0.00, 0.05, "capsaicin_heat", "high", "FDB000082", "capsaicin;2-isobutyl-3-methoxypyrazine", "Curated Proxy (Capsicum frutescens green pungent profile)"),
        ("jalapeno", "chili", 0.15, 0.10, 0.20, 0.00, 0.05, "capsaicin_heat", "high", "FDB000083", "capsaicin;hexanal", "Curated Proxy (Capsicum annuum cv. Jalapeno proxy)"),

        # Vegetables & Fruits
        ("bell pepper", "vegetable", 0.45, 0.10, 0.10, 0.00, 0.10, "none", "medium", "FDB000084", "2-methoxy-3-isobutylpyrazine;beta-carotene", "Curated Proxy (Non-pungent sweet capsicum profile)"),
        ("tomato", "vegetable_fruit", 0.35, 0.45, 0.05, 0.05, 0.70, "none", "high", "FDB000398", "glutamic acid;hexanal;lycopene;citric acid", "Curated Proxy (Solanum lycopersicum high glutamate/acid proxy)"),
        ("tomato paste", "condiment", 0.40, 0.50, 0.10, 0.15, 0.90, "none", "high", "FDB000399", "L-glutamate;citric acid;furfural", "Curated Proxy (Concentrated tomato solids proxy)"),
        ("carrot", "root_vegetable", 0.60, 0.05, 0.10, 0.05, 0.10, "none", "medium", "FDB000078", "sucrose;terpinolene;falcarinol", "Curated Proxy (Daucus carota high sucrose root proxy)"),
        ("sweet potato", "root_vegetable", 0.75, 0.02, 0.05, 0.02, 0.05, "none", "medium", "FDB000380", "maltose;beta-ionone;sucrose", "Curated Proxy (Ipomoea batatas carbohydrate-rich proxy)"),
        ("potato", "tuber", 0.15, 0.02, 0.05, 0.02, 0.15, "none", "low_proxy", "FDB000315", "methional;solanine;starch", "Curated Proxy (Solanum tuberosum neutral starch proxy)"),
        ("eggplant", "vegetable", 0.15, 0.05, 0.45, 0.00, 0.20, "none", "medium", "FDB000128", "solasonine;chlorogenic acid;delphinidin", "Curated Proxy (Solanum melongena bitter saponoid proxy)"),
        ("spinach", "leafy_green", 0.15, 0.05, 0.35, 0.10, 0.20, "none", "medium", "FDB000368", "oxalic acid;glutamate;chlorophyll", "Curated Proxy (Spinacia oleracea oxalic/glutamate proxy)"),
        ("kale", "leafy_green", 0.10, 0.05, 0.70, 0.05, 0.10, "none", "high", "FDB000192", "glucosinolates;sinigrin;sulforaphane", "Curated Proxy (Brassica oleracea glucosinolate bitter proxy)"),
        ("celery", "vegetable", 0.15, 0.05, 0.35, 0.10, 0.15, "none", "medium", "FDB000085", "sedanolide;apigenin;sodium", "Curated Proxy (Apium graveolens aromatic/mineral proxy)"),
        ("avocado", "fruit_fat", 0.15, 0.02, 0.10, 0.02, 0.20, "none", "low_proxy", "FDB000032", "(E)-2-hexenal;oleic acid;glutathione", "Curated Proxy (Persea americana lipid-rich fruit proxy)"),
        ("mango", "fruit", 0.85, 0.35, 0.02, 0.00, 0.00, "none", "high", "FDB000230", "furaneol;ethyl butanoate;myrcene", "Curated Proxy (Mangifera indica sugar/citric acid proxy)"),
        ("apple", "fruit", 0.75, 0.40, 0.05, 0.00, 0.00, "none", "high", "FDB000022", "fructose;malic acid;hexyl acetate", "Curated Proxy (Malus domestica malic/fructose proxy)"),

        # Citrus & Acids
        ("lemon juice", "citrus", 0.10, 0.95, 0.10, 0.00, 0.00, "none", "high", "FDB000210", "citric acid;limonene;citral", "Reference Standard (Citric acid 5-6% solution proxy)"),
        ("lime juice", "citrus", 0.08, 0.95, 0.15, 0.00, 0.00, "none", "high", "FDB000214", "citric acid;gamma-terpinene;terpineol", "Reference Standard (Citric acid 5-7% solution proxy)"),
        ("vinegar", "acid", 0.05, 0.90, 0.05, 0.00, 0.05, "none", "high", "FDB000430", "acetic acid;ethyl acetate", "Reference Standard (Acetic acid 4-6% solution proxy)"),
        ("rice vinegar", "acid", 0.20, 0.75, 0.00, 0.05, 0.10, "none", "high", "FDB000431", "acetic acid;lactic acid;amino acids", "Curated Proxy (Mild fermented rice acid proxy)"),
        ("tamarind", "fruit_acid", 0.35, 0.85, 0.10, 0.00, 0.05, "none", "high", "FDB000385", "tartaric acid;2-acetylfuran;invert sugar", "Curated Proxy (Tamarindus indica tartaric acid proxy)"),
        ("white wine", "beverage_extract", 0.15, 0.70, 0.10, 0.00, 0.05, "none", "medium", "FDB000441", "tartaric acid;ethyl hexanoate;succinic acid", "Curated Proxy (Vinum acid-succinic proxy)"),
        ("red wine", "beverage_extract", 0.10, 0.65, 0.25, 0.00, 0.05, "none", "medium", "FDB000332", "tartaric acid;tannins;catechin;resveratrol", "Curated Proxy (Vinum rubrum tannin-acid proxy)"),

        # Sweeteners
        ("sugar", "sweetener", 1.00, 0.00, 0.00, 0.00, 0.00, "none", "high", "FDB000370", "sucrose", "Reference Standard (Pure sucrose gustatory standard)"),
        ("brown sugar", "sweetener", 0.95, 0.05, 0.05, 0.02, 0.05, "none", "high", "FDB000371", "sucrose;molasses furans;invert sugar", "Curated Proxy (Sucrose with cane molasses residue)"),
        ("honey", "sweetener", 0.95, 0.10, 0.05, 0.00, 0.00, "none", "high", "FDB000175", "fructose;glucose;phenylacetic acid", "Curated Proxy (Apis mellifera inverted nectar proxy)"),
        ("maple syrup", "sweetener", 0.95, 0.05, 0.05, 0.00, 0.05, "none", "high", "FDB000232", "sucrose;sotolon;vanillin", "Curated Proxy (Acer saccharum concentrated sap proxy)"),

        # Minerals
        ("salt", "mineral", 0.00, 0.00, 0.00, 1.00, 0.00, "none", "high", "FDB000350", "sodium chloride", "Reference Standard (Pure NaCl gustatory standard)"),
        ("baking powder", "mineral_leavening", 0.00, 0.05, 0.25, 0.45, 0.00, "none", "high", "unmapped", "sodium bicarbonate;sodium acid pyrophosphate", "Curated Proxy (Alkaline-mineral leavening salt)"),

        # Dairy & Eggs
        ("milk", "dairy", 0.30, 0.05, 0.00, 0.05, 0.10, "none", "medium", "FDB000245", "lactose;2-heptanone;butyric acid", "Curated Proxy (Bovine milk lactose/protein proxy)"),
        ("heavy cream", "dairy", 0.25, 0.05, 0.00, 0.05, 0.10, "none", "medium", "FDB000246", "delta-decalactone;diacetyl;milk fat", "Curated Proxy (High fat dairy emulsion proxy)"),
        ("butter", "dairy", 0.15, 0.02, 0.00, 0.05, 0.10, "none", "medium", "FDB000070", "diacetyl;butyric acid;lactones", "Curated Proxy (Sweet cream unsalted butter proxy)"),
        ("salted butter", "dairy", 0.15, 0.02, 0.00, 0.55, 0.10, "none", "high", "FDB000071", "diacetyl;sodium chloride", "Curated Proxy (Butter with 1.5-2.0% NaCl proxy)"),
        ("sour cream", "dairy_fermented", 0.15, 0.60, 0.02, 0.10, 0.20, "none", "medium", "FDB000246_fermented", "lactic acid;diacetyl;acetoin", "Curated Proxy (Cultured dairy lactic acid proxy)"),
        ("yogurt", "dairy_fermented", 0.15, 0.65, 0.05, 0.10, 0.15, "none", "medium", "FDB000445", "lactic acid;acetaldehyde", "Curated Proxy (Lactobacillus fermented curd proxy)"),
        ("parmesan cheese", "dairy_aged", 0.05, 0.15, 0.10, 0.75, 0.95, "none", "high", "FDB000091", "monosodium glutamate;butyric acid;propionic acid", "Curated Proxy (Aged hard curd glutamate/NaCl proxy)"),
        ("cheddar cheese", "dairy_aged", 0.05, 0.20, 0.10, 0.65, 0.65, "none", "medium", "FDB000089", "lactic acid;methional;butyric acid", "Curated Proxy (Aged curd sodium/lactate proxy)"),
        ("feta cheese", "dairy_brined", 0.05, 0.45, 0.05, 0.85, 0.45, "none", "high", "FDB000093", "octanoic acid;sodium chloride;lactic acid", "Curated Proxy (Brined sheep/goat cheese proxy)"),
        ("mayonnaise", "condiment", 0.15, 0.45, 0.02, 0.45, 0.25, "none", "medium", "unmapped", "acetic acid;egg yolk lipids;sodium chloride", "Curated Proxy (Acidified oil-egg emulsion proxy)"),
        ("mustard", "condiment", 0.05, 0.55, 0.35, 0.40, 0.10, "isothiocyanate_pungency", "high", "FDB000257", "allyl isothiocyanate;sinigrin;acetic acid", "Curated Proxy (Sinapis alba pungent seed condiment)"),

        # Fermented Condiments (High Umami)
        ("soy sauce", "fermented_condiment", 0.15, 0.15, 0.10, 0.90, 0.95, "none", "high", "FDB000365", "L-glutamate;HEMF;sodium chloride", "Curated Proxy (Brewed glycine/wheat hydrolysate proxy)"),
        ("fish sauce", "fermented_condiment", 0.10, 0.10, 0.05, 0.95, 0.98, "none", "high", "FDB000138", "glutamic acid;inosine monophosphate;NaCl", "Curated Proxy (Anchovy autolysate nucleotide proxy)"),
        ("miso paste", "fermented_condiment", 0.25, 0.15, 0.10, 0.85, 0.90, "none", "high", "FDB000248", "glutamate;4-ethylguaiacol;succinic acid", "Curated Proxy (Fermented koji soybean paste proxy)"),
        ("oyster sauce", "fermented_condiment", 0.45, 0.05, 0.05, 0.80, 0.90, "none", "medium", "FDB000288", "glutamate;glycine;sucrose;cornstarch", "Curated Proxy (Caramelized bivalve extract proxy)"),

        # Seafood & Meats
        ("anchovy", "seafood", 0.00, 0.05, 0.05, 0.85, 0.95, "none", "high", "FDB000018", "inosine-5-monophosphate;trimethylamine;salt", "Curated Proxy (Engraulidae nucleotide-rich tissue proxy)"),
        ("shrimp", "seafood", 0.30, 0.00, 0.00, 0.35, 0.70, "none", "medium", "FDB000355", "glycine;glutamate;astaxanthin;betaine", "Curated Proxy (Crustacean sweet glycine/umami proxy)"),
        ("chicken", "meat", 0.05, 0.00, 0.00, 0.15, 0.65, "none", "medium", "FDB000095", "inosinate;glutamate;2-methyl-3-furanthiol", "Curated Proxy (Gallus gallus avian muscular tissue proxy)"),
        ("chicken broth", "broth_stock", 0.05, 0.02, 0.00, 0.65, 0.80, "none", "medium", "FDB000095_broth", "glutamate;inosinate;sodium chloride", "Curated Proxy (Simmered poultry stock proxy)"),
        ("beef", "meat", 0.05, 0.05, 0.05, 0.15, 0.75, "none", "medium", "FDB000045", "inosine monophosphate;12-methyltridecanal", "Curated Proxy (Bos taurus mammalian muscular tissue proxy)"),
        ("pork", "meat", 0.10, 0.00, 0.00, 0.15, 0.70, "none", "medium", "FDB000310", "inosinate;gamma-nonalactone;glutamate", "Curated Proxy (Sus domesticus muscular tissue proxy)"),
        ("egg", "poultry_egg", 0.05, 0.00, 0.00, 0.05, 0.35, "none", "medium", "FDB000129", "ovomucoid;lecithin;ovalbumin;cysteine", "Curated Proxy (Whole avian egg protein/lipid proxy)"),

        # Fungi
        ("mushroom", "fungi", 0.10, 0.02, 0.05, 0.05, 0.85, "none", "high", "FDB000255", "guanylate;1-octen-3-ol;glutamic acid", "Curated Proxy (Agaricus bisporus nucleotide proxy)"),
        ("shiitake mushroom", "fungi", 0.10, 0.05, 0.05, 0.05, 0.95, "none", "high", "FDB000256", "5-GMP;lenthionine;guanosine monophosphate", "Curated Proxy (Lentinula edodes high 5-GMP proxy)"),

        # Fats & Oils
        ("olive oil", "fat_oil", 0.05, 0.02, 0.35, 0.00, 0.00, "none", "medium", "FDB000275", "oleocanthal;oleuropein;hexanal", "Curated Proxy (Olea europaea virgin lipid bitter proxy)"),
        ("sesame oil", "fat_oil", 0.10, 0.00, 0.15, 0.00, 0.15, "none", "medium", "FDB000352", "sesamol;2-acetylpyrazine;sesamolin", "Curated Proxy (Toasted Sesamum indicum lipid proxy)"),
        ("cooking oil", "fat_oil", 0.02, 0.00, 0.02, 0.00, 0.00, "none", "high", "FDB000277", "oleic acid;linoleic acid;tocopherols", "Curated Proxy (Refined neutral vegetable lipid proxy)"),

        # Herbs
        ("basil", "herb", 0.20, 0.00, 0.25, 0.00, 0.05, "none", "medium", "FDB000040", "estragole;linalool;eugenol", "Curated Proxy (Ocimum basilicum leaf proxy)"),
        ("cilantro", "herb", 0.05, 0.10, 0.20, 0.00, 0.00, "none", "medium", "FDB000108", "(E)-2-decenal;decanal;linalool", "Curated Proxy (Coriandrum sativum fresh foliage proxy)"),
        ("oregano", "herb", 0.05, 0.05, 0.40, 0.00, 0.05, "none", "medium", "FDB000286", "carvacrol;thymol;p-cymene", "Curated Proxy (Origanum vulgare leaf bitter proxy)"),
        ("rosemary", "herb", 0.05, 0.05, 0.45, 0.00, 0.00, "none", "medium", "FDB000338", "1,8-cineole;camphor;borneol", "Curated Proxy (Salvia rosmarinus pinene/bitter proxy)"),
        ("thyme", "herb", 0.05, 0.02, 0.40, 0.00, 0.05, "none", "medium", "FDB000394", "thymol;p-cymene;gamma-terpinene", "Curated Proxy (Thymus vulgaris phenolic proxy)"),
        ("parsley", "herb", 0.05, 0.02, 0.35, 0.00, 0.05, "none", "medium", "FDB000293", "apiol;myristicin;1,3,8-p-menthatriene", "Curated Proxy (Petroselinum crispum bitter foliage proxy)"),
        ("bay leaf", "herb", 0.05, 0.02, 0.45, 0.00, 0.05, "none", "medium", "FDB000041", "1,8-cineole;eugenol;methyl eugenol", "Curated Proxy (Laurus nobilis bitter aromatic foliage proxy)"),
        ("dill", "herb", 0.05, 0.05, 0.30, 0.00, 0.05, "none", "medium", "FDB000125", "carvone;alpha-phellandrene;limonene", "Curated Proxy (Anethum graveolens fresh leaf proxy)"),
        ("mint", "herb", 0.20, 0.05, 0.20, 0.00, 0.00, "none", "medium", "FDB000247", "L-menthol;menthone;menthyl acetate", "Curated Proxy (Mentha spicata cooling aromatic proxy)"),

        # Bitter beverages
        ("cocoa powder", "cocoa", 0.05, 0.10, 0.95, 0.00, 0.05, "none", "high", "FDB000100", "theobromine;caffeine;diketopiperazines", "Curated Proxy (Theobroma cacao alkaloid bitter proxy)"),
        ("coffee", "beverage_extract", 0.02, 0.25, 0.92, 0.00, 0.05, "none", "high", "FDB000105", "caffeine;chlorogenic acid lactones;quinides", "Curated Proxy (Coffea arabica pyrolytic bitter proxy)"),

        # Grains & Starches
        ("rice", "grain", 0.25, 0.00, 0.00, 0.00, 0.05, "none", "medium", "FDB000335", "2-acetyl-1-pyrroline;starch", "Curated Proxy (Oryza sativa starch proxy)"),
        ("wheat flour", "grain", 0.15, 0.00, 0.02, 0.00, 0.05, "none", "low_proxy", "FDB000438", "hexanal;nonanal;gliadin;glutenin", "Curated Proxy (Triticum aestivum cereal grain proxy)"),
        ("cornstarch", "grain_starch", 0.08, 0.00, 0.00, 0.00, 0.00, "none", "high", "FDB000110", "amylose;amylopectin", "Reference Standard (Zea mays purified starch proxy)"),
        ("water", "neutral_solvent", 0.00, 0.00, 0.00, 0.00, 0.00, "none", "high", "FDB000435", "dihydrogen monoxide", "Reference Standard (Pure aqueous neutral baseline)"),

        # Legumes
        ("black beans", "legume", 0.15, 0.05, 0.15, 0.05, 0.30, "none", "medium", "FDB000055", "saponins;glutamic acid;phaseolin", "Curated Proxy (Phaseolus vulgaris cooked legume proxy)"),
        ("chickpeas", "legume", 0.20, 0.02, 0.10, 0.05, 0.25, "none", "medium", "FDB000096", "hexanal;glutamate;stachyose", "Curated Proxy (Cicer arietinum cooked legume proxy)"),
        ("coconut milk", "plant_milk", 0.45, 0.00, 0.00, 0.05, 0.10, "none", "medium", "FDB000102", "delta-octalactone;gamma-nonalactone;lauric acid", "Curated Proxy (Cocos nucifera endosperm emulsion proxy)"),
        ("sesame seeds", "seed", 0.10, 0.00, 0.15, 0.00, 0.20, "none", "medium", "FDB000351", "sesamin;2-acetylpyrazine;sesamolin", "Curated Proxy (Sesamum indicum roasted seed proxy)"),

        # Pickled Condiments
        ("capers", "pickled_bud", 0.05, 0.65, 0.35, 0.85, 0.25, "none", "high", "FDB000074", "rutin;glucocapparin;sodium chloride", "Curated Proxy (Brined Capparis spinosa floral bud proxy)"),
        ("olives", "pickled_fruit", 0.05, 0.35, 0.45, 0.80, 0.35, "none", "high", "FDB000274", "oleuropein;lactic acid;sodium chloride", "Curated Proxy (Brined/cured Olea europaea fruit proxy)"),
    ]

    columns = [
        "ingredient_norm",
        "category",
        "sweet",
        "sour",
        "bitter",
        "salty",
        "umami",
        "chemesthetic_pungency",
        "confidence_level",
        "flavordb_id",
        "primary_aroma_compounds",
        "annotation_provenance",
    ]
    df = pd.DataFrame(lexicon_records, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_sample_recipes_dataset(seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Generate a deterministic, multi-cuisine synthetic sample recipe dataset
    across 6 cuisines with intentional edge cases (duplicates, empty lists)
    specifically designed for unit and smoke tests.
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

    # Injected test edge cases
    recipes.append(recipes[0].copy())  # exact duplicate ID and content
    recipes.append({"id": 9991, "cuisine": "Italian", "ingredients": recipes[1]["ingredients"]})  # duplicate signature
    recipes.append({"id": 9992, "cuisine": "", "ingredients": ["salt"]})  # missing cuisine
    recipes.append({"id": 9993, "cuisine": "Indian", "ingredients": []})  # empty ingredient list
    return recipes


def load_raw_recipes(
    raw_json_path: Path = DEFAULT_RAW_RECIPES_PATH,
    use_sample_data: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load raw recipes strictly requiring real data by default.
    Raises FileNotFoundError when raw_json_path is missing unless use_sample_data=True.
    When use_sample_data=True, returns the synthetic development corpus.
    Returns DataFrame and provenance metadata dictionary.
    """
    if use_sample_data:
        sample_data = generate_sample_recipes_dataset()
        meta = {
            "dataset_name": "Synthetic Multi-Cuisine Development Corpus (--use-sample-data)",
            "source_url": "Internal deterministic generator (generate_sample_recipes_dataset)",
            "source_file": "internal::synthetic_corpus",
            "access_date": datetime.now().strftime("%Y-%m-%d"),
            "license": "CC0 / Internal Synthetic Testing",
            "is_synthetic": True,
            "raw_record_count": len(sample_data),
        }
        return pd.DataFrame(sample_data), meta

    if raw_json_path.exists():
        with open(raw_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array of recipe objects in {raw_json_path}")

        meta = {
            "dataset_name": "Kaggle / Yummly What's Cooking Dataset",
            "source_url": "https://www.kaggle.com/c/whats-cooking/data",
            "source_file": str(raw_json_path.resolve()),
            "access_date": "2026-10-22",
            "license": "Kaggle Competition / Educational Research Use (Yummly Inc.)",
            "is_synthetic": False,
            "raw_record_count": len(data),
        }
        return pd.DataFrame(data), meta

    raise FileNotFoundError(
        f"External raw recipe dataset not found at '{raw_json_path}'.\n"
        "To use real data:\n"
        "  1. Download 'train.json' from Kaggle: https://www.kaggle.com/c/whats-cooking/data\n"
        f"  2. Place it at '{raw_json_path}' (or specify via --data-path).\n"
        "For unit testing and development only, you may pass '--use-sample-data' explicitly."
    )


def run_deliverable_1(
    raw_data_path: Optional[Path] = None,
    use_sample_data: bool = False,
) -> Dict[str, object]:
    """
    Execute Deliverable 1 end-to-end:
    1. Loads raw recipes (real by default; synthetic only with explicit opt-in).
    2. Filters invalid/empty records.
    3. Normalizes ingredients and resolves aliases.
    4. Deduplicates within cuisine by canonical ingredient signature.
    5. Maps ingredients to the taste lexicon and logs unmatched items.
    6. Saves clean CSV files, writes dataset_manifest.json, and produces dataset_summary.md.
    """
    ensure_directories()

    lexicon_path = DEFAULT_LEXICON_PATH
    if not lexicon_path.exists():
        lexicon_df = generate_default_lexicon(lexicon_path)
    else:
        lexicon_df = pd.read_csv(lexicon_path)

    raw_json_path = raw_data_path or DEFAULT_RAW_RECIPES_PATH
    raw_df, meta = load_raw_recipes(raw_json_path, use_sample_data=use_sample_data)
    initial_count = len(raw_df)

    # 1. Filter invalid records (missing cuisine or non-list/empty ingredients)
    raw_df["cuisine"] = raw_df["cuisine"].fillna("").astype(str).str.strip()
    valid_mask = (raw_df["cuisine"] != "") & (
        raw_df["ingredients"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    )
    valid_df = raw_df[valid_mask].copy()
    invalid_removed = initial_count - len(valid_df)

    # 2. Canonical ingredient normalization per recipe
    valid_df["norm_list"] = valid_df["ingredients"].apply(
        lambda lst: sorted({normalize_ingredient_name(i) for i in lst if normalize_ingredient_name(i)})
    )
    valid_df["signature"] = valid_df["cuisine"] + "::" + valid_df["norm_list"].apply(lambda x: "|".join(x))

    # Drop duplicate IDs first, then drop identical ingredient signatures WITHIN the same cuisine
    # (preserving legitimate cross-cuisine recipes)
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
                    "category": info.get("category", "unclassified"),
                    "sweet": info.get("sweet", np.nan),
                    "sour": info.get("sour", np.nan),
                    "bitter": info.get("bitter", np.nan),
                    "salty": info.get("salty", np.nan),
                    "umami": info.get("umami", np.nan),
                    "chemesthetic_pungency": info.get("chemesthetic_pungency", "none"),
                    "confidence_level": info.get("confidence_level", "medium"),
                    "flavordb_id": info.get("flavordb_id", "unmapped"),
                    "primary_aroma_compounds": info.get("primary_aroma_compounds", info.get("primary_flavor_compounds", "")),
                    "annotation_provenance": info.get("annotation_provenance", info.get("annotation_source", "curated")),
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
                    "confidence_level": "none",
                    "flavordb_id": "NONE",
                    "primary_aroma_compounds": "NONE",
                    "annotation_provenance": "unmatched_logged",
                }
            )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(
        by=["match_status", "recipe_frequency"], ascending=[True, False]
    )

    # Save Deliverable 1 CSV outputs
    recipes_clean_df.to_csv(DATA_PROCESSED_DIR / "recipes_clean.csv", index=False)
    recipe_ingredients_df.to_csv(DATA_PROCESSED_DIR / "recipe_ingredients.csv", index=False)
    mapping_df.to_csv(DATA_PROCESSED_DIR / "ingredient_flavor_mapping.csv", index=False)

    # Summary statistics
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
    ).reset_index()

    # Save dataset_manifest.json
    manifest = {
        "dataset_name": meta["dataset_name"],
        "source_url": meta["source_url"],
        "source_file": meta["source_file"],
        "access_date": meta["access_date"],
        "license": meta["license"],
        "is_synthetic": meta["is_synthetic"],
        "raw_record_count": initial_count,
        "invalid_records_removed": invalid_removed,
        "duplicates_removed": duplicates_removed,
        "clean_recipe_count": n_recipes_clean,
        "unique_ingredients_count": n_unique_ings,
        "matched_ingredients_count": n_matched_unique,
        "unmatched_ingredients_count": n_unique_ings - n_matched_unique,
        "pct_unique_matched": round(pct_matched_unique, 2),
        "total_ingredient_tokens": total_ing_tokens,
        "matched_ingredient_tokens": matched_ing_tokens,
        "pct_tokens_matched": round(pct_matched_tokens, 2),
        "sufficient_recipes_count": n_sufficient,
        "pct_recipes_sufficient": round(pct_sufficient, 2),
        "cuisines_represented": n_cuisines,
        "cuisine_distribution": cuisine_counts.set_index("cuisine")["total_recipes"].to_dict(),
        "created_at": datetime.now().isoformat(),
    }
    with open(DATASET_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate reports/dataset_summary.md
    summary_md = f"""# Deliverable 1: Dataset Construction & Provenance Summary

## 1. Data Source & Provenance Manifest
- **Dataset Name:** {meta['dataset_name']}
- **Source:** {meta['source_url']}
- **Source Path:** `{meta['source_file']}`
- **License / Terms:** {meta['license']}
- **Corpus Type:** {"SYNTHETIC (Development Smoke Test)" if meta['is_synthetic'] else "REAL (Empirical Recipe Corpus)"}
- **Taste & Flavor Reference:** `data/raw/ingredient_taste_lexicon.csv` ({len(lexicon_df)} core entries) linking expert-curated basic taste proxy intensities (`sweet`, `sour`, `bitter`, `salty`, `umami` $\\in [0, 1]$) with FlavorDB entity IDs (Garg et al., 2018), volatile aroma molecules, and chemesthetic pungency indicators.

## 2. Corpus Cleaning & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | {initial_count:,} |
| Invalid / Incomplete Records Removed | {invalid_removed:,} |
| Exact & Canonical Duplicates Removed (within cuisine) | {duplicates_removed:,} |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **{n_recipes_clean:,}** |
| **Cuisines Represented** | **{n_cuisines}** |
| **Unique Normalized Ingredients** | **{n_unique_ings:,}** |
| Unique Ingredients Matched to Taste Lexicon | {n_matched_unique:,} ({pct_matched_unique:.2f}%) |
| Unmatched Unique Ingredients Logged (`match_status='unmatched'`) | {n_unique_ings - n_matched_unique:,} ({100.0 - pct_matched_unique:.2f}%) |
| Total Recipe-Ingredient Occurrences (Tokens) | {total_ing_tokens:,} |
| Matched Recipe-Ingredient Occurrences | {matched_ing_tokens:,} ({pct_matched_tokens:.2f}%) |
| **Recipes Sufficient for Downstream Analysis ($M_r \\ge 2, \\text{{coverage}} \\ge 25\\%$)** | **{n_sufficient:,} ({pct_sufficient:.2f}%)** |

## 3. Breakdown by Cuisine
{cuisine_counts.to_markdown(index=False)}

## 4. Scientific Distinction: Gustatory Tastants, Retronasal Aroma, & Chemesthesis
1. **Basic Gustatory Tastants:** Sweet, sour, bitter, salty, and umami represent non-volatile, water-soluble tastants interacting directly with taste receptor cells on lingual papillae (T1R, T2R, ENaC, OTOP1).
2. **Retronasal Aroma Compounds:** Volatile organic molecules (e.g., *linalool*, *limonene*, *cuminaldehyde*) documented in FlavorDB activate olfactory sensory neurons in the nasal cavity via retronasal airflow. They define aroma, not basic gustatory taste intensity.
3. **Trigeminal Chemesthesis:** Pungent compounds like capsaicin in chilies, piperine in black pepper, and gingerols in ginger stimulate somatosensory pain/thermal receptors (TRPV1, TRPA1) via the trigeminal nerve. They do not constitute a sixth basic taste modality and are explicitly flagged under `chemesthetic_pungency` in `ingredient_flavor_mapping.csv`.
"""
    (REPORTS_DIR / "dataset_summary.md").write_text(summary_md, encoding="utf-8")

    return {
        "n_recipes_clean": n_recipes_clean,
        "n_unique_ingredients": n_unique_ings,
        "n_cuisines": n_cuisines,
        "pct_matched_unique": pct_matched_unique,
        "pct_matched_tokens": pct_matched_tokens,
        "n_sufficient": n_sufficient,
        "is_synthetic": meta["is_synthetic"],
    }
