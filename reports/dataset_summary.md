# Deliverable 1: Dataset Construction & Provenance Summary

> **Corpus Classification:** **REAL EMPIRICAL CORPUS**

## 1. Data Source & Provenance Manifest
- **Dataset Name:** Kaggle / Yummly What's Cooking Dataset
- **Source URL:** https://www.kaggle.com/c/whats-cooking/data
- **Source File:** `data/raw/train.json`
- **SHA-256 Checksum:** `f18e7e568d1c606774bfb134ece51ef0a143cd5069e6f9241bb6036b45b7d2b0`
- **Access / File Date:** `2026-01-16`
- **License / Terms:** Unknown / Kaggle Playground Competition Rules (Yummly Inc.; educational/non-commercial research only, raw data not redistributed)
- **Taste Lexicon:** `data/raw/ingredient_taste_lexicon.csv` (122 entries; `reference_standard`, `curated_proxy`, and `taste_unknown` explicitly labeled; FlavorDB entity IDs verified against Garg et al., 2018)
- **Alias Dictionary:** `data/raw/ingredient_aliases.csv` (380 deterministic surface-to-canonical rules)

## 2. Corpus Cleaning, Deduplication & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | 39,774 |
| Invalid / Incomplete Records Removed | 0 |
| Duplicate Recipe IDs Removed | 0 |
| Within-Cuisine Canonical Duplicates Removed | 795 |
| Cross-Cuisine Identical Ingredient Sets Preserved | 55 |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **38,979** |
| **Cuisines Represented** | **20** |
| **Unique Normalized Ingredients** | **6,195** |
| Lexicon Matched with Taste Proxy (`match_status='matched'`) | 120 (1.94%) |
| Cataloged as Taste-Unknown (`match_status='taste_unknown'`) | 2 |
| Unmatched Unique Ingredients Logged (`match_status='unmatched'`) | 6,073 |
| Total Recipe-Ingredient Occurrences (Tokens) | 416,607 |
| Taste-Known Matched Occurrences | 266,841 (64.05%) |
| **Sufficient Recipes for Analysis ($M_r \ge 2$, coverage $\ge 25\%$)** | **37,447 (96.07%)** |

## 3. Breakdown by Cuisine
| cuisine      |   total_recipes |   sufficient_recipes |   mean_match_ratio |
|:-------------|----------------:|---------------------:|-------------------:|
| brazilian    |             434 |                  361 |             0.5754 |
| british      |             786 |                  735 |             0.6129 |
| cajun_creole |            1512 |                 1467 |             0.6528 |
| chinese      |            2630 |                 2579 |             0.6223 |
| filipino     |             745 |                  714 |             0.6387 |
| french       |            2590 |                 2454 |             0.63   |
| greek        |            1155 |                 1136 |             0.7214 |
| indian       |            2973 |                 2899 |             0.6181 |
| irish        |             658 |                  630 |             0.6443 |
| italian      |            7718 |                 7417 |             0.6366 |
| jamaican     |             506 |                  489 |             0.6714 |
| japanese     |            1379 |                 1313 |             0.621  |
| korean       |             812 |                  798 |             0.6773 |
| mexican      |            6316 |                 6035 |             0.6031 |
| moroccan     |             805 |                  790 |             0.7116 |
| russian      |             483 |                  474 |             0.6642 |
| southern_us  |            4195 |                 3976 |             0.6275 |
| spanish      |             970 |                  928 |             0.6498 |
| thai         |            1511 |                 1477 |             0.6001 |
| vietnamese   |             801 |                  775 |             0.6164 |

## 4. Top Unmatched Unique Ingredients (Logged for Transparency)
| ingredient_norm   |   recipe_frequency | example_raw_surface   |
|:------------------|-------------------:|:----------------------|
| pepper            |               4397 | pepper                |
| chili powder      |               2007 | chili powder          |
| salsa             |                956 | salsa                 |
| corn tortillas    |                947 | corn tortillas        |
| garam masala      |                917 | garam masala          |
| flour tortillas   |                871 | flour tortillas       |
| tomato sauce      |                862 | tomato sauce          |
| curry powder      |                693 | curry powder          |
| hot sauce         |                661 | hot sauce             |
| red chili peppers |                627 | red chili peppers     |
| cheese            |                623 | cheese                |
| lemon zest        |                578 | lemon zest            |
| orange            |                507 | orange                |
| lime wedges       |                476 | lime wedges           |
| red pepper        |                457 | red pepper            |

## 5. Scientific Distinction: Gustatory Tastants, Retronasal Aroma, & Chemesthesis
1. **Basic Gustatory Tastants:** Sweet, sour, bitter, salty, and umami represent non-volatile, water-soluble tastants acting on lingual taste receptor cells (T1R, T2R, ENaC, OTOP1).
2. **Retronasal Aroma Compounds:** Volatile organic molecules verified in FlavorDB (e.g., *linalool*, *citral*, *2,3,5,6-tetramethylpyrazine*) activate olfactory neurons via retronasal airflow and do **not** validate numerical basic-taste intensities.
3. **Trigeminal Chemesthesis:** Capsaicin, piperine, gingerol, allicin, and allyl isothiocyanate stimulate somatosensory TRP channels (TRPV1/TRPA1) via the trigeminal nerve. Chili heat is not a basic taste modality and is tracked separately under `chemesthetic_pungency`.
4. **Unknown Taste Values (`NaN`):** Refined lipids (`cooking oil`, `peanut oil`) and unestablished minor dimensions on low-confidence items are recorded as `NaN` rather than fabricated numbers.
