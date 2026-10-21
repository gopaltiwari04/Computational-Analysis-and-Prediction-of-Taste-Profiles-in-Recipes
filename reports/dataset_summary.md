# Deliverable 1: Dataset Construction & Provenance Summary

## 1. Data Source & Provenance Manifest
- **Dataset Name:** Kaggle / Yummly What's Cooking Dataset
- **Source:** https://www.kaggle.com/c/whats-cooking/data
- **Source Path:** `C:\Users\tewar\computational gastronomy\data\raw\train.json`
- **License / Terms:** Kaggle Competition / Educational Research Use (Yummly Inc.)
- **Corpus Type:** REAL (Empirical Recipe Corpus)
- **Taste & Flavor Reference:** `data/raw/ingredient_taste_lexicon.csv` (92 core entries) linking expert-curated basic taste proxy intensities (`sweet`, `sour`, `bitter`, `salty`, `umami` $\in [0, 1]$) with FlavorDB entity IDs (Garg et al., 2018), volatile aroma molecules, and chemesthetic pungency indicators.

## 2. Corpus Cleaning & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | 39,774 |
| Invalid / Incomplete Records Removed | 0 |
| Exact & Canonical Duplicates Removed (within cuisine) | 789 |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **38,985** |
| **Cuisines Represented** | **20** |
| **Unique Normalized Ingredients** | **6,226** |
| Unique Ingredients Matched to Taste Lexicon | 91 (1.46%) |
| Unmatched Unique Ingredients Logged (`match_status='unmatched'`) | 6,135 (98.54%) |
| Total Recipe-Ingredient Occurrences (Tokens) | 416,047 |
| Matched Recipe-Ingredient Occurrences | 254,350 (61.13%) |
| **Recipes Sufficient for Downstream Analysis ($M_r \ge 2, \text{coverage} \ge 25\%$)** | **37,033 (94.99%)** |

## 3. Breakdown by Cuisine
| cuisine      |   total_recipes |   sufficient_recipes |   mean_match_ratio |
|:-------------|----------------:|---------------------:|-------------------:|
| brazilian    |             430 |                  352 |           0.557774 |
| british      |             786 |                  716 |           0.565446 |
| cajun_creole |            1513 |                 1451 |           0.610863 |
| chinese      |            2631 |                 2571 |           0.6282   |
| filipino     |             745 |                  720 |           0.687705 |
| french       |            2591 |                 2422 |           0.581596 |
| greek        |            1156 |                 1122 |           0.65396  |
| indian       |            2973 |                 2893 |           0.617492 |
| irish        |             657 |                  613 |           0.575335 |
| italian      |            7719 |                 7292 |           0.589953 |
| jamaican     |             506 |                  487 |           0.632786 |
| japanese     |            1380 |                 1280 |           0.557761 |
| korean       |             813 |                  796 |           0.67307  |
| mexican      |            6313 |                 5996 |           0.600182 |
| moroccan     |             805 |                  783 |           0.650463 |
| russian      |             484 |                  471 |           0.623196 |
| southern_us  |            4198 |                 3897 |           0.572785 |
| spanish      |             969 |                  918 |           0.605152 |
| thai         |            1513 |                 1478 |           0.588112 |
| vietnamese   |             803 |                  775 |           0.604919 |

## 4. Scientific Distinction: Gustatory Tastants, Retronasal Aroma, & Chemesthesis
1. **Basic Gustatory Tastants:** Sweet, sour, bitter, salty, and umami represent non-volatile, water-soluble tastants interacting directly with taste receptor cells on lingual papillae (T1R, T2R, ENaC, OTOP1).
2. **Retronasal Aroma Compounds:** Volatile organic molecules (e.g., *linalool*, *limonene*, *cuminaldehyde*) documented in FlavorDB activate olfactory sensory neurons in the nasal cavity via retronasal airflow. They define aroma, not basic gustatory taste intensity.
3. **Trigeminal Chemesthesis:** Pungent compounds like capsaicin in chilies, piperine in black pepper, and gingerols in ginger stimulate somatosensory pain/thermal receptors (TRPV1, TRPA1) via the trigeminal nerve. They do not constitute a sixth basic taste modality and are explicitly flagged under `chemesthetic_pungency` in `ingredient_flavor_mapping.csv`.
