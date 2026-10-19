# Deliverable 1: Dataset Construction Summary

## 1. Data Source & Provenance
- **Source Note:** Generated deterministic multi-cuisine sample corpus (--use-sample-data)
- **Taste & Flavor Reference:** `data/raw/ingredient_taste_lexicon.csv` combining documented gustatory modality scores (`sweet`, `sour`, `bitter`, `salty`, `umami`) with FlavorDB entity IDs and representative volatile aroma molecules (Garg et al., 2018).

## 2. Corpus & Matching Metrics
| Metric | Value |
| :--- | :--- |
| Initial Raw Recipe Records | 184 |
| Invalid/Incomplete Records Removed | 2 |
| Exact & Canonical Duplicate Recipes Removed | 42 |
| **Cleaned Unique Recipes (`recipes_clean.csv`)** | **140** |
| **Cuisines Represented** | **6** |
| **Unique Normalized Ingredients** | **76** |
| Unique Ingredients Matched to Taste Lexicon | 69 (90.79%) |
| Unmatched Unique Ingredients Explicitly Logged | 7 (9.21%) |
| Total Recipe-Ingredient Occurrences (Tokens) | 1172 |
| Matched Recipe-Ingredient Occurrences | 1130 (96.42%) |
| **Recipes Sufficient for Downstream Analysis** | **140 (100.00%)** |

## 3. Breakdown by Cuisine
| cuisine                 |   total_recipes |   sufficient_recipes |   mean_match_ratio |
|:------------------------|----------------:|---------------------:|-------------------:|
| American_Bakery_Comfort |              23 |                   23 |           0.988957 |
| Indian                  |              25 |                   25 |           0.96376  |
| Italian                 |              20 |                   20 |           0.948335 |
| Japanese                |              27 |                   27 |           0.942607 |
| Mexican                 |              23 |                   23 |           0.978087 |
| Thai                    |              22 |                   22 |           0.981314 |

## 4. Scientific Distinction: Flavor Molecules vs. Gustatory Taste
Not every volatile flavor molecule in FlavorDB activates human gustatory taste receptors. Volatile compounds (e.g., *linalool*, *limonene*, *cuminaldehyde*) primarily drive retronasal olfaction (aroma), whereas basic taste perception (`sweet`, `sour`, `bitter`, `salty`, `umami`) is driven by water-soluble non-volatile tastants (e.g., sucrose, citric acid, quinine/glucosinolates, sodium chloride, L-glutamate/5'-ribonucleotides). Our mapping table explicitly separates `primary_flavor_compounds` (olfactory) from the five numerical taste scores (`sweet` through `umami`) and records unmatched ingredients (`match_status = "unmatched"`) without fabricating taste values.
