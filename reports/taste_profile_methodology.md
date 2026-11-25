# Deliverable 2: Recipe Taste Profile Methodology

> **Corpus Notice:** Generated from **REAL EMPIRICAL RECIPE CORPUS** (`Kaggle / Yummly What's Cooking Dataset`).

## 1. Mathematical Formulation
For a recipe $r$ containing taste-annotated matched ingredient set $M_r$ ($|M_r| \ge 2$), each ingredient $i \in M_r$ maps to a 5-dimensional basic-taste proxy vector $\mathbf{t}(i) \in ([0, 1] \cup \{\text{NaN}\})^5$ over $\mathcal{D} = \{\text{sweet}, \text{sour}, \text{bitter}, \text{salty}, \text{umami}\}$.

1. **Unweighted Mean Intensity Vector (with Missing-Score Exclusion):**
   Let $M_{r,d} = \{i \in M_r : t_d(i) \text{ is known}\}$. Then:
   $$u_d(r) = \frac{1}{|M_{r,d}|} \sum_{i \in M_{r,d}} t_d(i), \quad \text{or } \text{NaN} \text{ if } |M_{r,d}| = 0.$$
   *(When reliable ingredient mass weights $w_i$ are present, `aggregate_recipe_taste_vector` computes $u_d(r) = \sum_{i \in M_{r,d}} w_i t_d(i) / \sum_{i \in M_{r,d}} w_i$. Because standard recipe corpora lack gram quantities, no heuristic weights are invented; unweighted presence is used.)*

2. **$L_1$-Normalized Taste Share Profile (4-Simplex):**
   Treating unobserved ($\text{NaN}$) dimensions as $0$ contribution to the total modeled taste sum $S(r) = \sum_{d : u_d(r) \text{ known}} u_d(r)$:
   $$p_d(r) = \frac{u_d(r)}{S(r)} \text{ if } S(r) > 0, \quad \text{else } 0, \quad \text{so } \sum_{d \in \mathcal{D}} p_d(r) = 1 \text{ (when } S(r) > 0\text{)}.$$

3. **Modeled Taste-Share Evenness (Normalized Shannon Entropy):**
   $$H(r) = -\sum_{d \in \mathcal{D}: p_d(r) > 0} p_d(r) \ln p_d(r) \quad \text{[nats]}$$
   $$J(r) = \frac{H(r)}{\ln(5)} \in [0, 1]$$
   - $J(r) = 1.0$: perfectly uniform distribution across all 5 modeled taste shares.
   - $J(r) = 0.0$: single-modality concentration (or zero-vector).

## 2. Corpus Summary Statistics
- **Sufficient Recipes Profiled:** 37,447 (Unspecified zero-vector profiles: 0)
- **Mean Shannon Entropy ($H$):** 1.3892 nats (SD = 0.2275; theoretical max = $\ln 5 \approx 1.6094$)
- **Mean Taste-Share Evenness ($J$):** 0.8631 (SD = 0.1414)

## 3. Worked Example from Corpus: Recipe ID `10259` (`greek`)
### Matched Ingredient Vectors
| ingredient_norm   |   sweet | sour          |   bitter | salty         |   umami |
|:------------------|--------:|:--------------|---------:|:--------------|--------:|
| olives            |    0.05 | 0.35          |     0.45 | 0.8           |    0.35 |
| tomato            |    0.35 | 0.45          |     0.05 | 0.05          |    0.7  |
| garlic            |    0.15 | 0.05          |     0.1  | 0.0           |    0.35 |
| red onion         |    0.3  | 0.1           |     0.1  | 0.0           |    0.2  |
| chickpeas         |    0.2  | unknown (NaN) |     0.1  | unknown (NaN) |    0.25 |
| feta cheese       |    0.05 | 0.45          |     0.05 | 0.85          |    0.45 |

### Aggregated Recipe Profile
- **Raw Mean Vector ($\mathbf{u}$):** sweet=0.1833, sour=0.2800, bitter=0.1417, salty=0.3400, umami=0.3833
- **Normalized Taste Shares ($\mathbf{p}$):** sweet=0.1380, sour=0.2108, bitter=0.1066, salty=0.2560, umami=0.2886
- **Dominant Taste Modality:** `umami`
- **Shannon Entropy ($H$):** `1.5477` nats
- **Taste-Share Evenness ($J$):** `0.9616`

## 4. Methodological & Sensory Limitations
1. **Presence vs. Quantity/Concentration:** Because public recipe lists lack mass weights, unweighted averaging assigns equal influence to a tablespoon of salt or spice and a cup of grain or broth. The output represents an *ingredient-assembly taste proxy*, not physical concentration in a cooked serving.
2. **Distributional Evenness vs. Human Perceived Balance:** Pielou's evenness $J(r)$ measures mathematical uniformity of the 5 modeled taste shares; it does **not** directly measure human gastronomic harmony or culinary palatability (a dessert does not require equal bitterness, saltiness, and sourness to be well-balanced).
3. **Cooking Transformations Omitted:** Caramelization, Maillard reactions, fermentation, dilution, and taste-taste suppression (e.g., fat or sugar masking bitterness) are not captured by static ingredient lookups.
4. **Proxy Instrument:** Outputs must be interpreted as computational proxies rather than empirical sensory panel scores.
