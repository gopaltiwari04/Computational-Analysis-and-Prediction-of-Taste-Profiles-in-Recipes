# Deliverable 2: Recipe Taste Profile Methodology

## 1. Mathematical Formulation
For a recipe $r$ containing matched ingredient set $M_r$ ($|M_r| \ge 2$), each ingredient $i \in M_r$ maps to a 5-dimensional gustatory taste proxy vector $\mathbf{t}(i) \in [0, 1]^5$ over the basis $\mathcal{D} = \{\text{sweet}, \text{sour}, \text{bitter}, \text{salty}, \text{umami}\}$.

1. **Unweighted Mean Intensity Vector:**
   $$\mathbf{u}_d(r) = \frac{1}{|M_r|} \sum_{i \in M_r} \mathbf{t}_d(i)$$
   *(Note: If reliable mass/volume quantities $w_i$ are available, the formulation supports quantity-weighted aggregation $\mathbf{u}_d(r) = \sum w_i \mathbf{t}_d(i) / \sum w_i$. In standard recipe corpora lacking gram-level weights, unweighted presence is the only defensible baseline.)*

2. **$L_1$-Normalized Taste Simplex Profile:**
   $$\mathbf{p}_d(r) = \frac{\mathbf{u}_d(r)}{\sum_{d' \in \mathcal{D}} \mathbf{u}_{d'}(r)}, \quad \text{such that } \sum_{d \in \mathcal{D}} \mathbf{p}_d(r) = 1$$

3. **Taste Distributional Evenness (Normalized Shannon Entropy):**
   $$H(r) = -\sum_{d \in \mathcal{D}: \mathbf{p}_d(r) > 0} \mathbf{p}_d(r) \ln \mathbf{p}_d(r) \quad \text{[nats]}$$
   $$J(r) = \frac{H(r)}{\ln(5)} \in [0, 1]$$
   - $J(r) = 1.0$ indicates equal shares across all five taste dimensions (maximum dispersion).
   - $J(r) = 0.0$ indicates single-modality dominance (minimum dispersion).

## 2. Worked Example from Corpus: Recipe ID `10259` (`greek`)
### Matched Ingredient Vectors
| ingredient_norm   |   sweet |   sour |   bitter |   salty |   umami |
|:------------------|--------:|-------:|---------:|--------:|--------:|
| olives            |    0.05 |   0.35 |     0.45 |    0.8  |    0.35 |
| garlic            |    0.15 |   0.05 |     0.1  |    0    |    0.35 |
| black pepper      |    0    |   0    |     0.35 |    0    |    0.05 |
| red onion         |    0.3  |   0.1  |     0.1  |    0    |    0.2  |
| chickpeas         |    0.2  |   0.02 |     0.1  |    0.05 |    0.25 |

### Aggregated Recipe Profile
- **Raw Mean Vector ($\mathbf{u}$):** sweet=0.1400, sour=0.1040, bitter=0.2200, salty=0.1700, umami=0.2400
- **Normalized Profile ($\mathbf{p}$):** sweet=0.1602, sour=0.1190, bitter=0.2517, salty=0.1945, umami=0.2746
- **Dominant Taste Modality:** `umami`
- **Shannon Entropy ($H$):** `1.5673` nats
- **Distributional Evenness ($J$):** `0.9738`

## 3. Critical Methodological & Sensory Disclaimers
1. **Ingredient Presence vs. Concentration:** In the absence of measured gram weights, an unweighted average attributes equal presence weight to high-volume staples (e.g., rice, milk) and low-volume spices (e.g., cayenne, turmeric). The resulting vector reflects the *taste orientation of the ingredient assembly*, not physical molecular concentration in the plated dish.
2. **Entropy vs. Human Culinary Balance:** Normalized entropy $J(r)$ quantifies the mathematical uniformity of the modeled taste shares across the 5 dimensions. It is not an empirical measure of gastronomic harmony or hedonic palatability; a culinary dish does not require equal intensities of all five basic tastes to be perceived as balanced by human diners.
3. **Chemical Transformation Omitted:** Culinary processing (caramelization, Maillard browning, acid hydrolysis, salting, fermentation) transforms tastants dynamically during cooking. These static representations serve as an ingredient-level taste proxy rather than wet-lab sensory evaluations.
