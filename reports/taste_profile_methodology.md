# Deliverable 2: Recipe Taste Profile Methodology

## 1. Mathematical Formulation
For a recipe r with matched ingredient set M_r (size >= 2), each ingredient i in M_r maps to a 5-dimensional gustatory vector t(i) in [0, 1]^5 over D = {sweet, sour, bitter, salty, umami}.

1. **Unweighted Mean Intensity:** u_d(r) = (1 / |M_r|) * sum_{i in M_r} t_d(i)
2. **L1-Normalized Taste Profile:** p_d(r) = u_d(r) / sum_{d' in D} u_{d'}(r)
3. **Taste Diversity & Balance (Normalized Shannon Entropy):** H(r) = -sum_{d in D} p_d(r) ln p_d(r), and J(r) = H(r) / ln(5) in [0, 1]

## 2. Worked Example from Corpus: Recipe ID `1000` (Italian)
### Matched Ingredient Vectors
| ingredient_norm   |   sweet |   sour |   bitter |   salty |   umami |
|:------------------|--------:|-------:|---------:|--------:|--------:|
| tomato            |    0.35 |   0.45 |     0.05 |    0.05 |    0.7  |
| garlic            |    0.15 |   0.05 |     0.1  |    0    |    0.35 |
| olive oil         |    0.05 |   0.02 |     0.35 |    0    |    0    |
| basil             |    0.2  |   0    |     0.25 |    0    |    0.05 |
| parmesan cheese   |    0.05 |   0.15 |     0.1  |    0.75 |    0.95 |
| salt              |    0    |   0    |     0    |    1    |    0    |
| wheat flour       |    0.15 |   0    |     0.02 |    0    |    0.05 |
| onion             |    0.35 |   0.05 |     0.05 |    0    |    0.3  |

### Aggregated Recipe Profile
- **Raw Mean Vector (u):** sweet=0.1625, sour=0.0900, bitter=0.1150, salty=0.2250, umami=0.3000
- **Normalized Profile (p):** sweet=0.1821, sour=0.1008, bitter=0.1289, salty=0.2521, umami=0.3361
- **Dominant Taste:** `umami`
- **Taste Entropy (H):** `1.5194` nats | **Taste Balance (J):** `0.9440`

## 3. Methodological Scope & Sensory Disclaimer
This representation is an **ingredient-composition proxy** derived from unweighted presence of matched ingredients. It is **not** equivalent to gas chromatography-mass spectrometry (GC-MS), electronic tongue laboratory measurements, or trained human sensory panel evaluation.
