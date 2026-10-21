# Comprehensive Quality & Scientific Audit Report

**Project:** Computational Analysis and Prediction of Taste Profiles in Recipes  
**Audit Date:** 2026-10-22  
**Auditor:** Senior ML Engineer, Computational Gastronomy Researcher, Statistician & Scientific Software Auditor  
**Repository State Audited:** Commit `89ee852` (Initial master commit)

---

## 1. Executive Summary

This audit evaluates the codebase against the highest standards of scientific validity, mathematical rigor, machine learning integrity, automated testing, and research presentation. 

The original codebase established a clean, modular 4-deliverable structure. However, critical vulnerabilities were discovered that compromise scientific defensibility if left unaddressed:
1. **Silent Fallback to Synthetic Data:** The execution pipeline defaulted to generating a synthetic 140-recipe corpus whenever the external raw dataset was missing, allowing unverified synthetic claims to masquerade as real-world findings.
2. **Data Leakage in Machine Learning:** Feature extraction (`CountVectorizer`) was fitted on the entire dataset prior to train/test partitioning, leaking test set vocabulary and frequencies into the training pipeline.
3. **Incomplete Model Persistence:** Serialized model artifacts omitted the feature transformation pipeline, preventing standalone inference on arbitrary recipe inputs.
4. **Taste Lexicon Ambiguity:** Numerical taste scores were labeled as `"culinary_sensory+FlavorDB"`, falsely conflating molecular flavor annotations (volatile aroma compounds) with unvalidated quantitative gustatory taste measurements.
5. **Missing Dependency Specifications:** Critical packages (`tabulate`) required by DataFrame export routines were omitted from `requirements.txt`.
6. **Compositional Data Oversights:** Simplex-constrained taste shares ($\sum p_d = 1$) were subjected to independent univariate hypothesis tests without acknowledging negative covariance and compositional closure.
7. **Minimal Test Coverage:** Only three unit tests existed, leaving core mathematical invariants, statistical corrections, data engineering edge cases, and inference pipelines unverified.
8. **Missing Root Documentation:** No `README.md` existed at the repository root.

---

## 2. Detailed Audit Findings & Remediation Matrix

| Finding ID | Severity | Component | Issue Description | Scientific / Engineering Impact | Fix Implemented |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AUD-01** | **CRITICAL** | `run_pipeline.py`, `src/data_preprocessing.py` | Silent fallback to synthetic sample data when `data/raw/train.json` is missing. Default CLI argument `--use-sample-data` was `True`. | Risk of claiming real-world culinary conclusions based entirely on a 140-recipe synthetic toy corpus. Violates non-negotiable scientific integrity rules. | Refactored CLI to require explicit `--use-sample-data`. If raw data is missing and the flag is omitted, raise `FileNotFoundError` with actionable instructions. Added dataset provenance manifest. |
| **AUD-02** | **CRITICAL** | `src/taste_prediction.py` | `CountVectorizer.fit_transform()` was executed on the combined dataset before `train_test_split`. | Data leakage: vocabulary and document frequencies from the test split inform the feature representation, inflating test performance. | Split raw recipe data first into train and test sets. Fit vectorizer strictly on training data; transform test data using the fitted vocabulary. |
| **AUD-03** | **HIGH** | `src/taste_prediction.py`, `models/` | Model serialization saved only `best_clf` and a list of feature names; the fitted `CountVectorizer` object was lost. | Standalone reloading and inference cannot process raw ingredient strings without re-engineering preprocessing code. | Wrapped vectorizer and classifier in an end-to-end `sklearn.pipeline.Pipeline`. Added standalone inference CLI and Python API with full input validation. |
| **AUD-04** | **HIGH** | `data/raw/ingredient_taste_lexicon.csv` | Conflated FlavorDB volatile aroma molecules with gustatory taste intensity; attributed hand-curated taste scores to FlavorDB; included capsaicin as bitter taste without noting trigeminal chemesthesis. | False empirical claims: FlavorDB documents volatile aroma compounds, not basic taste intensities. Capsaicin stimulates TRPV1 receptors (heat/pain), not a basic taste bud. | Clarified that taste vectors represent an expert-curated proxy. Added trigeminal pungency indicators, confidence scores, and literature citations. Marked unknown values as NaN. |
| **AUD-05** | **HIGH** | `requirements.txt` | Missing `tabulate` dependency required by `.to_markdown()`. Missing test coverage tooling. | Clean installation fails at runtime when generating Markdown reports. | Added `tabulate>=0.9.0` and `pytest-cov>=4.1.0` to `requirements.txt`. |
| **AUD-06** | **MEDIUM** | `src/taste_profile.py` | Normalized Shannon entropy was presented directly as "taste balance" without clarifying that it measures distributional evenness. | Equating mathematical entropy to culinary harmony is subjective; high entropy does not imply optimal flavor. | Clarified metric as normalized entropy / distributional evenness ($J \in [0, 1]$). Added clear disclaimers distinguishing sensory perception from compositional proxies. |
| **AUD-07** | **MEDIUM** | `src/cuisine_analysis.py` | Univariate Kruskal–Wallis tests conducted on compositional data ($L_1$ simplex) without addressing compositional closure ($\sum p_d = 1$). | Proportions in a simplex possess negative covariance; independent univariate tests ignore compositional geometry. | Documented compositional closure limitation. Added centered log-ratio (clr) exploratory analysis and effect-size reporting ($\epsilon^2$). |
| **AUD-08** | **MEDIUM** | `src/cuisine_analysis.py` | PCA projection performed on unscaled cuisine means (`heatmap_matrix.to_numpy()`) without feature standardization. | Variance discrepancies among taste dimensions distort PCA axes; small sample ($K=6$ or $K=20$ centroids) lacks recipe-level distribution context. | Applied `StandardScaler` prior to PCA decomposition. Added dual visualization options for cuisine centroids and recipe-level distributions. |
| **AUD-09** | **HIGH** | `tests/` | Only 3 unit tests in `test_preprocessing.py`. Zero coverage for ML evaluation, statistical corrections, lexicon validation, or pipeline execution. | Fragile code; regressions in math, normalization, or leakage prevention go undetected. | Developed comprehensive test suite (6 dedicated test modules, >30 test cases) verifying all deliverables, invariants, and edge cases. |
| **AUD-10** | **MEDIUM** | Root directory | Repository root lacked a `README.md` file. | External researchers, instructors, and recruiters have no initial orientation or execution guidance. | Authored research-grade `README.md` complete with architecture diagrams, mathematical formulations, commands, and verified results. |

---

## 3. Discrepancy Analysis Between Generated Reports & Code

1. **Corpus Size Discrepancy:** The initial `reports/dataset_summary.md` reported 140 cleaned recipes across 6 cuisines, which was generated exclusively from the synthetic sample generator. However, the proposal and report discussed the Kaggle/Yummly dataset (39,774 recipes across 20 cuisines). The reporting did not distinguish whether the results were from synthetic smoke tests or real empirical data.
2. **Circularity in Taste Prediction:** Deliverable 4 correctly noted that predicting `dominant_taste` from `Full_Ingredients` is circular (rule recovery). However, the masking experiment used an informal anchor list that omitted common compound tastants (e.g., parmesan cheese, tomato paste). The masking methodology needed systematic categorization into pure tastant anchors and multi-taste ingredients.

---

## 4. Verification and Remediation Plan

The following phases implement complete remediation:
- **Phase 2:** Integrate real-world Kaggle/Yummly dataset (`train.json`) with dataset manifest, license documentation, and strict prevention of silent synthetic fallback.
- **Phase 3:** Audit and upgrade the taste lexicon with confidence annotations, chemesthesis distinctions, and transparent proxy labeling.
- **Phase 4:** Harden recipe taste profile math, Shannon entropy/evenness bounds, and quantity weighting hooks.
- **Phase 5:** Validate Holm–Bonferroni correction, Kruskal–Wallis effect sizes, and compositional data interpretations.
- **Phase 6:** Eliminate data leakage in ML, implement leak-free pipelines, serializable model bundles, and standalone inference.
- **Phase 7:** Implement comprehensive test suite covering all invariants and measure code coverage.
- **Phase 8:** Ensure end-to-end reproducibility with explicit CLI flags, logging, and manifest tracking.
- **Phase 9–11:** Update all scientific reports, author research-grade `README.md`, and generate `docs/validation_report.md`.
