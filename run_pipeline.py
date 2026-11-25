"""
Master reproducible execution script for all 4 Computational Gastronomy deliverables.

Usage:
    # 1. Run full pipeline on real dataset (requires data/raw/train.json):
    python run_pipeline.py

    # 2. Run pipeline specifying custom raw data path:
    python run_pipeline.py --data-path path/to/recipes.json

    # 3. Explicitly run synthetic development smoke test (isolated optional --output-dir):
    python run_pipeline.py --use-sample-data

    # 4. Validate raw recipes, lexicon, and aliases without running full pipeline:
    python run_pipeline.py --validate-only

    # 5. Predict dominant taste and class probabilities for a recipe:
    python run_pipeline.py --predict "Roma tomatoes, garlic, extra-virgin olive oil, fresh basil"

    # 6. Retrieve top-K nearest culinary neighbors for a recipe ID:
    python run_pipeline.py --top-k-similar 10259
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_ALIASES_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_RAW_RECIPES_PATH,
    RunContext,
    default_context,
    setup_logging,
)
from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import (
    load_aliases,
    load_lexicon,
    load_raw_recipes,
    run_deliverable_1,
    validate_raw_recipes_df,
)
from src.taste_prediction import (
    predict_recipe_taste,
    retrieve_similar_recipes,
    run_deliverable_4,
)
from src.taste_profile import run_deliverable_2


def build_context_from_output_dir(output_dir: Optional[str]) -> RunContext:
    """Create either the default repository RunContext or an isolated output RunContext."""
    if not output_dir:
        return default_context()
    root = Path(output_dir)
    return RunContext(
        processed_dir=root / "data" / "processed",
        reports_dir=root / "reports",
        results_dir=root / "results",
        models_dir=root / "models",
        manifest_path=root / "data" / "dataset_manifest.json",
        lexicon_path=DEFAULT_LEXICON_PATH,
        aliases_path=DEFAULT_ALIASES_PATH,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Computational Gastronomy 4-Deliverable Pipeline & Inference Suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        default=False,
        help="Explicitly opt-in to the built-in deterministic multi-cuisine synthetic corpus (testing/smoke only).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_RAW_RECIPES_PATH),
        help=f"Path to raw recipe JSON dataset (default: {DEFAULT_RAW_RECIPES_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional isolated output directory root (prevents overwriting default artifacts during smoke tests).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Validate raw recipe schema, taste lexicon, and alias files without executing training/reports.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Run inference on comma- or semicolon-separated ingredients.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["full", "masked"],
        default="full",
        help="Classifier setting for --predict ('full' rule-recovery or 'masked' contextual co-occurrence).",
    )
    parser.add_argument(
        "--top-k-similar",
        type=int,
        default=None,
        help="Retrieve top-K nearest neighbors for a recipe ID in continuous 5D taste simplex space.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of neighbors to retrieve for --top-k-similar (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    ctx = build_context_from_output_dir(args.output_dir)

    # Mode 1: Inference API
    if args.predict is not None:
        raw_ings = [i.strip() for i in args.predict.replace(";", ",").split(",") if i.strip()]
        try:
            res = predict_recipe_taste(
                raw_ings,
                model_path=ctx.models_dir / "taste_model.joblib",
                setting=args.setting,
            )
            print("\n================ RECIPE TASTE PREDICTION ================")
            print(f"Model Corpus Type:          {'SYNTHETIC' if res['is_synthetic_model'] else 'REAL DATA'}")
            print(f"Evaluation Setting:         {res['model_setting']}")
            print(f"Status:                     {res['status']}")
            print(f"Input Raw Ingredients:      {raw_ings}")
            print(f"Normalized Ingredients:     {res['normalized_ingredients']}")
            print(f"Known in Model Vocab:       {res['known_ingredients']}")
            print(f"Out-of-Vocabulary (OOV):    {res['unknown_ingredients']}")
            print(f"Predicted Dominant Taste:   {str(res['predicted_dominant_taste']).upper()}")
            if res["status"] == "success":
                print("Class Probability Distribution:")
                for modality, prob in sorted(res["probabilities"].items()):
                    bar = "#" * int(prob * 30)
                    print(f"  {modality.capitalize():<8}: {prob:.4f} | {bar}")
            else:
                print("No confident prediction emitted (empty or 100% out-of-vocabulary input).")
            print("=========================================================\n")
            return 0
        except Exception as exc:
            print(f"ERROR: Inference failed: {exc}", file=sys.stderr)
            return 1

    # Mode 2: Nearest Neighbor Retrieval
    if args.top_k_similar is not None:
        csv_path = ctx.processed_dir / "recipe_taste_features.csv"
        if not csv_path.exists():
            print(f"ERROR: Features file not found at {csv_path}. Run the pipeline first.", file=sys.stderr)
            return 1
        try:
            df = pd.read_csv(csv_path)
            sim_df = retrieve_similar_recipes(df, query_recipe_id=args.top_k_similar, top_k=args.k)
            query_row = df[df["recipe_id"] == args.top_k_similar].iloc[0]
            print(f"\n======== TOP-{args.k} SIMILAR RECIPES FOR ID {args.top_k_similar} ========")
            print(f"Query Cuisine: {query_row['cuisine']} | Dominant Taste: {query_row['dominant_taste']}")
            print(sim_df.to_string(index=False))
            print("==============================================================\n")
            return 0
        except Exception as exc:
            print(f"ERROR: Retrieval failed: {exc}", file=sys.stderr)
            return 1

    raw_path = Path(args.data_path)

    # Mode 3: Validate-Only Check
    if args.validate_only:
        try:
            lex_df = load_lexicon(ctx.lexicon_path)
            aliases = load_aliases(ctx.aliases_path)
            raw_df, meta = load_raw_recipes(raw_path, use_sample_data=args.use_sample_data)
            valid_df, breakdown = validate_raw_recipes_df(raw_df, aliases=aliases)
            print("Validation PASSED:")
            print(f"  Dataset:   {meta['dataset_name']} (synthetic={meta['is_synthetic']})")
            print(f"  SHA-256:   {meta['sha256']}")
            print(f"  Records:   {len(raw_df):,} raw -> {len(valid_df):,} valid (breakdown={breakdown})")
            print(f"  Lexicon:   {len(lex_df)} verified entries")
            print(f"  Aliases:   {len(aliases)} canonical rules")
            return 0
        except Exception as exc:
            print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
            return 1

    # Mode 4: Full 4-Deliverable Pipeline Execution
    corpus_type = "SYNTHETIC SAMPLE DATA (--use-sample-data)" if args.use_sample_data else f"REAL DATA ({raw_path})"
    print("\n================ COMPUTATIONAL GASTRONOMY PIPELINE ================")
    print(f"Mode: {corpus_type}")
    if args.output_dir:
        print(f"Output Root: {args.output_dir}")
    print("===================================================================\n")

    try:
        print("[1/4] Running Deliverable 1: Dataset Construction & Normalization...")
        d1_stats = run_deliverable_1(raw_data_path=raw_path, use_sample_data=args.use_sample_data, ctx=ctx)
        source_label = "SYNTHETIC" if d1_stats["is_synthetic"] else "REAL"
        print(f"      [{source_label}] Clean recipes: {d1_stats['n_recipes_clean']:,} | Cuisines: {d1_stats['n_cuisines']}")
        print(f"      Unique ingredients: {d1_stats['n_unique_ingredients']:,} | Token match rate: {d1_stats['pct_matched_tokens']:.1f}%")
        print(f"      Sufficient recipes for analysis: {d1_stats['n_sufficient']:,}\n")

        print("[2/4] Running Deliverable 2: Computational Recipe Taste Profile...")
        d2_df = run_deliverable_2(ctx=ctx)
        print(f"      Generated 5D taste simplex profiles for {len(d2_df):,} sufficient recipes.")
        print(f"      Mean taste entropy: {d2_df['taste_entropy'].mean():.4f} nats | Mean evenness (J): {d2_df['taste_evenness'].mean():.4f}\n")

        print("[3/4] Running Deliverable 3: Cross-Cuisine Taste Profile Analysis...")
        d3_df = run_deliverable_3(ctx=ctx)
        print(f"      Analyzed {len(d3_df)} cuisines via Kruskal-Wallis + Holm-Bonferroni + CLR PERMANOVA.")
        print("      Generated cross-cuisine heatmap, boxplots, z-score lifts, and PCA biplot.\n")

        print("[4/4] Running Deliverable 4: Lightweight Recipe Taste Prediction Model...")
        d4_df = run_deliverable_4(ctx=ctx)
        print("      Leak-free model benchmarking complete. Summary:")
        print("      " + "-" * 85)
        print(d4_df[["evaluation_setting", "model", "test_accuracy", "test_macro_f1"]].to_string(index=False))
        print("      " + "-" * 85)

        print("\nPipeline execution complete! All verified artifacts written under:")
        print(f"  - Processed CSVs: {ctx.processed_dir}")
        print(f"  - Manifest:       {ctx.manifest_path}")
        print(f"  - Reports:        {ctx.reports_dir}")
        print(f"  - Plots/Tables:   {ctx.results_dir}")
        print(f"  - Model bundle:   {ctx.models_dir / 'taste_model.joblib'}\n")
        return 0

    except FileNotFoundError as fnf_err:
        print(f"\nDATA ERROR:\n{fnf_err}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nPIPELINE ERROR: Execution failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
