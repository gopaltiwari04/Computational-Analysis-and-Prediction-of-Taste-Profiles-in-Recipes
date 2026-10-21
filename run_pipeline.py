"""
Master reproducible execution script for all 4 Computational Gastronomy deliverables.

Usage:
    # 1. Run full pipeline on real dataset (requires data/raw/train.json):
    python run_pipeline.py

    # 2. Run pipeline specifying custom raw data path:
    python run_pipeline.py --data-path path/to/recipes.json

    # 3. Explicitly run synthetic development smoke test:
    python run_pipeline.py --use-sample-data

    # 4. Predict dominant taste and class probabilities for a recipe:
    python run_pipeline.py --predict "Roma tomatoes, garlic, extra-virgin olive oil, fresh basil"

    # 5. Retrieve top-3 nearest culinary neighbors for a recipe ID:
    python run_pipeline.py --top-k-similar 10259
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DEFAULT_RAW_RECIPES_PATH
from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import run_deliverable_1
from src.taste_prediction import predict_recipe_taste, retrieve_similar_recipes, run_deliverable_4
from src.taste_profile import run_deliverable_2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Computational Gastronomy 4-Deliverable Pipeline & Inference Suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        default=False,
        help="Explicitly opt-in to use the built-in deterministic multi-cuisine synthetic corpus (for testing only).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_RAW_RECIPES_PATH),
        help=f"Path to raw recipe JSON dataset (default: {DEFAULT_RAW_RECIPES_PATH}).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Run inference on a comma- or semicolon-separated ingredient string (e.g. 'tomato, garlic, olive oil').",
    )
    parser.add_argument(
        "--top-k-similar",
        type=int,
        default=None,
        help="Retrieve top-K nearest neighbors for a recipe ID in the continuous taste simplex space.",
    )

    args = parser.parse_args()

    # Mode 1: Inference API
    if args.predict:
        raw_ings = [i.strip() for i in args.predict.replace(";", ",").split(",") if i.strip()]
        try:
            res = predict_recipe_taste(raw_ings)
            print("\n================ RECIPE TASTE PREDICTION ================")
            print(f"Input Raw Ingredients:      {raw_ings}")
            print(f"Normalized Ingredients:     {res['normalized_ingredients']}")
            print(f"Predicted Dominant Taste:   {res['predicted_dominant_taste'].upper()}")
            print("Class Probability Distribution:")
            for modality, prob in res["probabilities"].items():
                bar = "#" * int(prob * 30)
                print(f"  {modality.capitalize():<8}: {prob:.4f} | {bar}")
            print("=========================================================\n")
            return 0
        except Exception as exc:
            print(f"ERROR: Inference failed: {exc}", file=sys.stderr)
            return 1

    # Mode 2: Nearest Neighbor Retrieval
    if args.top_k_similar is not None:
        csv_path = DATA_PROCESSED_DIR / "recipe_taste_features.csv"
        if not csv_path.exists():
            print(f"ERROR: Features file not found at {csv_path}. Run the pipeline first.", file=sys.stderr)
            return 1
        try:
            df = pd.read_csv(csv_path)
            sim_df = retrieve_similar_recipes(df, query_recipe_id=args.top_k_similar, top_k=3)
            query_row = df[df["recipe_id"] == args.top_k_similar].iloc[0]
            print(f"\n======== TOP-3 SIMILAR RECIPES FOR ID {args.top_k_similar} ========")
            print(f"Query Cuisine: {query_row['cuisine']} | Dominant Taste: {query_row['dominant_taste']}")
            print(sim_df.to_string(index=False))
            print("=======================================================\n")
            return 0
        except Exception as exc:
            print(f"ERROR: Retrieval failed: {exc}", file=sys.stderr)
            return 1

    # Mode 3: Full Pipeline Execution
    raw_path = Path(args.data_path)
    corpus_type = "SYNTHETIC SAMPLE DATA (--use-sample-data)" if args.use_sample_data else f"REAL DATA ({raw_path})"
    print(f"\n================ COMPUTATIONAL GASTRONOMY PIPELINE ================")
    print(f"Mode: {corpus_type}")
    print("===================================================================\n")

    try:
        print("[1/4] Running Deliverable 1: Dataset Construction & Normalization...")
        d1_stats = run_deliverable_1(raw_data_path=raw_path, use_sample_data=args.use_sample_data)
        source_label = "SYNTHETIC" if d1_stats["is_synthetic"] else "REAL"
        print(f"      [{source_label}] Clean recipes: {d1_stats['n_recipes_clean']:,} | Cuisines: {d1_stats['n_cuisines']}")
        print(f"      Unique ingredients: {d1_stats['n_unique_ingredients']:,} | Token match rate: {d1_stats['pct_matched_tokens']:.1f}%")
        print(f"      Sufficient recipes for analysis: {d1_stats['n_sufficient']:,}\n")

        print("[2/4] Running Deliverable 2: Computational Recipe Taste Profile...")
        d2_df = run_deliverable_2()
        print(f"      Generated 5D taste simplex profiles for {len(d2_df):,} sufficient recipes.")
        print(f"      Mean taste entropy: {d2_df['taste_entropy'].mean():.4f} nats | Mean evenness (J): {d2_df['taste_balance'].mean():.4f}\n")

        print("[3/4] Running Deliverable 3: Cross-Cuisine Taste Profile Analysis...")
        d3_df = run_deliverable_3()
        print(f"      Analyzed {len(d3_df)} cuisines via Kruskal-Wallis + Holm-Bonferroni correction.")
        print(f"      Generated cross-cuisine heatmap, boxplots, z-score lifts, and standardized PCA projection.\n")

        print("[4/4] Running Deliverable 4: Lightweight Recipe Taste Prediction Model...")
        d4_df = run_deliverable_4()
        print("      Leak-free model benchmarking complete. Summary:")
        print("      " + "-" * 75)
        print(d4_df[["evaluation_setting", "model", "test_accuracy", "test_macro_f1"]].to_string(index=False))
        print("      " + "-" * 75)

        print("\nPipeline execution complete! All artifacts verified and written:")
        print("  - Data tables:      data/processed/ (recipes_clean.csv, recipe_ingredients.csv, recipe_taste_features.csv)")
        print("  - Manifest:         data/dataset_manifest.json")
        print("  - Reports:          reports/ (dataset_summary.md, taste_profile_methodology.md, cross_cuisine_analysis.md, prediction_evaluation.md)")
        print("  - Visualizations:   results/ (heatmaps, boxplots, distinctiveness, PCA, confusion matrix, feature importances)")
        print("  - Model bundle:     models/taste_model.joblib (end-to-end serializable sklearn pipeline)\n")
        return 0

    except FileNotFoundError as fnf_err:
        print(f"\nDATA ERROR:\n{fnf_err}\n", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nPIPELINE ERROR: Execution failed: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
