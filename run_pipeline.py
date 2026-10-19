"""
Master reproducible execution script for all 4 deliverables.
Usage:
    python run_pipeline.py --use-sample-data
"""
import argparse
import sys

from src.cuisine_analysis import run_deliverable_3
from src.data_preprocessing import run_deliverable_1
from src.taste_prediction import run_deliverable_4
from src.taste_profile import run_deliverable_2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Computational Gastronomy 4-Deliverable Pipeline."
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        default=True,
        help="Use deterministic built-in multi-cuisine sample corpus if data/raw/train.json is absent.",
    )
    parser.add_argument(
        "--require-external-data",
        action="store_true",
        default=False,
        help="Fail with FileNotFoundError if data/raw/train.json is not present.",
    )
    args = parser.parse_args()

    use_sample = not args.require_external_data
    try:
        print("[1/4] Running Deliverable 1: Dataset Construction & Normalization...")
        d1_stats = run_deliverable_1(use_sample_if_missing=use_sample)
        print(f"      Cleaned recipes: {d1_stats['n_recipes_clean']} | Unique ingredients: {d1_stats['n_unique_ingredients']} | Token match rate: {d1_stats['pct_matched_tokens']:.1f}%")

        print("[2/4] Running Deliverable 2: Computational Recipe Taste Profile...")
        d2_df = run_deliverable_2()
        print(f"      Generated taste profiles for {len(d2_df)} sufficient recipes.")

        print("[3/4] Running Deliverable 3: Cross-Cuisine Taste Profile Analysis...")
        d3_df = run_deliverable_3()
        print(f"      Analyzed {len(d3_df)} cuisines with Kruskal-Wallis + Holm-Bonferroni correction.")

        print("[4/4] Running Deliverable 4: Lightweight Recipe Taste Prediction Model...")
        d4_df = run_deliverable_4()
        print("      Model comparison complete. Summary:")
        print(d4_df.to_string(index=False))
        print("\nAll 4 deliverables completed successfully. Artifacts written to data/processed/, results/, models/, and reports/.")
        return 0
    except Exception as exc:
        print(f"ERROR: Pipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())