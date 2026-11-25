"""Standalone validation script for raw recipe corpora, taste lexicon, and alias tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_ALIASES_PATH, DEFAULT_LEXICON_PATH, DEFAULT_RAW_RECIPES_PATH
from src.data_preprocessing import (
    load_aliases,
    load_lexicon,
    load_raw_recipes,
    validate_raw_recipes_df,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Computational Gastronomy data inputs.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_RAW_RECIPES_PATH)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--aliases-path", type=Path, default=DEFAULT_ALIASES_PATH)
    parser.add_argument("--use-sample-data", action="store_true", default=False)
    args = parser.parse_args()

    try:
        lex_df = load_lexicon(args.lexicon_path)
        aliases = load_aliases(args.aliases_path)
        raw_df, meta = load_raw_recipes(args.data_path, use_sample_data=args.use_sample_data)
        valid_df, breakdown = validate_raw_recipes_df(raw_df, aliases=aliases)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("VALIDATION PASSED")
    print(f"  Dataset:        {meta['dataset_name']}")
    print(f"  Synthetic:      {meta['is_synthetic']}")
    print(f"  SHA-256:        {meta['sha256']}")
    print(f"  Raw records:    {len(raw_df):,}")
    print(f"  Valid records:  {len(valid_df):,} (dropped breakdown: {breakdown})")
    print(f"  Lexicon rows:   {len(lex_df)}")
    print(f"  Alias rules:    {len(aliases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
