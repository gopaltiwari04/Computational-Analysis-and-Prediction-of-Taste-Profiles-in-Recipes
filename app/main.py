"""
Deployable FastAPI web application for the Computational Gastronomy Taste Profile Framework.
Designed with zero heavy scientific-library imports at runtime so it deploys cleanly on
Vercel Serverless Functions as well as Docker, Render, and local Uvicorn.
"""
from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.engine import (
    compute_lexicon_profile,
    derive_anchors,
    is_anchor,
    load_aliases_csv,
    load_csv_records,
    load_lexicon_csv,
    normalize_ingredient,
    predict_from_weights,
    retrieve_similar_from_rows,
)
from src.config import RunContext, TASTE_DIMENSIONS, default_context

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AnalyzeRequest(BaseModel):
    ingredients: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of raw recipe ingredient strings.",
    )


def create_app(ctx: Optional[RunContext] = None) -> FastAPI:
    """Create FastAPI instance bound to the given RunContext (or env-configured paths)."""
    if ctx is None:
        env_root = os.environ.get("CG_DATA_ROOT")
        if env_root:
            root = Path(env_root)
            context = RunContext(
                processed_dir=root / "data" / "processed",
                reports_dir=root / "reports",
                results_dir=root / "results",
                models_dir=root / "models",
                manifest_path=root / "data" / "dataset_manifest.json",
                lexicon_path=default_context().lexicon_path,
                aliases_path=default_context().aliases_path,
            )
        else:
            context = default_context()
    else:
        context = ctx

    app = FastAPI(
        title="Computational Gastronomy: Recipe Taste Profile & Prediction Explorer",
        version="1.1.0",
        description=(
            "Lightweight web service for recipe ingredient normalization, 5D taste-simplex "
            "proxy modeling, cross-cuisine statistical exploration, and leak-free taste classification."
        ),
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if context.results_dir.exists():
        app.mount("/results", StaticFiles(directory=str(context.results_dir)), name="results")

    @lru_cache(maxsize=1)
    def _load_resources() -> Dict[str, Any]:
        lex_rows = load_lexicon_csv(context.lexicon_path)
        lex_lookup = {str(r["ingredient_norm"]): r for r in lex_rows}
        aliases = load_aliases_csv(context.aliases_path)
        anchors = derive_anchors(lex_rows)

        manifest: Dict[str, Any] = {}
        if context.manifest_path.exists():
            manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))

        feat_num_cols = set(TASTE_DIMENSIONS) | {
            "recipe_id",
            "n_ingredients_matched",
            "n_known_taste_dims",
            "taste_entropy",
            "taste_evenness",
            "taste_balance",
        }
        features_rows = load_csv_records(
            context.processed_dir / "recipe_taste_features.csv",
            numeric_cols=feat_num_cols,
        )

        cuisine_num_cols = {"n_recipes", "mean_taste_evenness", "std_taste_evenness", "max_distinctiveness_z"}
        for d in TASTE_DIMENSIONS:
            cuisine_num_cols.update({f"{d}_mean", f"{d}_std", f"{d}_ci95_halfwidth", f"{d}_median", f"{d}_iqr", f"{d}_z_distinctiveness"})
        cuisine_rows = load_csv_records(
            context.results_dir / "cuisine_comparison.csv",
            numeric_cols=cuisine_num_cols,
        )

        model_comp_rows: List[Dict[str, Any]] = []
        mc_path = context.results_dir / "model_comparison.csv"
        if mc_path.exists():
            with open(mc_path, "r", encoding="utf-8", newline="") as f:
                model_comp_rows = [dict(r) for r in csv.DictReader(f)]

        weights_path = context.models_dir / "taste_model_weights.json"
        weights_bundle: Optional[Dict[str, Any]] = None
        model_meta: Dict[str, Any] = {"available": False}

        if weights_path.exists():
            weights_bundle = json.loads(weights_path.read_text(encoding="utf-8"))
        else:
            joblib_path = context.models_dir / "taste_model.joblib"
            if joblib_path.exists():
                from src.taste_prediction import export_model_weights_json
                weights_bundle = export_model_weights_json(joblib_path, weights_path)

        if weights_bundle is not None:
            model_meta = {
                "available": True,
                "classes": weights_bundle["full"]["classes"],
                "vocabulary_size": len(weights_bundle["full"]["weights"]),
                "masked_vocabulary_size": len(weights_bundle.get("masked", {}).get("weights", {})),
                "anchor_count": len(weights_bundle.get("anchor_tokens", anchors)),
                "is_synthetic": bool(weights_bundle.get("is_synthetic", False)),
                "trained_at": weights_bundle.get("trained_at", "unknown"),
            }

        return {
            "lex_rows": lex_rows,
            "lex_lookup": lex_lookup,
            "aliases": aliases,
            "anchors": anchors,
            "manifest": manifest,
            "features_rows": features_rows,
            "cuisine_rows": cuisine_rows,
            "model_comp_rows": model_comp_rows,
            "weights_bundle": weights_bundle,
            "model_meta": model_meta,
        }

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def serve_index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(str(index_path))

    @app.get("/favicon.ico", include_in_schema=False)
    def serve_favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        res = _load_resources()
        manifest = res["manifest"]
        return {
            "status": "ok",
            "version": "1.1.0",
            "corpus": {
                "dataset_name": manifest.get("dataset_name", "Not yet generated"),
                "is_synthetic": bool(manifest.get("is_synthetic", False)),
                "clean_recipe_count": manifest.get("clean_recipe_count", 0),
                "sufficient_recipes_count": manifest.get("sufficient_recipes_count", 0),
                "cuisines_represented": manifest.get("cuisines_represented", 0),
                "pct_tokens_matched": manifest.get("pct_tokens_matched", 0.0),
                "lexicon_entries_count": len(res["lex_rows"]),
                "aliases_count": len(res["aliases"]),
            },
            "model": res["model_meta"],
            "model_comparison": res["model_comp_rows"],
        }

    @app.post("/api/analyze")
    def analyze_recipe(payload: AnalyzeRequest) -> Dict[str, Any]:
        res = _load_resources()
        lex_lookup = res["lex_lookup"]
        aliases = res["aliases"]
        anchors = res["anchors"]
        weights_bundle = res["weights_bundle"]

        cleaned_inputs = [s.strip() for s in payload.ingredients if isinstance(s, str) and s.strip()]
        if not cleaned_inputs:
            raise HTTPException(status_code=400, detail="Please provide at least one non-empty ingredient.")

        ingredient_details = []
        seen_norm = set()
        taste_rows: List[List[Optional[float]]] = []

        for raw in cleaned_inputs:
            norm = normalize_ingredient(raw, aliases)
            is_dup = bool(norm and norm in seen_norm)
            if norm:
                seen_norm.add(norm)

            info = lex_lookup.get(norm) if norm else None
            has_taste = bool(info is not None and any(info[d] is not None for d in TASTE_DIMENSIONS))
            is_masked = bool(norm and is_anchor(norm, anchors, aliases))

            if info is not None and not is_dup and has_taste:
                taste_rows.append([info[d] for d in TASTE_DIMENSIONS])

            if info is not None:
                detail = {
                    "raw": raw,
                    "normalized": norm,
                    "duplicate_after_norm": is_dup,
                    "status": "matched" if has_taste else "taste_unknown",
                    "is_masked_anchor": is_masked,
                    "category": info.get("category", "unclassified"),
                    "taste_scores": {d: info[d] for d in TASTE_DIMENSIONS},
                    "chemesthetic_pungency": info.get("chemesthetic_pungency", "none"),
                    "confidence_level": info.get("confidence_level", "unknown"),
                    "annotation_status": info.get("annotation_status", "curated_proxy"),
                    "flavordb_entity_id": str(info.get("flavordb_entity_id", "NONE")),
                    "flavordb_entity_name": str(info.get("flavordb_entity_name", "NONE")),
                    "primary_aroma_compounds": str(info.get("primary_aroma_compounds", "NONE")),
                    "known_tastant_or_pungent_compounds": str(
                        info.get("known_tastant_or_pungent_compounds", "NONE")
                    ),
                }
            else:
                detail = {
                    "raw": raw,
                    "normalized": norm,
                    "duplicate_after_norm": is_dup,
                    "status": "unmatched" if norm else "empty_after_norm",
                    "is_masked_anchor": is_masked,
                    "category": "unmatched",
                    "taste_scores": {d: None for d in TASTE_DIMENSIONS},
                    "chemesthetic_pungency": "none",
                    "confidence_level": "unknown",
                    "annotation_status": "unmatched",
                    "flavordb_entity_id": "NONE",
                    "flavordb_entity_name": "NONE",
                    "primary_aroma_compounds": "NONE",
                    "known_tastant_or_pungent_compounds": "NONE",
                }
            ingredient_details.append(detail)

        lexicon_proxy = compute_lexicon_profile(taste_rows)

        ml_predictions: Dict[str, Any] = {"available": False}
        if weights_bundle is not None:
            pred_full = predict_from_weights(weights_bundle, cleaned_inputs, aliases, setting="full")
            non_anchor_inputs = [
                d["normalized"]
                for d in ingredient_details
                if d["normalized"] and not d["is_masked_anchor"]
            ]
            pred_masked = predict_from_weights(weights_bundle, non_anchor_inputs, aliases, setting="masked")
            ml_predictions = {
                "available": True,
                "setting_1_full": pred_full,
                "setting_2_masked": pred_masked,
            }

        return {
            "ingredients": ingredient_details,
            "lexicon_proxy": lexicon_proxy,
            "ml_predictions": ml_predictions,
            "disclaimer": (
                "Outputs represent ingredient-presence computational proxies, not gram-weighted "
                "physical molarity or empirical human sensory panel ratings."
            ),
        }

    @app.get("/api/cuisines")
    def get_cuisines() -> Dict[str, Any]:
        res = _load_resources()
        cuisine_rows = res["cuisine_rows"]
        if not cuisine_rows:
            raise HTTPException(status_code=404, detail="Cuisine comparison data not found. Run pipeline first.")
        return {"cuisines": cuisine_rows, "manifest": res["manifest"]}

    @app.get("/api/recipes/sample")
    def get_sample_recipes() -> Dict[str, Any]:
        res = _load_resources()
        rows = [r for r in res["features_rows"] if r.get("dominant_taste") != "unspecified"]
        if not rows:
            return {"samples": []}

        rows_sorted = sorted(
            rows,
            key=lambda r: (
                -(r.get("n_known_taste_dims") or 0),
                -(r.get("n_ingredients_matched") or 0),
                int(r["recipe_id"]),
            ),
        )
        seen_cuisines = set()
        samples = []
        for r in rows_sorted:
            c = str(r["cuisine"])
            if c in seen_cuisines:
                continue
            seen_cuisines.add(c)
            samples.append(
                {
                    "recipe_id": int(r["recipe_id"]),
                    "cuisine": c,
                    "dominant_taste": str(r["dominant_taste"]),
                    "ingredients": str(r.get("ingredients_normalized", "")).split(";"),
                }
            )
            if len(samples) >= 10:
                break
        return {"samples": samples}

    @app.get("/api/similar/{recipe_id}")
    def get_similar(recipe_id: int, top_k: int = Query(5, ge=1, le=20)) -> Dict[str, Any]:
        res = _load_resources()
        rows = res["features_rows"]
        if not rows:
            raise HTTPException(status_code=404, detail="Recipe features dataset not available.")
        try:
            q_row, neighbors = retrieve_similar_from_rows(rows, query_recipe_id=recipe_id, top_k=top_k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Recipe ID {recipe_id} not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        evenness = q_row.get("taste_evenness")
        if evenness is None:
            evenness = q_row.get("taste_balance")

        return {
            "query": {
                "recipe_id": int(q_row["recipe_id"]),
                "cuisine": str(q_row["cuisine"]),
                "dominant_taste": str(q_row["dominant_taste"]),
                "taste_evenness": round(float(evenness), 4) if evenness is not None else None,
                "simplex": {d: q_row[d] for d in TASTE_DIMENSIONS},
                "ingredients": str(q_row.get("ingredients_normalized", "")).split(";"),
            },
            "neighbors": neighbors,
        }

    @app.get("/api/lexicon")
    def get_lexicon(
        q: str = Query("", description="Optional ingredient substring filter."),
        status: str = Query("", description="Optional annotation_status filter."),
    ) -> Dict[str, Any]:
        res = _load_resources()
        rows = res["lex_rows"]
        ql = q.strip().lower()
        st = status.strip()

        filtered = []
        for r in rows:
            if st and r.get("annotation_status") != st:
                continue
            if ql:
                haystack = " ".join(
                    [
                        str(r.get("ingredient_norm", "")),
                        str(r.get("category", "")),
                        str(r.get("known_tastant_or_pungent_compounds", "")),
                    ]
                ).lower()
                if ql not in haystack:
                    continue
            filtered.append(r)

        return {"count": len(filtered), "entries": filtered}

    return app


app = create_app()
