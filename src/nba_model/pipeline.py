from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .model import TrainingBundle, predict_from_odds, train_models
from .odds import fetch_raw_odds, odds_to_dataframe
from .settings import AppSettings, season_strings


def train_and_save_model(settings: AppSettings) -> TrainingBundle:
    settings.ensure_paths()
    seasons = season_strings(settings.lookback_seasons)
    bundle = train_models(seasons=seasons)
    bundle.save(settings.model_path)
    return bundle


def generate_predictions(settings: AppSettings, force_retrain: bool = False) -> dict[str, Any]:
    settings.ensure_paths()
    if force_retrain or not settings.model_path.exists():
        try:
            bundle = train_and_save_model(settings=settings)
        except Exception:
            if settings.model_path.exists():
                bundle = TrainingBundle.load(settings.model_path)
            else:
                raise
    else:
        bundle = TrainingBundle.load(settings.model_path)

    if not settings.odds_api_key:
        raise ValueError("ODDS_API_KEY is required to fetch live NBA lines.")

    raw_odds = fetch_raw_odds(
        api_key=settings.odds_api_key,
        regions=settings.odds_regions,
        markets=settings.odds_markets,
        bookmakers=settings.odds_bookmakers,
    )
    odds_df = odds_to_dataframe(raw_odds)
    predictions_df = predict_from_odds(bundle=bundle, odds_df=odds_df, min_edge_pct=settings.min_edge_pct)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_generated_at": bundle.generated_at,
        "model_metrics": bundle.metrics,
        "residual_total_std": bundle.residual_total_std,
        "games": _serialize_predictions(predictions_df),
    }
    settings.predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"generated_at": None, "model_generated_at": None, "model_metrics": {}, "games": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_predictions(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    serializable = df.copy()
    for col in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[col]):
            serializable[col] = serializable[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serializable = serializable.where(pd.notna(serializable), None)
    return serializable.to_dict(orient="records")
