from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .model import TrainingBundle, predict_from_odds, train_models
from .odds import fetch_raw_odds, odds_to_dataframe
from .performance import compute_performance, empty_performance_payload, update_pick_history
from .settings import AppSettings, season_strings


def train_and_save_model(settings: AppSettings) -> TrainingBundle:
    settings.ensure_paths()
    seasons = season_strings(settings.lookback_seasons)
    bundle = train_models(seasons=seasons)
    bundle.save(settings.model_path)
    return bundle


def _coerce_target_date(target_date: date | str | None) -> date | None:
    if target_date is None:
        return None
    if isinstance(target_date, date):
        return target_date
    parsed = pd.to_datetime(str(target_date), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid target date: {target_date}")
    return parsed.date()


def _filter_predictions_by_date(
    predictions_df: pd.DataFrame,
    target_date: date | str | None,
    target_timezone: str,
) -> tuple[pd.DataFrame, date | None]:
    coerced_date = _coerce_target_date(target_date)
    if coerced_date is None or predictions_df.empty:
        return predictions_df, coerced_date
    if "commence_time" not in predictions_df.columns:
        return predictions_df, coerced_date

    filtered = predictions_df.copy()
    commence_ts = pd.to_datetime(filtered["commence_time"], utc=True, errors="coerce")
    try:
        local_dates = commence_ts.dt.tz_convert(target_timezone).dt.date
    except Exception as exc:
        raise ValueError(f"Invalid target timezone: {target_timezone}") from exc

    filtered = filtered.loc[local_dates == coerced_date].copy()
    return filtered.reset_index(drop=True), coerced_date


def _slate_lock_time_utc(predictions_df: pd.DataFrame) -> pd.Timestamp | None:
    if predictions_df.empty or "commence_time" not in predictions_df.columns:
        return None
    starts = pd.to_datetime(predictions_df["commence_time"], utc=True, errors="coerce").dropna()
    if starts.empty:
        return None
    return starts.min()


def _existing_locked_payload(
    settings: AppSettings,
    filter_date: date | None,
    now_utc: pd.Timestamp,
) -> dict[str, Any] | None:
    if filter_date is None or not settings.predictions_path.exists():
        return None
    existing = load_predictions(settings.predictions_path)
    existing_filter_date = str(existing.get("filter_date") or "").strip()
    if existing_filter_date != filter_date.isoformat():
        return None
    existing_games = existing.get("games", [])
    if not existing_games:
        return None
    existing_df = pd.DataFrame(existing_games)
    lock_time = _slate_lock_time_utc(existing_df)
    if lock_time is None:
        return None
    if now_utc >= lock_time:
        return existing
    return None


def generate_predictions(
    settings: AppSettings,
    force_retrain: bool = False,
    target_date: date | str | None = None,
    target_timezone: str = "America/New_York",
) -> dict[str, Any]:
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
    predictions_df, filter_date = _filter_predictions_by_date(
        predictions_df=predictions_df,
        target_date=target_date,
        target_timezone=target_timezone,
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    locked_payload = _existing_locked_payload(
        settings=settings,
        filter_date=filter_date,
        now_utc=now_utc,
    )
    if locked_payload is not None:
        # Slate is locked and already published for this date; return it unchanged.
        locked_payload["slate_locked"] = True
        return locked_payload

    lock_time = _slate_lock_time_utc(predictions_df)
    if filter_date is not None and lock_time is not None and now_utc >= lock_time:
        lock_et = lock_time.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")
        raise ValueError(
            f"Slate for {filter_date.isoformat()} is locked (first tip: {lock_et}). "
            "No new picks can be published after lock."
        )
    generated_at = datetime.now(timezone.utc).isoformat()

    history_df = update_pick_history(
        path=settings.picks_history_path,
        predictions_df=predictions_df,
        generated_at=generated_at,
    )
    performance = compute_performance(history_df=history_df, timezone=target_timezone)

    payload = {
        "generated_at": generated_at,
        "model_generated_at": bundle.generated_at,
        "model_metrics": bundle.metrics,
        "residual_total_std": bundle.residual_total_std,
        "filter_date": filter_date.isoformat() if filter_date else None,
        "filter_timezone": target_timezone if filter_date else None,
        "performance": performance,
        "games": _serialize_predictions(predictions_df),
    }
    settings.predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "generated_at": None,
            "model_generated_at": None,
            "model_metrics": {},
            "filter_date": None,
            "filter_timezone": None,
            "performance": empty_performance_payload(),
            "games": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "performance" not in payload:
        payload["performance"] = empty_performance_payload()
    return payload


def _serialize_predictions(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    serializable = df.copy()
    for col in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[col]):
            serializable[col] = serializable[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serializable = serializable.where(pd.notna(serializable), None)
    return serializable.to_dict(orient="records")
