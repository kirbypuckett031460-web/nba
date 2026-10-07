from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .stats import fetch_schedule_results_for_season, normalize_team_name, team_lookup


def _zero_stats() -> dict[str, Any]:
    return {
        "yesterday_record": "0-0",
        "yesterday_pct": 0.0,
        "ytd_record": "0-0",
        "ytd_pct": 0.0,
        "yesterday_wins": 0,
        "yesterday_losses": 0,
        "yesterday_pushes": 0,
        "ytd_wins": 0,
        "ytd_losses": 0,
        "ytd_pushes": 0,
    }


def empty_performance_payload() -> dict[str, Any]:
    return {"moneyline": _zero_stats(), "totals": _zero_stats()}


def _season_for_date(d: date) -> str:
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _season_start(d: date) -> date:
    start = d.year if d.month >= 10 else d.year - 1
    return date(start, 10, 1)


def _format_record(wins: int, losses: int, pushes: int) -> str:
    if pushes > 0:
        return f"{wins}-{losses}-{pushes}"
    return f"{wins}-{losses}"


def _pct(wins: int, losses: int) -> float:
    denom = wins + losses
    if denom <= 0:
        return 0.0
    return round((wins / denom) * 100.0, 1)


def _prediction_key(row: pd.Series) -> str:
    game_id = row.get("game_id")
    if game_id is not None and str(game_id).strip():
        return str(game_id).strip()
    return f"{row.get('home_team','')}-{row.get('away_team','')}-{row.get('commence_time','')}"


def _canonicalizer() -> dict[str, str]:
    lookup = team_lookup()
    canon: dict[str, str] = {}
    for key, team in lookup.items():
        canon[normalize_team_name(key)] = team["full_name"]
    return canon


def _canonical_team(name: str, canon_lookup: dict[str, str]) -> str:
    norm = normalize_team_name(name)
    return canon_lookup.get(norm, str(name))


def load_pick_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()
    if not isinstance(raw, list):
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()
    return pd.DataFrame(raw)


def update_pick_history(path: Path, predictions_df: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    history_df = load_pick_history(path)
    existing: dict[str, dict[str, Any]] = {}
    if not history_df.empty:
        for _, row in history_df.iterrows():
            key = str(row.get("prediction_key", "")).strip() or _prediction_key(row)
            existing[key] = row.to_dict()

    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            record = {
                "prediction_key": _prediction_key(row),
                "game_id": row.get("game_id"),
                "commence_time": row.get("commence_time"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "moneyline_pick": row.get("moneyline_pick"),
                "total_pick": row.get("total_pick"),
                "total_line": row.get("total_line"),
                "generated_at": generated_at,
            }
            key = record["prediction_key"]
            prev = existing.get(key)
            if prev is None or str(prev.get("generated_at", "")) <= generated_at:
                existing[key] = record

    merged_df = pd.DataFrame(list(existing.values()))
    if merged_df.empty:
        return merged_df

    merged_df["commence_time"] = pd.to_datetime(merged_df["commence_time"], utc=True, errors="coerce")
    merged_df = merged_df.sort_values("commence_time").reset_index(drop=True)
    serializable = merged_df.copy()
    serializable["commence_time"] = serializable["commence_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable.to_dict(orient="records"), indent=2), encoding="utf-8")
    return merged_df


def _results_for_season(season: str, canon_lookup: dict[str, str]) -> pd.DataFrame:
    try:
        results = fetch_schedule_results_for_season(season)
    except Exception:
        return pd.DataFrame()
    if results.empty:
        return results

    out = results.copy()
    out["game_date_et"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out["home_canon"] = out["home_team"].map(lambda x: _canonical_team(str(x), canon_lookup))
    out["away_canon"] = out["away_team"].map(lambda x: _canonical_team(str(x), canon_lookup))
    out["total_points"] = pd.to_numeric(out["home_points"], errors="coerce") + pd.to_numeric(out["away_points"], errors="coerce")
    out["winner_canon"] = np.where(
        pd.to_numeric(out["home_points"], errors="coerce") > pd.to_numeric(out["away_points"], errors="coerce"),
        out["home_canon"],
        out["away_canon"],
    )
    return out[
        ["game_date_et", "home_canon", "away_canon", "total_points", "winner_canon", "home_points", "away_points"]
    ].dropna(subset=["game_date_et"])


def compute_performance(
    history_df: pd.DataFrame,
    timezone: str = "America/New_York",
    as_of_date: date | None = None,
) -> dict[str, Any]:
    if as_of_date is None:
        as_of_date = datetime.now(ZoneInfo(timezone)).date()

    yesterday = as_of_date - timedelta(days=1)
    season_start = _season_start(as_of_date)
    season = _season_for_date(as_of_date)

    if history_df.empty:
        return empty_performance_payload()

    canon_lookup = _canonicalizer()
    hist = history_df.copy()
    hist["commence_time"] = pd.to_datetime(hist["commence_time"], utc=True, errors="coerce")
    hist = hist.dropna(subset=["commence_time"]).copy()
    hist["game_date_et"] = hist["commence_time"].dt.tz_convert(timezone).dt.date
    hist["home_canon"] = hist["home_team"].map(lambda x: _canonical_team(str(x), canon_lookup))
    hist["away_canon"] = hist["away_team"].map(lambda x: _canonical_team(str(x), canon_lookup))

    results = _results_for_season(season=season, canon_lookup=canon_lookup)
    if results.empty:
        return empty_performance_payload()

    merged = hist.merge(
        results,
        on=["game_date_et", "home_canon", "away_canon"],
        how="inner",
    )
    if merged.empty:
        return empty_performance_payload()

    merged = merged[(merged["game_date_et"] >= season_start) & (merged["game_date_et"] <= yesterday)].copy()
    if merged.empty:
        return empty_performance_payload()

    # Moneyline
    ml = merged.copy()
    ml["moneyline_pick"] = ml["moneyline_pick"].astype(str)
    ml = ml[~ml["moneyline_pick"].str.contains("Pass", case=False, na=False)].copy()
    ml["ml_pick_team"] = ml["moneyline_pick"].str.replace(" ML", "", regex=False)
    ml["ml_pick_canon"] = ml["ml_pick_team"].map(lambda x: _canonical_team(str(x), canon_lookup))
    ml["ml_win"] = ml["ml_pick_canon"] == ml["winner_canon"]

    ml_ytd = ml
    ml_yday = ml[ml["game_date_et"] == yesterday]

    ml_ytd_w = int(ml_ytd["ml_win"].sum()) if not ml_ytd.empty else 0
    ml_ytd_l = int((~ml_ytd["ml_win"]).sum()) if not ml_ytd.empty else 0
    ml_yday_w = int(ml_yday["ml_win"].sum()) if not ml_yday.empty else 0
    ml_yday_l = int((~ml_yday["ml_win"]).sum()) if not ml_yday.empty else 0

    # Totals
    tot = merged.copy()
    tot["total_pick"] = tot["total_pick"].astype(str)
    tot["total_line"] = pd.to_numeric(tot["total_line"], errors="coerce")
    tot = tot[tot["total_pick"].isin(["Over", "Under"]) & pd.notna(tot["total_line"])].copy()
    tot["tot_win"] = np.where(
        (tot["total_pick"] == "Over") & (tot["total_points"] > tot["total_line"]),
        True,
        np.where(
            (tot["total_pick"] == "Under") & (tot["total_points"] < tot["total_line"]),
            True,
            False,
        ),
    )
    tot["tot_push"] = tot["total_points"] == tot["total_line"]
    tot["tot_loss"] = (~tot["tot_win"]) & (~tot["tot_push"])

    tot_ytd = tot
    tot_yday = tot[tot["game_date_et"] == yesterday]

    tot_ytd_w = int(tot_ytd["tot_win"].sum()) if not tot_ytd.empty else 0
    tot_ytd_l = int(tot_ytd["tot_loss"].sum()) if not tot_ytd.empty else 0
    tot_ytd_p = int(tot_ytd["tot_push"].sum()) if not tot_ytd.empty else 0
    tot_yday_w = int(tot_yday["tot_win"].sum()) if not tot_yday.empty else 0
    tot_yday_l = int(tot_yday["tot_loss"].sum()) if not tot_yday.empty else 0
    tot_yday_p = int(tot_yday["tot_push"].sum()) if not tot_yday.empty else 0

    return {
        "moneyline": {
            "yesterday_record": _format_record(ml_yday_w, ml_yday_l, 0),
            "yesterday_pct": _pct(ml_yday_w, ml_yday_l),
            "ytd_record": _format_record(ml_ytd_w, ml_ytd_l, 0),
            "ytd_pct": _pct(ml_ytd_w, ml_ytd_l),
            "yesterday_wins": ml_yday_w,
            "yesterday_losses": ml_yday_l,
            "yesterday_pushes": 0,
            "ytd_wins": ml_ytd_w,
            "ytd_losses": ml_ytd_l,
            "ytd_pushes": 0,
        },
        "totals": {
            "yesterday_record": _format_record(tot_yday_w, tot_yday_l, tot_yday_p),
            "yesterday_pct": _pct(tot_yday_w, tot_yday_l),
            "ytd_record": _format_record(tot_ytd_w, tot_ytd_l, tot_ytd_p),
            "ytd_pct": _pct(tot_ytd_w, tot_ytd_l),
            "yesterday_wins": tot_yday_w,
            "yesterday_losses": tot_yday_l,
            "yesterday_pushes": tot_yday_p,
            "ytd_wins": tot_ytd_w,
            "ytd_losses": tot_ytd_l,
            "ytd_pushes": tot_ytd_p,
        },
    }
