from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.nba_model.pipeline import load_predictions
from src.nba_model.settings import AppSettings


st.set_page_config(page_title="NBA Picks Dashboard", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
      /* Hide Streamlit chrome links/buttons on public embed page */
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="stStatusWidget"],
      #MainMenu {
        display: none !important;
      }
      [data-testid="stAppViewContainer"] header a[href*="github.com"],
      [data-testid="stAppViewContainer"] header a[href*="fork"] {
        display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("NBA Picks")
st.caption("Live market lines from The Odds API with model-implied probabilities and edges.")

settings = AppSettings()


def _prediction_file_path(settings: AppSettings) -> str:
    path = str(settings.predictions_path).replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path[1:]
    return path


def _load_predictions_from_github(settings: AppSettings) -> tuple[dict, str] | None:
    owner = (settings.github_owner or "").strip()
    repo = (settings.github_repo or "").strip()
    if not owner or not repo:
        return None

    branch = os.getenv("PREDICTIONS_BRANCH", "main")
    prediction_path = _prediction_file_path(settings)
    token = (settings.github_token or "").strip()

    # Private repos can use GitHub Contents API with bearer auth.
    if token:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{prediction_path}"
        headers = {
            "Accept": "application/vnd.github.raw",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=20)
        if response.status_code == 200:
            return json.loads(response.text), "github-api"

    # Public fallback (or private if raw endpoint is accessible in your setup).
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{prediction_path}"
    response = requests.get(raw_url, params={"_ts": int(time.time())}, timeout=20)
    if response.status_code == 200:
        return json.loads(response.text), "github-raw"

    return None


if st.button("Refresh picks", use_container_width=False):
    st.rerun()

payload_source = "local-file"
remote = _load_predictions_from_github(settings)
if remote is not None:
    payload, payload_source = remote
else:
    payload = load_predictions(settings.predictions_path)
games = payload.get("games", [])

if not games:
    st.warning("No predictions are currently available. Please check back after the next model refresh.")
    st.stop()

df = pd.DataFrame(games)
df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
df["game_time_et"] = df["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%a %b %d, %I:%M %p ET")
df["slate_date_et"] = df["commence_time"].dt.tz_convert("America/New_York").dt.date

for pct_col in [
    "model_home_win_prob",
    "model_away_win_prob",
    "model_over_prob",
    "model_under_prob",
]:
    if pct_col in df.columns:
        df[pct_col] = (pd.to_numeric(df[pct_col], errors="coerce") * 100).round(1)

edge_cols = ["home_edge_pct", "away_edge_pct", "over_edge_pct", "under_edge_pct", "best_moneyline_edge_pct"]
for col in edge_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

def _as_record_text(value: str | None, default: str = "0-0") -> str:
    if value and str(value).strip():
        return str(value).strip()
    return default


def _as_pct_text(value: float | int | str | None, default: str = "0.0%") -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        val = float(value)
        if val <= 1.0:
            val = val * 100.0
        return f"{val:.1f}%"
    except Exception:
        return default


def _prob_to_american(prob: float | int | None) -> str:
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "—"
    p = float(prob)
    if p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)
    return f"{odds:+d}"


performance = payload.get("performance", {})
moneyline_perf = performance.get("moneyline", {})
totals_perf = performance.get("totals", {})

slate_date = df["slate_date_et"].dropna().min()
if pd.notna(slate_date):
    slate_date_text = pd.Timestamp(slate_date).strftime("%a, %b %d, %Y")
else:
    slate_date_text = "N/A"

st.markdown(f"**Slate Date: {slate_date_text}**")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Moneyline Yesterday", _as_record_text(moneyline_perf.get("yesterday_record")))
    st.caption(_as_pct_text(moneyline_perf.get("yesterday_pct")))
with metric_cols[1]:
    st.metric("Moneyline YTD", _as_record_text(moneyline_perf.get("ytd_record")))
    st.caption(_as_pct_text(moneyline_perf.get("ytd_pct", 0.0)))
with metric_cols[2]:
    st.metric("Totals Yesterday", _as_record_text(totals_perf.get("yesterday_record")))
    st.caption(_as_pct_text(totals_perf.get("yesterday_pct")))
with metric_cols[3]:
    st.metric("Totals YTD", _as_record_text(totals_perf.get("ytd_record")))
    st.caption(_as_pct_text(totals_perf.get("ytd_pct", 0.0)))

tab_ml, tab_ou = st.tabs(["Moneyline Picks", "Over/Under Picks"])

ml = df.copy()
ml["pick_team"] = ml["moneyline_pick"].astype(str).str.replace(" ML", "", regex=False)
is_pass = ml["pick_team"].str.contains("Pass", case=False, na=False)
is_home_pick = ml["pick_team"].eq(ml["home_team"])
is_away_pick = ml["pick_team"].eq(ml["away_team"])
ml["mkt"] = np.select(
    [is_home_pick, is_away_pick],
    [ml["home_moneyline"], ml["away_moneyline"]],
    default=np.nan,
)
ml.loc[is_pass, "mkt"] = np.nan
ml["fair_prob"] = np.where(
    ml["pick_team"].eq(ml["home_team"]),
    ml["model_home_win_prob"],
    np.where(ml["pick_team"].eq(ml["away_team"]), ml["model_away_win_prob"], np.nan),
)
ml["fair"] = ml["fair_prob"].apply(_prob_to_american)
ml["edge_text"] = ml["best_moneyline_edge_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
ml["confidence"] = ml["fair_prob"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
ml_table = ml[
    [
        "game_time_et",
        "away_team",
        "home_team",
        "mkt",
        "fair",
        "pick_team",
        "edge_text",
        "confidence",
        "best_moneyline_edge_pct",
        "fair_prob",
    ]
].rename(
    columns={
        "game_time_et": "Game Time (ET)",
        "away_team": "Away",
        "home_team": "Home",
        "mkt": "Mkt",
        "fair": "Fair",
        "pick_team": "Pick",
        "edge_text": "Edge",
        "confidence": "Confidence",
    }
)
ml_table = ml_table.sort_values("best_moneyline_edge_pct", ascending=False, na_position="last")
ml_display = ml_table.drop(columns=["best_moneyline_edge_pct", "fair_prob"]).copy()

def _edge_color(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "background-color: #182235; color: #95a7c5;"
    value = float(v)
    if value >= 8:
        return "background-color: #0f8f6f; color: #eafff8;"
    if value >= 3:
        return "background-color: #1c6d59; color: #eafff8;"
    if value >= 0:
        return "background-color: #2e3b54; color: #dbe8ff;"
    return "background-color: #5b1f2f; color: #ffd9e3;"


def _confidence_color(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "background-color: #182235; color: #95a7c5;"
    p = float(v)
    if p >= 0.70:
        return "background-color: #145f6a; color: #e9ffff;"
    if p >= 0.60:
        return "background-color: #1c4f66; color: #e9f6ff;"
    return "background-color: #2d3e57; color: #d5def0;"


ml_styler = (
    ml_display.style.hide(axis="index")
    .apply(lambda s: [_edge_color(v) for v in ml_table["best_moneyline_edge_pct"]], subset=["Edge"])
    .apply(lambda s: [_confidence_color(v) for v in ml_table["fair_prob"]], subset=["Confidence"])
)

with tab_ml:
    st.dataframe(ml_styler, use_container_width=True)

ou = df.copy()
ou["pick"] = ou["total_pick"]
ou["edge_raw"] = np.where(
    ou["pick"].eq("Over"),
    ou["over_edge_pct"],
    np.where(ou["pick"].eq("Under"), ou["under_edge_pct"], np.nan),
)
ou["confidence_raw"] = np.where(
    ou["pick"].eq("Over"),
    ou["model_over_prob"],
    np.where(ou["pick"].eq("Under"), ou["model_under_prob"], np.nan),
)
ou["edge"] = ou["edge_raw"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
ou["confidence"] = ou["confidence_raw"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
ou_table = ou[
    [
        "game_time_et",
        "away_team",
        "home_team",
        "total_line",
        "model_projected_total",
        "pick",
        "edge",
        "confidence",
        "edge_raw",
        "confidence_raw",
    ]
].rename(
    columns={
        "game_time_et": "Game Time (ET)",
        "away_team": "Away",
        "home_team": "Home",
        "total_line": "Line",
        "model_projected_total": "Projection",
        "pick": "Pick",
        "edge": "Edge",
        "confidence": "Confidence",
    }
)
ou_table = ou_table.sort_values("edge_raw", ascending=False, na_position="last")
ou_display = ou_table.drop(columns=["edge_raw", "confidence_raw"]).copy()

ou_styler = (
    ou_display.style.hide(axis="index")
    .apply(lambda s: [_edge_color(v) for v in ou_table["edge_raw"]], subset=["Edge"])
    .apply(lambda s: [_confidence_color(v) for v in ou_table["confidence_raw"]], subset=["Confidence"])
)

with tab_ou:
    st.dataframe(ou_styler, use_container_width=True)

st.caption(f"Last updated: {payload.get('generated_at')} | Source: {payload_source}")
