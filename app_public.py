from __future__ import annotations

import json
import os
import time

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
st.title("🏀 NBA Moneyline + Totals Model")
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

top_recos = df[
    [
        "game_time_et",
        "away_team",
        "home_team",
        "moneyline_pick",
        "best_moneyline_edge_pct",
        "total_pick",
        "total_line",
        "model_projected_total",
    ]
].copy()

st.subheader("Best Current Recommendations")
st.dataframe(top_recos.sort_values("best_moneyline_edge_pct", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Full Model Table")
full_cols = [
    "game_time_et",
    "away_team",
    "home_team",
    "home_moneyline",
    "away_moneyline",
    "total_line",
    "model_home_win_prob",
    "model_away_win_prob",
    "model_projected_total",
    "model_over_prob",
    "model_under_prob",
    "home_edge_pct",
    "away_edge_pct",
    "over_edge_pct",
    "under_edge_pct",
]
st.dataframe(df[full_cols], use_container_width=True, hide_index=True)

st.caption(f"Last updated: {payload.get('generated_at')} | Source: {payload_source}")
