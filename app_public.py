from __future__ import annotations

import pandas as pd
import streamlit as st

from src.nba_model.pipeline import load_predictions
from src.nba_model.settings import AppSettings


st.set_page_config(page_title="NBA Picks Dashboard", page_icon="🏀", layout="wide")
st.title("🏀 NBA Moneyline + Totals Model")
st.caption("Live market lines from The Odds API with model-implied probabilities and edges.")

settings = AppSettings()
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

st.caption(f"Last updated: {payload.get('generated_at')}")
