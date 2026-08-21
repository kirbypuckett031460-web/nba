from __future__ import annotations

import hmac
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from src.nba_model.pipeline import generate_predictions, load_predictions, train_and_save_model
from src.nba_model.settings import AppSettings


st.set_page_config(page_title="NBA Model Admin", page_icon="🏀", layout="wide")
settings = AppSettings()


def _check_passphrase(entered: str, expected: str) -> bool:
    return hmac.compare_digest((entered or "").strip(), (expected or "").strip())


def _auth_guard() -> bool:
    if "admin_auth_ok" not in st.session_state:
        st.session_state.admin_auth_ok = False

    if st.session_state.admin_auth_ok:
        return True

    st.title("NBA Model Admin")
    st.caption("Passphrase-protected control panel")
    if not settings.admin_passphrase:
        st.error("ADMIN_PASSPHRASE is not configured. Set it in environment/secrets before using admin tools.")
        return False

    entered = st.text_input("Enter admin passphrase", type="password")
    if st.button("Unlock Admin", use_container_width=True):
        st.session_state.admin_auth_ok = _check_passphrase(entered, settings.admin_passphrase)
        if not st.session_state.admin_auth_ok:
            st.error("Incorrect passphrase.")
            return False
        st.rerun()
    return False


def _dispatch_workflow(owner: str, repo: str, workflow_file: str, token: str, ref: str) -> tuple[bool, str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"ref": ref}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code == 204:
        return True, "Workflow dispatch requested."
    return False, f"Dispatch failed ({response.status_code}): {response.text}"


if not _auth_guard():
    st.stop()

st.title("🏀 NBA Model Admin")
st.caption("Moneyline + Total Points model operations")

settings.ensure_paths()

left, right = st.columns([1, 1])
with left:
    st.subheader("Model Operations")
    force_retrain = st.checkbox("Force full retrain before generating picks", value=True)
    if st.button("Generate Latest Predictions", type="primary", use_container_width=True):
        with st.spinner("Running model + pulling live Odds API lines..."):
            payload = generate_predictions(settings=settings, force_retrain=force_retrain)
        st.success(f"Generated {len(payload.get('games', []))} game predictions.")

    if st.button("Train Model Only", use_container_width=True):
        with st.spinner("Training model from NBA stats history..."):
            bundle = train_and_save_model(settings=settings)
        st.success("Model retrained.")
        st.json(bundle.metrics)

with right:
    st.subheader("GitHub Workflow Control")
    st.caption("Scheduled workflow also runs automatically every day at noon ET.")
    branch_ref = st.text_input("Workflow branch ref", value=os.getenv("GITHUB_REF_NAME", "main"))
    if st.button("Run Daily Workflow Now", use_container_width=True):
        missing = [k for k, v in {
            "GITHUB_OWNER": settings.github_owner,
            "GITHUB_REPO": settings.github_repo,
            "GITHUB_TOKEN": settings.github_token,
            "GITHUB_WORKFLOW_FILE": settings.github_workflow_file,
        }.items() if not v]
        if missing:
            st.error(f"Missing required settings for dispatch: {', '.join(missing)}")
        else:
            ok, message = _dispatch_workflow(
                owner=settings.github_owner,
                repo=settings.github_repo,
                workflow_file=settings.github_workflow_file,
                token=settings.github_token,
                ref=branch_ref,
            )
            if ok:
                st.success(message)
            else:
                st.error(message)

st.divider()
payload = load_predictions(settings.predictions_path)
games = payload.get("games", [])
st.write(f"Last refresh: `{payload.get('generated_at')}` | Model built: `{payload.get('model_generated_at')}`")
if payload.get("model_metrics"):
    st.json(payload["model_metrics"])

if not games:
    st.info("No prediction data found yet. Run generation to populate.")
    st.stop()

df = pd.DataFrame(games)
df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
df["game_time_local"] = df["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %I:%M %p ET")

display_cols = [
    "game_time_local",
    "away_team",
    "home_team",
    "moneyline_pick",
    "best_moneyline_edge_pct",
    "total_pick",
    "total_line",
    "model_projected_total",
    "over_edge_pct",
    "under_edge_pct",
]
available = [c for c in display_cols if c in df.columns]
st.subheader("Current Model Edges")
st.dataframe(
    df[available].sort_values("best_moneyline_edge_pct", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Admin loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
