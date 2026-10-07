# NBA Moneyline + Totals Modeling Suite

Two Streamlit applications share a Python modeling backend:

- **Admin app** (`app_admin.py`): passphrase-protected control panel to retrain models, refresh predictions from The Odds API, and manually trigger your GitHub workflow.
  - Includes a date scope selector so you can generate picks for a specific **ET date** or all upcoming dates.
  - Manual workflow dispatch now forwards the selected date to CI so published public picks match the admin-selected slate.
- **Public app** (`app_public.py`): read-only predictions dashboard for website visitors.
  - Includes a manual **Refresh picks** button.
  - Pulls latest prediction JSON from GitHub (`main` branch) when `GITHUB_OWNER`/`GITHUB_REPO` are configured, with local file fallback.

## What the model uses

The model combines multiple NBA signal families to improve predictive quality:

- Team **advanced season profile** (off/def rating, net rating, pace, TS%, eFG%, rebounding and turnover factors) derived from Basketball-Reference season tables.
- Team **four factors and misc context** (opponent factor stats, transition/paint/2nd chance scoring tendencies).
- Team **rolling form** from recent games (offense, defense, margin, possessions, rest, back-to-back) from Basketball-Reference game schedules/results.
- Team **Elo rating trajectory** learned from game outcomes across multiple seasons.
- Live market inputs from **The Odds API**:
  - Moneyline (`h2h`)
  - Total points line (`totals`)

Models:

- **Moneyline model**: gradient-boosted classifier predicts `P(home win)`.
- **Total model**: gradient-boosted regressor predicts expected game total.
- **Over/Under probabilities** are derived from regressor output + residual variance estimate.

## Project structure

```text
.
├── app_admin.py
├── app_public.py
├── requirements.txt
├── scripts/
│   └── daily_refresh.py
├── src/
│   └── nba_model/
│       ├── __init__.py
│       ├── model.py
│       ├── odds.py
│       ├── pipeline.py
│       ├── settings.py
│       └── stats.py
└── .github/workflows/nba_daily_refresh.yml
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and fill in:
   - `ODDS_API_KEY`
   - `ADMIN_PASSPHRASE`
   - optionally GitHub values for workflow dispatch from admin UI (`GITHUB_OWNER`, `GITHUB_REPO`, `GITHUB_WORKFLOW_FILE`, `GITHUB_TOKEN`)

## Run locally

Public app:

```bash
streamlit run app_public.py
```

Admin app:

```bash
streamlit run app_admin.py
```

## Daily automation at noon ET

Workflow file: `.github/workflows/nba_daily_refresh.yml`

- Scheduled daily and DST-safe by running at both `16:00` and `17:00` UTC, then filtering in-step for exactly noon in `America/New_York`.
- Pulls latest lines from The Odds API.
- Retrains model + regenerates prediction files.
- Commits updated `data/latest_predictions.json` and `data/latest_summary.json`.

Set repo secret:

- `ODDS_API_KEY`
- `DISCORD_WEBHOOK_URL` (or `DISCORD_WEBHOOK`) to post refreshed picks to Discord from CI

## Notes for embedding

Streamlit apps are embedded with an `<iframe>` from a deployed URL (Streamlit Cloud, your own VM, etc). Example snippet is included in the assistant response.

## Training data resilience

- The trainer uses recent completed NBA seasons by default (currently goes through last completed season).
- If Basketball-Reference requests fail in your hosting environment, the app falls back to the bundled cache:
  - `data/historical_game_logs.csv`

## Public record tracking (YTD / yesterday)

- Prediction history is persisted to `data/picks_history.json`.
- As final scores become available, the app computes moneyline/totals records for:
  - yesterday
  - current season YTD
- If no completed games exist yet (e.g., before opening night), cards correctly show `0-0` and `0.0%`.