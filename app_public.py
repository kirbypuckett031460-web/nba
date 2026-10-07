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


TEAM_COLORS: dict[str, tuple[str, str]] = {
    # NHL abbreviations
    "ANA": ("#B9975B", "#111827"),
    "ARI": ("#8C2633", "#F2A900"),
    "BOS": ("#FFB81C", "#111827"),
    "BUF": ("#003087", "#FFFFFF"),
    "CGY": ("#C8102E", "#FFFFFF"),
    "CAR": ("#CC0000", "#FFFFFF"),
    "CHI": ("#CF0A2C", "#FFFFFF"),
    "COL": ("#6F263D", "#FFFFFF"),
    "CBJ": ("#002654", "#FFFFFF"),
    "DAL": ("#006847", "#FFFFFF"),
    "DET": ("#CE1126", "#FFFFFF"),
    "EDM": ("#041E42", "#FF4C00"),
    "FLA": ("#041E42", "#C8102E"),
    "LAK": ("#111111", "#FFFFFF"),
    "MIN": ("#154734", "#DDCBA4"),
    "MTL": ("#AF1E2D", "#FFFFFF"),
    "NSH": ("#FFB81C", "#111827"),
    "NJD": ("#CE1126", "#FFFFFF"),
    "NYI": ("#00539B", "#F47D30"),
    "NYR": ("#0038A8", "#FFFFFF"),
    "OTT": ("#C52032", "#FFFFFF"),
    "PHI": ("#F74902", "#FFFFFF"),
    "PIT": ("#FCB514", "#111827"),
    "SJS": ("#006D75", "#FFFFFF"),
    "SEA": ("#001628", "#99D9D9"),
    "STL": ("#002F87", "#FFB81C"),
    "TBL": ("#002868", "#FFFFFF"),
    "TOR": ("#00205B", "#FFFFFF"),
    "UTA": ("#6E7580", "#FFFFFF"),
    "VAN": ("#00205B", "#FFFFFF"),
    "VGK": ("#B4975A", "#111827"),
    "WSH": ("#041E42", "#C8102E"),
    "WPG": ("#041E42", "#AC162C"),
    # NBA abbreviations
    "ATL": ("#E03A3E", "#FFFFFF"),
    "BKN": ("#111111", "#FFFFFF"),
    "BOS-NBA": ("#007A33", "#FFFFFF"),
    "CHA": ("#1D1160", "#FFFFFF"),
    "CHI-NBA": ("#CE1141", "#FFFFFF"),
    "CLE": ("#860038", "#FDBB30"),
    "DAL-NBA": ("#00538C", "#FFFFFF"),
    "DEN": ("#0E2240", "#FEC524"),
    "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#FFFFFF"),
    "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#FFFFFF"),
    "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#12173F"),
    "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"),
    "MIN-NBA": ("#0C2340", "#78BE20"),
    "NOP": ("#0C2340", "#C8102E"),
    "NYK": ("#006BB6", "#F58426"),
    "OKC": ("#007AC1", "#EF3B24"),
    "ORL": ("#0077C0", "#FFFFFF"),
    "PHI-NBA": ("#006BB6", "#FFFFFF"),
    "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#FFFFFF"),
    "SAC": ("#5A2D81", "#FFFFFF"),
    "SAS": ("#111111", "#C4CED4"),
    "TORNBA": ("#CE1141", "#FFFFFF"),
    "UTA-NBA": ("#002B5C", "#F9A01B"),
    "WAS": ("#002B5C", "#E31837"),
}

TEAM_ALIASES: dict[str, str] = {
    # NHL full names / nicknames
    "ANAHEIM DUCKS": "ANA",
    "ARIZONA COYOTES": "ARI",
    "BOSTON BRUINS": "BOS",
    "BUFFALO SABRES": "BUF",
    "CALGARY FLAMES": "CGY",
    "CAROLINA HURRICANES": "CAR",
    "CHICAGO BLACKHAWKS": "CHI",
    "COLORADO AVALANCHE": "COL",
    "COLUMBUS BLUE JACKETS": "CBJ",
    "DALLAS STARS": "DAL",
    "DETROIT RED WINGS": "DET",
    "EDMONTON OILERS": "EDM",
    "FLORIDA PANTHERS": "FLA",
    "LOS ANGELES KINGS": "LAK",
    "MINNESOTA WILD": "MIN",
    "MONTREAL CANADIENS": "MTL",
    "NASHVILLE PREDATORS": "NSH",
    "NEW JERSEY DEVILS": "NJD",
    "NEW YORK ISLANDERS": "NYI",
    "NEW YORK RANGERS": "NYR",
    "OTTAWA SENATORS": "OTT",
    "PHILADELPHIA FLYERS": "PHI",
    "PITTSBURGH PENGUINS": "PIT",
    "SAN JOSE SHARKS": "SJS",
    "SEATTLE KRAKEN": "SEA",
    "ST LOUIS BLUES": "STL",
    "TAMPA BAY LIGHTNING": "TBL",
    "TORONTO MAPLE LEAFS": "TOR",
    "UTAH HOCKEY CLUB": "UTA",
    "VANCOUVER CANUCKS": "VAN",
    "VEGAS GOLDEN KNIGHTS": "VGK",
    "WASHINGTON CAPITALS": "WSH",
    "WINNIPEG JETS": "WPG",
    # NBA full names / nicknames
    "ATLANTA HAWKS": "ATL",
    "BROOKLYN NETS": "BKN",
    "BOSTON CELTICS": "BOS-NBA",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI-NBA",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL-NBA",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN-NBA",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI-NBA",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TORNBA",
    "UTAH JAZZ": "UTA-NBA",
    "WASHINGTON WIZARDS": "WAS",
}

NHL_ABBR_TO_FULL: dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}

NBA_ABBR_TO_FULL: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


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
df["game_time_et"] = df["commence_time"].dt.tz_convert("America/New_York").dt.strftime("%I:%M %p ET")
df["slate_date_et"] = df["commence_time"].dt.tz_convert("America/New_York").dt.date

for pct_col in ["model_home_win_prob", "model_away_win_prob", "model_over_prob", "model_under_prob"]:
    if pct_col in df.columns:
        df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")

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


def _american_text(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    odds = int(round(float(value)))
    return f"{odds:+d}" if odds > 0 else f"{odds:d}"


def _to_prob(value: float | int | None) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    val = float(value)
    if val < 0:
        return None
    if val > 1:
        val = val / 100.0
    return val if 0.0 <= val <= 1.0 else None


def _team_color_style(team: str | None) -> str:
    if team is None:
        return "background-color: #202b3e; color: #d7e3fa;"
    key = str(team).strip().upper()
    key = TEAM_ALIASES.get(key, key)
    if key in TEAM_COLORS:
        bg, fg = TEAM_COLORS[key]
        return f"background-color: {bg}; color: {fg}; font-weight: 700;"
    # fallback: try first token of full name or unchanged uppercase
    token = key.split()[-1] if " " in key else key
    if token in TEAM_COLORS:
        bg, fg = TEAM_COLORS[token]
        return f"background-color: {bg}; color: {fg}; font-weight: 700;"
    return "background-color: #202b3e; color: #d7e3fa; font-weight: 700;"


def _looks_like_abbr(name: str | None) -> bool:
    if name is None:
        return False
    text = str(name).strip()
    return 2 <= len(text) <= 5 and text.replace("-", "").isalpha() and text.upper() == text


def _infer_league_from_slate(frame: pd.DataFrame) -> str:
    teams = pd.concat(
        [
            frame.get("away_team", pd.Series(dtype=object)).astype(str),
            frame.get("home_team", pd.Series(dtype=object)).astype(str),
        ],
        ignore_index=True,
    )
    abbrs = [t.strip().upper() for t in teams if _looks_like_abbr(t)]
    if not abbrs:
        return "unknown"
    nhl_hits = sum(1 for t in abbrs if t in NHL_ABBR_TO_FULL)
    nba_hits = sum(1 for t in abbrs if t in NBA_ABBR_TO_FULL)
    if nhl_hits > nba_hits:
        return "nhl"
    if nba_hits > nhl_hits:
        return "nba"
    return "unknown"


def _full_name(team: str | None, league: str) -> str:
    if team is None:
        return ""
    text = str(team).strip()
    if not _looks_like_abbr(text):
        return text
    key = text.upper()
    if league == "nhl":
        return NHL_ABBR_TO_FULL.get(key, text)
    if league == "nba":
        return NBA_ABBR_TO_FULL.get(key, text)
    # Fallback: prefer NHL mapping if available, otherwise NBA.
    return NHL_ABBR_TO_FULL.get(key, NBA_ABBR_TO_FULL.get(key, text))


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

league_hint = _infer_league_from_slate(df)

ml = df.copy()
ml["away_display"] = ml["away_team"].map(lambda x: _full_name(x, league_hint))
ml["home_display"] = ml["home_team"].map(lambda x: _full_name(x, league_hint))
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
ml["mkt"] = pd.to_numeric(ml["mkt"], errors="coerce")
ml["fair"] = ml["fair_prob"].apply(_prob_to_american)
ml["edge_text"] = ml["best_moneyline_edge_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
ml["confidence_prob"] = ml["fair_prob"].map(_to_prob)
ml["confidence"] = ml["confidence_prob"].map(lambda x: f"{x*100:.1f}%" if x is not None else "—")
ml["mkt_text"] = ml["mkt"].map(_american_text)
ml["pick_display"] = np.where(
    is_home_pick,
    ml["home_display"],
    np.where(is_away_pick, ml["away_display"], ml["pick_team"].map(lambda x: _full_name(x, league_hint))),
)
ml.loc[is_pass, "pick_display"] = "Pass"
ml_table = ml[
    [
        "game_time_et",
        "away_display",
        "home_display",
        "mkt_text",
        "fair",
        "pick_display",
        "edge_text",
        "confidence",
        "best_moneyline_edge_pct",
        "confidence_prob",
    ]
].rename(
    columns={
        "game_time_et": "Game Time (ET)",
        "away_display": "Away",
        "home_display": "Home",
        "mkt_text": "Mkt",
        "fair": "Fair",
        "pick_display": "Pick",
        "edge_text": "Edge",
        "confidence": "Confidence",
    }
)
ml_table = ml_table.sort_values("best_moneyline_edge_pct", ascending=False, na_position="last")
ml_display = ml_table.drop(columns=["best_moneyline_edge_pct", "confidence_prob"]).copy()

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


def _ou_pick_color(value: str | None) -> str:
    pick = str(value or "").strip().lower()
    if pick == "over":
        return "background-color: #0f8f6f; color: #eafff8; font-weight: 700;"
    if pick == "under":
        return "background-color: #5b1f2f; color: #ffd9e3; font-weight: 700;"
    return "background-color: #2e3b54; color: #dbe8ff; font-weight: 700;"


ml_styler = (
    ml_display.style.hide(axis="index")
    .apply(lambda s: [_edge_color(v) for v in ml_table["best_moneyline_edge_pct"]], subset=["Edge"])
    .apply(lambda s: [_confidence_color(v) for v in ml_table["confidence_prob"]], subset=["Confidence"])
    .apply(lambda s: [_team_color_style(v) for v in ml_table["Pick"]], subset=["Pick"])
)

with tab_ml:
    st.dataframe(ml_styler, use_container_width=True)

ou = df.copy()
ou["away_display"] = ou["away_team"].map(lambda x: _full_name(x, league_hint))
ou["home_display"] = ou["home_team"].map(lambda x: _full_name(x, league_hint))
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
ou["confidence_prob"] = ou["confidence_raw"].map(_to_prob)
ou["edge"] = ou["edge_raw"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
ou["confidence"] = ou["confidence_prob"].map(lambda x: f"{x*100:.1f}%" if x is not None else "—")
ou_table = ou[
    [
        "game_time_et",
        "away_display",
        "home_display",
        "total_line",
        "model_projected_total",
        "pick",
        "edge",
        "confidence",
        "edge_raw",
        "confidence_prob",
    ]
].rename(
    columns={
        "game_time_et": "Game Time (ET)",
        "away_display": "Away",
        "home_display": "Home",
        "total_line": "Line",
        "model_projected_total": "Projection",
        "pick": "Pick",
        "edge": "Edge",
        "confidence": "Confidence",
    }
)
ou_table = ou_table.sort_values("edge_raw", ascending=False, na_position="last")
ou_display = ou_table.drop(columns=["edge_raw", "confidence_prob"]).copy()

ou_styler = (
    ou_display.style.hide(axis="index")
    .apply(lambda s: [_edge_color(v) for v in ou_table["edge_raw"]], subset=["Edge"])
    .apply(lambda s: [_confidence_color(v) for v in ou_table["confidence_prob"]], subset=["Confidence"])
    .apply(lambda s: [_ou_pick_color(v) for v in ou_table["Pick"]], subset=["Pick"])
)

with tab_ou:
    st.dataframe(ou_styler, use_container_width=True)

st.caption(f"Last updated: {payload.get('generated_at')} | Source: {payload_source}")
