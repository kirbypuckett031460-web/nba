from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from nba_api.stats.static import teams


def normalize_team_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace("*", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def team_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for team in teams.get_teams():
        variants = {
            team["full_name"],
            team["nickname"],
            team["abbreviation"],
            f"{team['city']} {team['nickname']}",
            team["city"],
        }
        for variant in variants:
            lookup[normalize_team_name(variant)] = team
    return lookup


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _season_end_year(season: str) -> int:
    start = int(season.split("-")[0])
    return start + 1


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        )
    }
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, headers=headers, timeout=40)
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_exc = exc
            # Treat 4xx (except 429) as not retriable and skip.
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code and 400 <= status_code < 500 and status_code != 429:
                return ""
            if attempt == 3:
                break
            time.sleep(2**attempt)

    if last_exc:
        raise last_exc
    return ""


def _bref_month_urls(season_end_year: int) -> list[str]:
    base = f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_games.html"
    html = _fetch_html(base)
    if not html:
        return []
    matches = sorted(set(re.findall(rf"/leagues/NBA_{season_end_year}_games-[a-z]+\.html", html)))
    return [f"https://www.basketball-reference.com{path}" for path in matches]


def _bref_games_for_season(season: str) -> pd.DataFrame:
    season_end = _season_end_year(season)
    month_urls = _bref_month_urls(season_end)
    if not month_urls:
        return pd.DataFrame()

    tables: list[pd.DataFrame] = []
    for month_url in month_urls:
        try:
            month_tables = pd.read_html(month_url)
        except Exception:
            continue
        if not month_tables:
            continue
        table = month_tables[0]
        table.columns = [str(col).strip() for col in table.columns]
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    season_games = pd.concat(tables, ignore_index=True)
    season_games = season_games.rename(
        columns={
            "Visitor/Neutral": "away_team",
            "Home/Neutral": "home_team",
            "PTS": "away_points",
            "PTS.1": "home_points",
            "Date": "game_date",
        }
    )
    keep = ["game_date", "away_team", "home_team", "away_points", "home_points"]
    season_games = season_games[[col for col in keep if col in season_games.columns]].copy()
    season_games = _coerce_numeric(season_games, ["away_points", "home_points"])
    season_games = season_games.dropna(subset=["away_points", "home_points"])
    season_games["away_points"] = season_games["away_points"].astype(int)
    season_games["home_points"] = season_games["home_points"].astype(int)
    season_games["game_date"] = pd.to_datetime(season_games["game_date"], errors="coerce")
    season_games = season_games.dropna(subset=["game_date"]).copy()
    season_games["season"] = season
    return season_games.sort_values("game_date").reset_index(drop=True)


def fetch_game_logs(seasons: list[str]) -> pd.DataFrame:
    lookup = team_lookup()
    rows: list[dict[str, Any]] = []
    for season in seasons:
        try:
            season_games = _bref_games_for_season(season)
        except Exception:
            continue
        for _, game in season_games.iterrows():
            game_date = pd.to_datetime(game["game_date"])
            away_team = str(game["away_team"])
            home_team = str(game["home_team"])
            away_points = int(game["away_points"])
            home_points = int(game["home_points"])
            game_id = f"{season}_{game_date.strftime('%Y%m%d')}_{normalize_team_name(away_team)}_{normalize_team_name(home_team)}"

            away_meta = lookup.get(normalize_team_name(away_team), {})
            home_meta = lookup.get(normalize_team_name(home_team), {})
            away_id = int(away_meta.get("id", abs(hash(away_team)) % 100000))
            home_id = int(home_meta.get("id", abs(hash(home_team)) % 100000))

            rows.append(
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date,
                    "TEAM_ID": home_id,
                    "TEAM_NAME": home_team,
                    "MATCHUP": f"{home_team} vs. {away_team}",
                    "WL": "W" if home_points > away_points else "L",
                    "PTS": home_points,
                    "PLUS_MINUS": home_points - away_points,
                    "SEASON": season,
                }
            )
            rows.append(
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": game_date,
                    "TEAM_ID": away_id,
                    "TEAM_NAME": away_team,
                    "MATCHUP": f"{away_team} @ {home_team}",
                    "WL": "W" if away_points > home_points else "L",
                    "PTS": away_points,
                    "PLUS_MINUS": away_points - home_points,
                    "SEASON": season,
                }
            )

    if not rows:
        raise ValueError("No game logs found from Basketball Reference.")
    logs = pd.DataFrame(rows)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    return logs.sort_values("GAME_DATE").reset_index(drop=True)


def fetch_team_season_snapshot(season: str) -> pd.DataFrame:
    season_end = _season_end_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_end}.html"
    tables = pd.read_html(url)
    candidate_tables = []
    for table in tables:
        cols = set(str(c) for c in table.columns)
        required = {"Team", "FG", "FGA", "3P", "FTA", "ORB", "DRB", "TRB", "AST", "TOV", "PTS"}
        if required.issubset(cols):
            candidate_tables.append(table.copy())

    if len(candidate_tables) < 2:
        raise ValueError(f"Could not locate offense/defense team tables for season {season}.")

    team_off = candidate_tables[0]
    team_opp = candidate_tables[1]

    for frame in (team_off, team_opp):
        frame["Team"] = frame["Team"].astype(str).str.replace("*", "", regex=False).str.strip()
        frame.dropna(subset=["Team"], inplace=True)
        frame = frame[~frame["Team"].str.contains("League Average", na=False)]

    team_off = team_off[~team_off["Team"].str.contains("League Average", na=False)].copy()
    team_opp = team_opp[~team_opp["Team"].str.contains("League Average", na=False)].copy()

    numeric_cols = [c for c in team_off.columns if c not in {"Team"}]
    team_off = _coerce_numeric(team_off, numeric_cols)
    team_opp = _coerce_numeric(team_opp, numeric_cols)

    merged = team_off.merge(team_opp, on="Team", suffixes=("", "_opp"))
    merged.rename(columns={"Team": "TEAM_NAME"}, inplace=True)

    off_poss = merged["FGA"] + 0.44 * merged["FTA"] - merged["ORB"] + merged["TOV"]
    def_poss = merged["FGA_opp"] + 0.44 * merged["FTA_opp"] - merged["ORB_opp"] + merged["TOV_opp"]
    pace = (off_poss + def_poss) / 2.0

    merged["adv_off_rating"] = (merged["PTS"] / off_poss) * 100.0
    merged["adv_def_rating"] = (merged["PTS_opp"] / def_poss) * 100.0
    merged["adv_net_rating"] = merged["adv_off_rating"] - merged["adv_def_rating"]
    merged["adv_pace"] = pace
    merged["adv_ts_pct"] = merged["PTS"] / (2.0 * (merged["FGA"] + 0.44 * merged["FTA"]))
    merged["adv_efg_pct"] = (merged["FG"] + 0.5 * merged["3P"]) / merged["FGA"]
    merged["adv_tov_pct"] = (merged["TOV"] / off_poss) * 100.0
    merged["adv_oreb_pct"] = (merged["ORB"] / (merged["ORB"] + merged["DRB_opp"])) * 100.0
    merged["adv_dreb_pct"] = (merged["DRB"] / (merged["DRB"] + merged["ORB_opp"])) * 100.0
    merged["adv_reb_pct"] = (merged["TRB"] / (merged["TRB"] + merged["TRB_opp"])) * 100.0
    merged["adv_ft_rate"] = merged["FTA"] / merged["FGA"]
    merged["adv_ast_ratio"] = (merged["AST"] / off_poss) * 100.0

    merged["ff_efg_pct"] = merged["adv_efg_pct"]
    merged["ff_tm_tov_pct"] = merged["adv_tov_pct"]
    merged["ff_oreb_pct"] = merged["adv_oreb_pct"]
    merged["ff_ft_rate"] = merged["adv_ft_rate"]
    merged["ff_opp_efg_pct"] = (merged["FG_opp"] + 0.5 * merged["3P_opp"]) / merged["FGA_opp"]
    merged["ff_opp_tov_pct"] = (merged["TOV_opp"] / def_poss) * 100.0
    merged["ff_opp_oreb_pct"] = (merged["ORB_opp"] / (merged["ORB_opp"] + merged["DRB"])) * 100.0
    merged["ff_opp_ft_rate"] = merged["FTA_opp"] / merged["FGA_opp"]

    feature_cols = [c for c in merged.columns if c.startswith("adv_") or c.startswith("ff_")]
    output = merged[["TEAM_NAME"] + feature_cols].copy()

    lookup = team_lookup()
    output["TEAM_ID"] = output["TEAM_NAME"].apply(
        lambda n: int(lookup.get(normalize_team_name(n), {}).get("id", abs(hash(str(n))) % 100000))
    )
    output["season"] = season
    return output


def _estimate_possessions(row: pd.Series) -> float:
    if all(col in row.index for col in ["FGA", "FTA", "OREB", "TOV"]) and pd.notna(row.get("FGA")):
        return float(row["FGA"] + 0.44 * row["FTA"] - row["OREB"] + row["TOV"])
    # Proxy pace for schedule-level data without box score columns.
    return float(max((row["PTS"] + row["opp_pts"]) / 2.0, 80.0))


def build_team_form(logs: pd.DataFrame, lookback_games: int = 10) -> pd.DataFrame:
    logs = logs.copy()
    logs["is_home"] = logs["MATCHUP"].str.contains("vs.", na=False).astype(int)

    team_game_totals = logs.groupby("GAME_ID")["PTS"].transform("sum")
    logs["opp_pts"] = team_game_totals - logs["PTS"]
    logs["possessions_est"] = logs.apply(_estimate_possessions, axis=1)

    logs = logs.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    logs["days_rest"] = logs.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days.fillna(5).clip(lower=0)
    logs["is_b2b"] = (logs["days_rest"] <= 1).astype(int)

    rolling_specs = {
        "PTS": "rolling_pts_for",
        "opp_pts": "rolling_pts_against",
        "PLUS_MINUS": "rolling_margin",
        "possessions_est": "rolling_possessions",
    }
    for source_col, target_col in rolling_specs.items():
        logs[target_col] = (
            logs.groupby("TEAM_ID")[source_col]
            .transform(lambda s: s.shift(1).rolling(lookback_games, min_periods=3).mean())
        )

    return logs


def to_game_level(team_logs: pd.DataFrame) -> pd.DataFrame:
    home_rows = team_logs[team_logs["is_home"] == 1].copy()
    away_rows = team_logs[team_logs["is_home"] == 0].copy()

    home_cols = {
        "TEAM_ID": "home_team_id",
        "TEAM_NAME": "home_team",
        "PTS": "home_points",
        "rolling_pts_for": "home_rolling_pts_for",
        "rolling_pts_against": "home_rolling_pts_against",
        "rolling_margin": "home_rolling_margin",
        "rolling_possessions": "home_rolling_possessions",
        "days_rest": "home_days_rest",
        "is_b2b": "home_b2b",
        "SEASON": "season",
    }
    away_cols = {
        "TEAM_ID": "away_team_id",
        "TEAM_NAME": "away_team",
        "PTS": "away_points",
        "rolling_pts_for": "away_rolling_pts_for",
        "rolling_pts_against": "away_rolling_pts_against",
        "rolling_margin": "away_rolling_margin",
        "rolling_possessions": "away_rolling_possessions",
        "days_rest": "away_days_rest",
        "is_b2b": "away_b2b",
    }

    home_keep = ["GAME_ID", "GAME_DATE"] + list(home_cols.keys())
    away_keep = ["GAME_ID"] + list(away_cols.keys())
    home_rows = home_rows[home_keep].rename(columns=home_cols)
    away_rows = away_rows[away_keep].rename(columns=away_cols)

    games = home_rows.merge(away_rows, on="GAME_ID", how="inner")
    games["home_win"] = (games["home_points"] > games["away_points"]).astype(int)
    games["total_points"] = games["home_points"] + games["away_points"]
    games = games.sort_values("GAME_DATE").reset_index(drop=True)
    return games


@dataclass
class EloResult:
    game_frame: pd.DataFrame
    latest_elos: dict[int, float]


def add_elo_features(games: pd.DataFrame, k_factor: float = 22.0, home_advantage: float = 70.0) -> EloResult:
    games = games.copy().sort_values("GAME_DATE").reset_index(drop=True)
    ratings: dict[int, float] = {}
    home_elo_pre: list[float] = []
    away_elo_pre: list[float] = []

    for _, row in games.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        h_elo = ratings.get(home_id, 1500.0)
        a_elo = ratings.get(away_id, 1500.0)
        home_elo_pre.append(h_elo)
        away_elo_pre.append(a_elo)

        expected_home = 1.0 / (1.0 + 10.0 ** (((a_elo) - (h_elo + home_advantage)) / 400.0))
        actual_home = float(row["home_win"])
        delta = k_factor * (actual_home - expected_home)
        ratings[home_id] = h_elo + delta
        ratings[away_id] = a_elo - delta

    games["home_elo_pre"] = home_elo_pre
    games["away_elo_pre"] = away_elo_pre
    games["elo_diff"] = games["home_elo_pre"] - games["away_elo_pre"]
    return EloResult(game_frame=games, latest_elos=ratings)
