from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


@dataclass
class MarketPrices:
    home_price: float | None = None
    away_price: float | None = None
    total_line: float | None = None
    over_price: float | None = None
    under_price: float | None = None
    source_bookmakers: int = 0


def american_to_implied_prob(american_odds: float | int | None) -> float | None:
    if american_odds is None or np.isnan(float(american_odds)):
        return None
    odds = float(american_odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _as_timestamp(iso_ts: str) -> datetime:
    return datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))


def fetch_raw_odds(
    api_key: str,
    regions: str = "us",
    markets: str = "h2h,totals",
    bookmakers: str = "",
) -> list[dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    response = requests.get(ODDS_API_BASE, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _consensus_moneyline(bookmakers: list[dict[str, Any]], home_team: str, away_team: str) -> tuple[float | None, float | None]:
    home_prices: list[float] = []
    away_prices: list[float] = []
    for book in bookmakers:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = outcome.get("price")
                if price is None:
                    continue
                if name == home_team:
                    home_prices.append(float(price))
                elif name == away_team:
                    away_prices.append(float(price))
    home_consensus = float(np.mean(home_prices)) if home_prices else None
    away_consensus = float(np.mean(away_prices)) if away_prices else None
    return home_consensus, away_consensus


def _consensus_total(bookmakers: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    totals: list[float] = []
    over_prices: list[float] = []
    under_prices: list[float] = []
    for book in bookmakers:
        for market in book.get("markets", []):
            if market.get("key") != "totals":
                continue
            outcomes = market.get("outcomes", [])
            line_values = [outcome.get("point") for outcome in outcomes if outcome.get("point") is not None]
            if line_values:
                totals.extend([float(x) for x in line_values])
            for outcome in outcomes:
                name = str(outcome.get("name", "")).lower()
                price = outcome.get("price")
                if price is None:
                    continue
                if name == "over":
                    over_prices.append(float(price))
                elif name == "under":
                    under_prices.append(float(price))
    total_consensus = float(np.mean(totals)) if totals else None
    over_consensus = float(np.mean(over_prices)) if over_prices else None
    under_consensus = float(np.mean(under_prices)) if under_prices else None
    return total_consensus, over_consensus, under_consensus


def odds_to_dataframe(raw_games: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in raw_games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        bookmakers = game.get("bookmakers", [])
        home_ml, away_ml = _consensus_moneyline(bookmakers, home_team=home_team, away_team=away_team)
        total_line, over_price, under_price = _consensus_total(bookmakers)
        rows.append(
            {
                "game_id": game.get("id"),
                "commence_time": _as_timestamp(game["commence_time"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "total_line": total_line,
                "over_price": over_price,
                "under_price": under_price,
                "bookmaker_count": len(bookmakers),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["home_implied_prob"] = df["home_moneyline"].apply(american_to_implied_prob)
    df["away_implied_prob"] = df["away_moneyline"].apply(american_to_implied_prob)
    df["over_implied_prob"] = df["over_price"].apply(american_to_implied_prob)
    df["under_implied_prob"] = df["under_price"].apply(american_to_implied_prob)
    df = df.sort_values("commence_time").reset_index(drop=True)
    return df
