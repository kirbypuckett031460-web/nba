from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _first_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "")
        if value and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class AppSettings:
    odds_api_key: str = _first_env("ODDS_API_KEY", "THE_ODDS_API_KEY", "ODDSAPI_KEY")
    odds_regions: str = os.getenv("ODDS_REGIONS", "us")
    odds_markets: str = os.getenv("ODDS_MARKETS", "h2h,totals")
    odds_bookmakers: str = os.getenv("ODDS_BOOKMAKERS", "")
    model_path: Path = Path(os.getenv("MODEL_PATH", "data/model_artifact.joblib"))
    predictions_path: Path = Path(os.getenv("PREDICTIONS_PATH", "data/latest_predictions.json"))
    picks_history_path: Path = Path(os.getenv("PICKS_HISTORY_PATH", "data/picks_history.json"))
    lookback_seasons: int = int(os.getenv("LOOKBACK_SEASONS", "5"))
    min_edge_pct: float = float(os.getenv("MIN_EDGE_PCT", "2.0"))
    admin_passphrase: str = os.getenv("ADMIN_PASSPHRASE", "").strip()
    github_owner: str = os.getenv("GITHUB_OWNER", "").strip()
    github_repo: str = os.getenv("GITHUB_REPO", "").strip()
    github_workflow_file: str = os.getenv("GITHUB_WORKFLOW_FILE", "nba_daily_refresh.yml").strip()
    github_token: str = os.getenv("GITHUB_TOKEN", "").strip()

    def ensure_paths(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self.picks_history_path.parent.mkdir(parents=True, exist_ok=True)


def season_strings(lookback: int) -> list[str]:
    """Return NBA season identifiers like 2024-25 ordered oldest->newest."""
    from datetime import datetime

    now = datetime.utcnow()
    # Train through the most recently completed season to avoid relying on
    # upcoming/current-season schedules with sparse or missing results.
    current_season_start = now.year - 1
    seasons: list[str] = []
    for year in range(current_season_start - lookback + 1, current_season_start + 1):
        seasons.append(f"{year}-{str(year + 1)[-2:]}")
    return seasons
