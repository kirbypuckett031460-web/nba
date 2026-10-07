from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Ensure repo root is importable when this script is executed directly from CI.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nba_model.pipeline import generate_predictions
from src.nba_model.odds import OddsApiAuthError
from src.nba_model.settings import AppSettings


def _write_summary(payload: dict, refresh_status: str, refresh_message: str | None = None) -> None:
    summary_path = Path("data/latest_summary.json")
    summary: dict[str, object] = {
        "generated_at": payload.get("generated_at"),
        "games_count": len(payload.get("games", [])),
        "metrics": payload.get("model_metrics", {}),
        "filter_date": payload.get("filter_date"),
        "filter_timezone": payload.get("filter_timezone"),
        "performance": payload.get("performance", {}),
        "refresh_status": refresh_status,
    }
    if refresh_message:
        summary["refresh_message"] = refresh_message
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _candidate_api_keys() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for env_name in ("ODDS_API_KEY", "THE_ODDS_API_KEY", "ODDSAPI_KEY"):
        value = (os.getenv(env_name, "") or "").strip()
        if value and value not in seen:
            candidates.append(value)
            seen.add(value)
    return candidates


def main() -> None:
    base_settings = AppSettings()
    target_date = (os.getenv("TARGET_DATE", "") or "").strip() or None
    target_timezone = (os.getenv("TARGET_TIMEZONE", "") or "America/New_York").strip()

    key_candidates = _candidate_api_keys()
    if not key_candidates and base_settings.odds_api_key:
        key_candidates = [base_settings.odds_api_key]
    if not key_candidates:
        raise SystemExit(
            "No Odds API key found. Add one of these GitHub Actions secrets: "
            "ODDS_API_KEY (preferred), THE_ODDS_API_KEY, or ODDSAPI_KEY."
        )

    auth_error: OddsApiAuthError | None = None
    for idx, candidate_key in enumerate(key_candidates):
        os.environ["ODDS_API_KEY"] = candidate_key
        settings = AppSettings()
        try:
            payload = generate_predictions(
                settings=settings,
                force_retrain=(idx == 0),
                target_date=target_date,
                target_timezone=target_timezone,
            )
            _write_summary(payload, refresh_status="fresh")
            print(f"Generated {len(payload.get('games', []))} games into {settings.predictions_path}")
            return
        except OddsApiAuthError as exc:
            auth_error = exc
            continue

    if auth_error and base_settings.predictions_path.exists():
        stale_payload = json.loads(base_settings.predictions_path.read_text(encoding="utf-8"))
        _write_summary(
            stale_payload,
            refresh_status="stale_odds_auth_failed",
            refresh_message=str(auth_error),
        )
        print("Odds API auth failed. Kept existing published predictions and updated summary status.")
        return

    if auth_error:
        raise SystemExit(str(auth_error))
    raise SystemExit("Prediction refresh failed before model generation completed.")


if __name__ == "__main__":
    main()
