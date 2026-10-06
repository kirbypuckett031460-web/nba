from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

# Ensure repo root is importable when this script is executed directly from CI.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nba_model.pipeline import generate_predictions
from src.nba_model.settings import AppSettings


def main() -> None:
    settings = AppSettings()
    payload = generate_predictions(settings=settings, force_retrain=True)

    summary_path = Path("data/latest_summary.json")
    summary = {
        "generated_at": payload.get("generated_at"),
        "games_count": len(payload.get("games", [])),
        "metrics": payload.get("model_metrics", {}),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Generated {summary['games_count']} games into {settings.predictions_path}")


if __name__ == "__main__":
    main()
