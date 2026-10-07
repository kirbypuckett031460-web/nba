from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


def _pick_confidence_prob(row: pd.Series) -> float | None:
    pick = str(row.get("moneyline_pick", ""))
    if pick.endswith(" ML") and pick.startswith(str(row.get("home_team", ""))):
        val = row.get("model_home_win_prob")
    elif pick.endswith(" ML") and pick.startswith(str(row.get("away_team", ""))):
        val = row.get("model_away_win_prob")
    else:
        return None
    if val is None or pd.isna(val):
        return None
    val = float(val)
    if val > 1:
        val = val / 100.0
    if 0 <= val <= 1:
        return val
    return None


def _tot_confidence_prob(row: pd.Series) -> float | None:
    pick = str(row.get("total_pick", ""))
    if pick == "Over":
        val = row.get("model_over_prob")
    elif pick == "Under":
        val = row.get("model_under_prob")
    else:
        return None
    if val is None or pd.isna(val):
        return None
    val = float(val)
    if val > 1:
        val = val / 100.0
    if 0 <= val <= 1:
        return val
    return None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_edge(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.1f}%"


def _to_et(ts: Any) -> str:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        return "—"
    return dt.tz_convert("America/New_York").strftime("%I:%M %p ET")


def _top_moneyline_lines(frame: pd.DataFrame, limit: int = 8) -> str:
    if frame.empty:
        return "No games."
    ml = frame.copy()
    ml["conf_prob"] = ml.apply(_pick_confidence_prob, axis=1)
    ml = ml.sort_values("best_moneyline_edge_pct", ascending=False, na_position="last").head(limit)
    lines: list[str] = []
    for _, row in ml.iterrows():
        lines.append(
            f"{_to_et(row.get('commence_time'))} | {row.get('away_team')} @ {row.get('home_team')} | "
            f"{row.get('moneyline_pick')} | edge {_fmt_edge(row.get('best_moneyline_edge_pct'))} | conf {_fmt_pct(row.get('conf_prob'))}"
        )
    return "\n".join(lines)


def _top_totals_lines(frame: pd.DataFrame, limit: int = 8) -> str:
    if frame.empty:
        return "No games."
    tot = frame.copy()
    tot["edge_raw"] = tot.apply(
        lambda r: r.get("over_edge_pct") if str(r.get("total_pick", "")) == "Over" else r.get("under_edge_pct"),
        axis=1,
    )
    tot["conf_prob"] = tot.apply(_tot_confidence_prob, axis=1)
    tot = tot.sort_values("edge_raw", ascending=False, na_position="last").head(limit)
    lines: list[str] = []
    for _, row in tot.iterrows():
        lines.append(
            f"{_to_et(row.get('commence_time'))} | {row.get('away_team')} @ {row.get('home_team')} | "
            f"{row.get('total_pick')} {row.get('total_line')} | edge {_fmt_edge(row.get('edge_raw'))} | conf {_fmt_pct(row.get('conf_prob'))}"
        )
    return "\n".join(lines)


def build_discord_payload(predictions_payload: dict[str, Any]) -> dict[str, Any]:
    games = predictions_payload.get("games", []) or []
    frame = pd.DataFrame(games)
    game_count = len(frame)
    generated_at = str(predictions_payload.get("generated_at") or "")
    filter_date = predictions_payload.get("filter_date")
    filter_tz = predictions_payload.get("filter_timezone") or "America/New_York"
    title_scope = f"{filter_date} ({filter_tz})" if filter_date else "All upcoming dates"

    if generated_at:
        try:
            generated_disp = (
                pd.to_datetime(generated_at, utc=True, errors="coerce")
                .tz_convert("America/New_York")
                .strftime("%Y-%m-%d %I:%M %p ET")
            )
        except Exception:
            generated_disp = generated_at
    else:
        generated_disp = "N/A"

    description = (
        f"**Scope:** {title_scope}\n"
        f"**Games:** {game_count}\n"
        f"**Generated:** {generated_disp}"
    )

    return {
        "content": "🏀 NBA picks refresh complete",
        "embeds": [
            {
                "title": "NBA Model Picks",
                "description": description,
                "color": 3447003,
                "fields": [
                    {"name": "Moneyline Picks", "value": _top_moneyline_lines(frame), "inline": False},
                    {"name": "Over/Under Picks", "value": _top_totals_lines(frame), "inline": False},
                ],
            }
        ],
    }


def main() -> None:
    webhook = (
        (os.getenv("DISCORD_WEBHOOK_URL", "") or "").strip()
        or (os.getenv("DISCORD_WEBHOOK", "") or "").strip()
    )
    if not webhook:
        print("DISCORD_WEBHOOK_URL / DISCORD_WEBHOOK not set; skipping Discord post.")
        return

    predictions_path = Path(os.getenv("PREDICTIONS_PATH", "data/latest_predictions.json"))
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    discord_payload = build_discord_payload(payload)

    if os.getenv("DISCORD_DRY_RUN", "").strip() == "1":
        print(json.dumps(discord_payload, indent=2))
        return

    response = requests.post(webhook, json=discord_payload, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"Discord webhook failed ({response.status_code}): {response.text}")
    print(f"Posted picks to Discord ({response.status_code}).")


if __name__ == "__main__":
    main()
