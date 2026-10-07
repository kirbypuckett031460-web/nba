from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def _fmt_american(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    odds = int(round(float(value)))
    return f"{odds:+d}" if odds > 0 else str(odds)


def _fmt_float(value: Any, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _to_et(ts: Any) -> str:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        return "—"
    return dt.tz_convert("America/New_York").strftime("%I:%M %p ET")


def _format_cell(value: Any, width: int) -> str:
    text = str(value if value is not None else "—")
    if len(text) > width:
        return text[: max(width - 1, 1)] + "…"
    return text.ljust(width)


def _auto_widths(headers: list[str], rows: list[list[Any]], max_col_width: int = 24) -> list[int]:
    widths: list[int] = []
    for idx, header in enumerate(headers):
        col_max = len(str(header))
        for row in rows:
            if idx < len(row):
                col_max = max(col_max, len(str(row[idx])))
        widths.append(min(col_max, max_col_width))
    return widths


def _table_block(
    headers: list[str],
    rows: list[list[Any]],
    widths: list[int],
    max_chars: int = 1700,
) -> str:
    header_line = " | ".join(_format_cell(h, w) for h, w in zip(headers, widths))
    divider = "-+-".join("-" * w for w in widths)
    lines = [header_line, divider]
    kept = 0
    for row in rows:
        row_line = " | ".join(_format_cell(v, w) for v, w in zip(row, widths))
        candidate = "```text\n" + "\n".join(lines + [row_line]) + "\n```"
        if len(candidate) > max_chars and kept > 0:
            break
        lines.append(row_line)
        kept += 1
    block = "```text\n" + "\n".join(lines) + "\n```"
    omitted = max(len(rows) - kept, 0)
    if omitted > 0:
        block += f"\n…and {omitted} more game(s)."
    return block


def _moneyline_table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No games."
    ml = frame.copy()
    ml["conf_prob"] = ml.apply(_pick_confidence_prob, axis=1)
    ml["pick_team"] = ml["moneyline_pick"].astype(str).str.replace(" ML", "", regex=False)
    ml["mkt"] = ml.apply(
        lambda r: r.get("home_moneyline")
        if str(r.get("pick_team", "")) == str(r.get("home_team", ""))
        else (r.get("away_moneyline") if str(r.get("pick_team", "")) == str(r.get("away_team", "")) else None),
        axis=1,
    )
    ml["fair"] = ml["conf_prob"].apply(
        lambda p: "—"
        if p is None
        else (f"{-round((p / (1 - p)) * 100):d}" if p >= 0.5 else f"+{round(((1 - p) / p) * 100):d}")
    )
    ml = ml.sort_values("commence_time", ascending=True, na_position="last")
    rows: list[list[Any]] = []
    for _, row in ml.iterrows():
        rows.append(
            [
                _to_et(row.get("commence_time")),
                row.get("away_team"),
                row.get("home_team"),
                _fmt_american(row.get("mkt")),
                row.get("fair"),
                row.get("pick_team"),
                _fmt_edge(row.get("best_moneyline_edge_pct")),
                _fmt_pct(row.get("conf_prob")),
            ]
        )
    headers = ["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]
    widths = _auto_widths(headers, rows, max_col_width=24)
    return _table_block(headers=headers, rows=rows, widths=widths)


def _totals_table_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No games."
    tot = frame.copy()
    tot["edge_raw"] = tot.apply(
        lambda r: r.get("over_edge_pct") if str(r.get("total_pick", "")) == "Over" else r.get("under_edge_pct"),
        axis=1,
    )
    tot["conf_prob"] = tot.apply(_tot_confidence_prob, axis=1)
    tot = tot.sort_values("commence_time", ascending=True, na_position="last")
    rows: list[list[Any]] = []
    for _, row in tot.iterrows():
        rows.append(
            [
                _to_et(row.get("commence_time")),
                row.get("away_team"),
                row.get("home_team"),
                _fmt_float(row.get("total_line"), 1),
                _fmt_float(row.get("model_projected_total"), 1),
                row.get("total_pick"),
                _fmt_edge(row.get("edge_raw")),
                _fmt_pct(row.get("conf_prob")),
            ]
        )
    headers = ["Game Time (ET)", "Away", "Home", "Line", "Proj", "Pick", "Edge", "Confidence"]
    widths = _auto_widths(headers, rows, max_col_width=24)
    return _table_block(headers=headers, rows=rows, widths=widths)


def build_discord_messages(predictions_payload: dict[str, Any]) -> list[dict[str, Any]]:
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
        f"Scope: {title_scope}\n"
        f"Games: {game_count}\n"
        f"Generated: {generated_disp}"
    )

    return [
        {"content": f"🏀 NBA picks\n{description}"},
        {"content": f"Moneyline Picks\n{_moneyline_table_text(frame)}"},
        {"content": f"Over/Under Picks\n{_totals_table_text(frame)}"},
    ]


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
    discord_messages = build_discord_messages(payload)

    if os.getenv("DISCORD_DRY_RUN", "").strip() == "1":
        print(json.dumps(discord_messages, indent=2))
        return

    for msg in discord_messages:
        response = requests.post(webhook, json=msg, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(f"Discord webhook failed ({response.status_code}): {response.text}")
    print(f"Posted picks to Discord ({len(discord_messages)} message(s)).")


if __name__ == "__main__":
    main()
