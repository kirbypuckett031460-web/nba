from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont


NBA_TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "ATL": (224, 58, 62),
    "BKN": (17, 17, 17),
    "BOS": (0, 122, 51),
    "CHA": (29, 17, 96),
    "CHI": (206, 17, 65),
    "CLE": (134, 0, 56),
    "DAL": (0, 83, 140),
    "DEN": (14, 34, 64),
    "DET": (206, 17, 65),
    "GSW": (29, 66, 138),
    "HOU": (206, 17, 65),
    "IND": (0, 45, 98),
    "LAC": (200, 16, 46),
    "LAL": (85, 37, 131),
    "MEM": (93, 118, 169),
    "MIA": (152, 0, 46),
    "MIL": (0, 71, 27),
    "MIN": (12, 35, 64),
    "NOP": (12, 35, 64),
    "NYK": (0, 107, 182),
    "OKC": (0, 122, 193),
    "ORL": (0, 119, 192),
    "PHI": (0, 107, 182),
    "PHX": (29, 17, 96),
    "POR": (224, 58, 62),
    "SAC": (90, 45, 129),
    "SAS": (17, 17, 17),
    "TOR": (206, 17, 65),
    "UTA": (0, 43, 92),
    "WAS": (0, 43, 92),
}

NBA_TEAM_NAME_TO_ABBR_RAW: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Brooklyn Nets": "BKN",
    "Boston Celtics": "BOS",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def _normalized_pick(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalized_team_key(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().replace(".", "").split())


NBA_TEAM_NAME_TO_ABBR: dict[str, str] = {
    _normalized_team_key(name): abbr for name, abbr in NBA_TEAM_NAME_TO_ABBR_RAW.items()
}


def _team_to_abbr(value: Any) -> str:
    key = _normalized_team_key(value)
    if not key:
        return "—"
    if key in NBA_TEAM_COLORS:
        return key
    if key in NBA_TEAM_NAME_TO_ABBR:
        return NBA_TEAM_NAME_TO_ABBR[key]
    key_compact = key.replace("-NBA", "").replace(" NBA", "")
    if key_compact in NBA_TEAM_COLORS:
        return key_compact
    text = str(value or "").strip()
    return text if text else "—"


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
    pick = _normalized_pick(row.get("total_pick"))
    if pick.startswith("over"):
        val = row.get("model_over_prob")
    elif pick.startswith("under"):
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


def _fmt_half_line(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    snapped = round(float(value) * 2.0) / 2.0
    return f"{snapped:.1f}"


def _fmt_whole(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return str(int(round(float(value))))


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


def _moneyline_table_data(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
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
    rows: list[dict[str, Any]] = []
    for _, row in ml.iterrows():
        pick_abbr = _team_to_abbr(row.get("pick_team"))
        rows.append(
            {
                "Game Time (ET)": _to_et(row.get("commence_time")),
                "Away": row.get("away_team"),
                "Home": row.get("home_team"),
                "Mkt": _fmt_american(row.get("mkt")),
                "Fair": row.get("fair"),
                "Pick": pick_abbr,
                "Edge": _fmt_edge(row.get("best_moneyline_edge_pct")),
                "Confidence": _fmt_pct(row.get("conf_prob")),
                "__edge_val": row.get("best_moneyline_edge_pct"),
                "__conf_val": row.get("conf_prob"),
                "__pick_type": "moneyline",
                "__pick_abbr": pick_abbr,
            }
        )
    return rows


def _totals_table_data(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    tot = frame.copy()
    tot["pick_norm"] = tot["total_pick"].map(_normalized_pick)
    tot["edge_raw"] = np.where(
        tot["pick_norm"].str.startswith("over"),
        tot["over_edge_pct"],
        np.where(tot["pick_norm"].str.startswith("under"), tot["under_edge_pct"], np.nan),
    )
    tot["conf_prob"] = tot.apply(_tot_confidence_prob, axis=1)
    tot = tot.sort_values("commence_time", ascending=True, na_position="last")
    rows: list[dict[str, Any]] = []
    for _, row in tot.iterrows():
        rows.append(
            {
                "Game Time (ET)": _to_et(row.get("commence_time")),
                "Away": row.get("away_team"),
                "Home": row.get("home_team"),
                "Line": _fmt_half_line(row.get("total_line")),
                "Proj": _fmt_whole(row.get("model_projected_total")),
                "Pick": row.get("total_pick"),
                "Edge": _fmt_edge(row.get("edge_raw")),
                "Confidence": _fmt_pct(row.get("conf_prob")),
                "__edge_val": row.get("edge_raw"),
                "__conf_val": row.get("conf_prob"),
                "__pick_type": "total",
            }
        )
    return rows


def _edge_bg(edge_val: Any) -> tuple[int, int, int]:
    try:
        v = float(edge_val)
    except Exception:
        return (38, 51, 71)
    if v >= 8:
        return (15, 143, 111)
    if v >= 3:
        return (28, 109, 89)
    if v >= 0:
        return (46, 59, 84)
    return (91, 31, 47)


def _conf_bg(conf_val: Any) -> tuple[int, int, int]:
    try:
        v = float(conf_val)
    except Exception:
        return (38, 51, 71)
    if v > 1:
        v = v / 100.0
    if v >= 0.7:
        return (20, 95, 106)
    if v >= 0.6:
        return (28, 79, 102)
    return (45, 62, 87)


def _pick_bg(value: Any, row: dict[str, Any] | None = None) -> tuple[int, int, int]:
    if row and row.get("__pick_type") == "moneyline":
        abbr = str(row.get("__pick_abbr") or _team_to_abbr(value)).upper()
        team_color = NBA_TEAM_COLORS.get(abbr)
        if team_color:
            return team_color
        return (30, 95, 180)

    text = _normalized_pick(value)
    if text.startswith("over"):
        return (15, 143, 111)
    if text.startswith("under"):
        return (206, 17, 65)
    return (30, 95, 180)


def _truncate_for_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_px: int) -> str:
    value = str(text)
    if draw.textlength(value, font=font) <= max_px:
        return value
    ellipsis = "…"
    while value and draw.textlength(value + ellipsis, font=font) > max_px:
        value = value[:-1]
    return (value + ellipsis) if value else ellipsis


def _draw_table(
    draw: ImageDraw.ImageDraw,
    left_x: int,
    start_y: int,
    headers: list[str],
    rows: list[dict[str, Any]],
    widths: dict[str, int],
    row_h: int,
    font_cell: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_header: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    # Header row
    y = start_y
    cx = left_x
    for h in headers:
        w = widths[h]
        draw.rectangle([cx, y, cx + w, y + row_h], fill=(45, 55, 79), outline=(61, 74, 103))
        draw.text((cx + 8, y + 9), h, fill=(216, 225, 247), font=font_header)
        cx += w
    y += row_h

    for idx, row in enumerate(rows):
        base_bg = (23, 32, 50) if idx % 2 == 0 else (20, 29, 46)
        cx = left_x
        for h in headers:
            w = widths[h]
            bg = base_bg
            if h == "Pick":
                bg = _pick_bg(row.get(h), row=row)
            elif h == "Edge":
                bg = _edge_bg(row.get("__edge_val"))
            elif h == "Confidence":
                bg = _conf_bg(row.get("__conf_val"))
            draw.rectangle([cx, y, cx + w, y + row_h], fill=bg, outline=(47, 58, 83))
            text = _truncate_for_width(draw, str(row.get(h, "—")), font_cell, max_px=w - 14)
            draw.text((cx + 7, y + 9), text, fill=(236, 241, 252), font=font_cell)
            cx += w
        y += row_h
    return y


def _slate_date_text(frame: pd.DataFrame, payload: dict[str, Any]) -> str:
    filter_date = payload.get("filter_date")
    if filter_date:
        parsed = pd.to_datetime(filter_date, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%a, %b %d, %Y")
    if frame.empty or "commence_time" not in frame.columns:
        return "N/A"
    ts = pd.to_datetime(frame["commence_time"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return "N/A"
    return ts.min().tz_convert("America/New_York").strftime("%a, %b %d, %Y")


def _generated_at_text(payload: dict[str, Any]) -> str:
    raw = payload.get("generated_at")
    if not raw:
        return "N/A"
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return "N/A"
    return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")


def _render_combined_card_image(
    payload: dict[str, Any],
    ml_rows: list[dict[str, Any]],
    ou_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not ml_rows:
        ml_rows = [{"Game Time (ET)": "—", "Away": "—", "Home": "—", "Mkt": "—", "Fair": "—", "Pick": "—", "Edge": "—", "Confidence": "—", "__edge_val": None, "__conf_val": None, "__pick_type": "moneyline"}]
    if not ou_rows:
        ou_rows = [{"Game Time (ET)": "—", "Away": "—", "Home": "—", "Line": "—", "Proj": "—", "Pick": "—", "Edge": "—", "Confidence": "—", "__edge_val": None, "__conf_val": None, "__pick_type": "total"}]

    max_rows_per_table = 14
    ml_omitted = max(len(ml_rows) - max_rows_per_table, 0)
    ou_omitted = max(len(ou_rows) - max_rows_per_table, 0)
    ml_rows = ml_rows[:max_rows_per_table]
    ou_rows = ou_rows[:max_rows_per_table]

    widths = {
        "Game Time (ET)": 140,
        "Away": 200,
        "Home": 200,
        "Mkt": 68,
        "Fair": 68,
        "Line": 68,
        "Proj": 68,
        "Pick": 150,
        "Edge": 92,
        "Confidence": 110,
    }
    headers_ml = ["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"]
    headers_ou = ["Game Time (ET)", "Away", "Home", "Line", "Proj", "Pick", "Edge", "Confidence"]

    row_h = 34
    section_title_h = 26
    table_gap = 16
    top_pad = 18
    left_pad = 18
    bottom_pad = 18
    table_w = sum(widths[h] for h in headers_ml)
    total_w = left_pad * 2 + table_w

    header_block_h = 82
    ml_block_h = section_title_h + row_h * (1 + len(ml_rows)) + (18 if ml_omitted else 0)
    ou_block_h = section_title_h + row_h * (1 + len(ou_rows)) + (18 if ou_omitted else 0)
    total_h = top_pad + header_block_h + ml_block_h + table_gap + ou_block_h + bottom_pad

    img = Image.new("RGB", (total_w, total_h), (15, 21, 33))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    perf = payload.get("performance", {}) if isinstance(payload, dict) else {}
    moneyline_perf = perf.get("moneyline", {}) if isinstance(perf, dict) else {}
    totals_perf = perf.get("totals", {}) if isinstance(perf, dict) else {}
    slate_date = _slate_date_text(pd.DataFrame(payload.get("games", []) or []), payload)
    updated = _generated_at_text(payload)

    title_text = "NBA Picks"
    title_w = int(draw.textlength(title_text, font=font_title))
    draw.text(((total_w - title_w) // 2, top_pad), title_text, fill=(241, 245, 255), font=font_title)

    stats_line_1 = (
        "ML Yesterday "
        + _record_text(moneyline_perf, "yesterday_record")
        + "   |   ML YTD "
        + _record_text(moneyline_perf, "ytd_record")
    )
    stats_line_2 = (
        "Totals Yesterday "
        + _record_text(totals_perf, "yesterday_record")
        + "   |   Totals YTD "
        + _record_text(totals_perf, "ytd_record")
    )
    meta_line = f"Slate Date: {slate_date}   |   Last updated: {updated}"

    for idx, line in enumerate([stats_line_1, stats_line_2, meta_line]):
        font_for_line = font_bold if idx < 2 else font_small
        color = (213, 222, 243) if idx < 2 else (156, 169, 197)
        line_w = int(draw.textlength(line, font=font_for_line))
        y_line = top_pad + 38 + idx * 18
        draw.text(((total_w - line_w) // 2, y_line), line, fill=color, font=font_for_line)

    y = top_pad + header_block_h
    draw.text((left_pad, y), "Moneyline Picks", fill=(236, 241, 252), font=font_bold)
    y += section_title_h
    y = _draw_table(draw, left_pad, y, headers_ml, ml_rows, widths, row_h, font_small, font_small)
    if ml_omitted:
        draw.text((left_pad, y + 4), f"...and {ml_omitted} more game(s).", fill=(173, 186, 214), font=font_small)
        y += 18

    y += table_gap
    draw.text((left_pad, y), "Over/Under Picks", fill=(236, 241, 252), font=font_bold)
    y += section_title_h
    y = _draw_table(draw, left_pad, y, headers_ou, ou_rows, widths, row_h, font_small, font_small)
    if ou_omitted:
        draw.text((left_pad, y + 4), f"...and {ou_omitted} more game(s).", fill=(173, 186, 214), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def _record_text(perf: dict[str, Any], key: str) -> str:
    record = str(perf.get(key, "0-0"))
    pct_key = "yesterday_pct" if key == "yesterday_record" else "ytd_pct"
    pct_val = perf.get(pct_key, 0.0)
    try:
        pct = float(pct_val)
        if pct <= 1.0:
            pct *= 100.0
    except Exception:
        pct = 0.0
    return f"{record} ({pct:.1f}%)"


def build_discord_payload(predictions_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "🏀 NBA picks",
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
    frame = pd.DataFrame(payload.get("games", []) or [])
    ml_rows = _moneyline_table_data(frame)
    ou_rows = _totals_table_data(frame)

    if os.getenv("DISCORD_DRY_RUN", "").strip() == "1":
        print(json.dumps(discord_payload, indent=2))
        print(f"moneyline_rows={len(ml_rows)} totals_rows={len(ou_rows)}")
        return

    with tempfile.TemporaryDirectory(prefix="discord-picks-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        card_img = tmp_path / "nba-picks-card.png"
        _render_combined_card_image(payload, ml_rows, ou_rows, card_img)

        payload_json = {
            "content": str(discord_payload.get("content", "🏀 NBA picks")),
            "embeds": [
                {
                    "title": "NBA Picks",
                    "color": 3447003,
                    "image": {"url": "attachment://nba-picks-card.png"},
                }
            ],
        }

        with card_img.open("rb") as card_file:
            upload = requests.post(
                webhook,
                data={"payload_json": json.dumps(payload_json)},
                files={
                    "file0": ("nba-picks-card.png", card_file, "image/png"),
                },
                timeout=30,
            )
        if upload.status_code >= 400:
            raise RuntimeError(f"Discord webhook failed ({upload.status_code}): {upload.text}")

    print("Posted picks to Discord with one message and embedded table images.")


if __name__ == "__main__":
    main()
