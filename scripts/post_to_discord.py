from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont


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
        rows.append(
            {
                "Game Time (ET)": _to_et(row.get("commence_time")),
                "Away": row.get("away_team"),
                "Home": row.get("home_team"),
                "Mkt": _fmt_american(row.get("mkt")),
                "Fair": row.get("fair"),
                "Pick": row.get("pick_team"),
                "Edge": _fmt_edge(row.get("best_moneyline_edge_pct")),
                "Confidence": _fmt_pct(row.get("conf_prob")),
                "__edge_val": row.get("best_moneyline_edge_pct"),
                "__conf_val": row.get("conf_prob"),
            }
        )
    return rows


def _totals_table_data(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    tot = frame.copy()
    tot["edge_raw"] = tot.apply(
        lambda r: r.get("over_edge_pct") if str(r.get("total_pick", "")) == "Over" else r.get("under_edge_pct"),
        axis=1,
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
                "Line": _fmt_float(row.get("total_line"), 1),
                "Proj": _fmt_float(row.get("model_projected_total"), 1),
                "Pick": row.get("total_pick"),
                "Edge": _fmt_edge(row.get("edge_raw")),
                "Confidence": _fmt_pct(row.get("conf_prob")),
                "__edge_val": row.get("edge_raw"),
                "__conf_val": row.get("conf_prob"),
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


def _pick_bg(value: Any) -> tuple[int, int, int]:
    text = str(value or "").upper()
    if text == "OVER":
        return (15, 143, 111)
    if text == "UNDER":
        return (91, 31, 47)
    return (30, 95, 180)


def _render_table_image(
    title: str,
    headers: list[str],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        rows = [{h: "—" for h in headers} | {"__edge_val": None, "__conf_val": None}]

    max_rows = 22
    omitted = 0
    if len(rows) > max_rows:
        omitted = len(rows) - max_rows
        rows = rows[:max_rows]

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
    row_h = 34
    title_h = 42
    top_pad = 14
    left_pad = 16
    table_w = sum(widths[h] for h in headers)
    total_w = left_pad * 2 + table_w
    total_h = top_pad + title_h + row_h * (len(rows) + 1) + (34 if omitted else 14)

    img = Image.new("RGB", (total_w, total_h), (15, 21, 33))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((left_pad, top_pad), title, fill=(232, 238, 252), font=font_bold)
    y = top_pad + title_h
    x = left_pad

    # Header row
    cx = x
    for h in headers:
        w = widths[h]
        draw.rectangle([cx, y, cx + w, y + row_h], fill=(30, 36, 49), outline=(48, 58, 78))
        draw.text((cx + 8, y + 8), h, fill=(205, 216, 238), font=font_small)
        cx += w
    y += row_h

    for idx, row in enumerate(rows):
        base_bg = (18, 24, 37) if idx % 2 == 0 else (21, 28, 43)
        cx = x
        for h in headers:
            w = widths[h]
            bg = base_bg
            if h == "Pick":
                bg = _pick_bg(row.get(h))
            elif h == "Edge":
                bg = _edge_bg(row.get("__edge_val"))
            elif h == "Confidence":
                bg = _conf_bg(row.get("__conf_val"))
            draw.rectangle([cx, y, cx + w, y + row_h], fill=bg, outline=(48, 58, 78))
            text = str(row.get(h, "—"))
            draw.text((cx + 8, y + 8), text, fill=(232, 238, 252), font=font_small)
            cx += w
        y += row_h

    if omitted:
        draw.text((left_pad, y + 8), f"...and {omitted} more game(s).", fill=(173, 186, 214), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")


def build_discord_payload() -> dict[str, Any]:
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
    discord_payload = build_discord_payload()
    frame = pd.DataFrame(payload.get("games", []) or [])
    ml_rows = _moneyline_table_data(frame)
    ou_rows = _totals_table_data(frame)

    if os.getenv("DISCORD_DRY_RUN", "").strip() == "1":
        print(json.dumps(discord_payload, indent=2))
        print(f"moneyline_rows={len(ml_rows)} totals_rows={len(ou_rows)}")
        return

    response = requests.post(webhook, json=discord_payload, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"Discord webhook failed ({response.status_code}): {response.text}")

    with tempfile.TemporaryDirectory(prefix="discord-picks-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        ml_img = tmp_path / "moneyline-picks.png"
        ou_img = tmp_path / "totals-picks.png"
        _render_table_image(
            "Moneyline Picks",
            ["Game Time (ET)", "Away", "Home", "Mkt", "Fair", "Pick", "Edge", "Confidence"],
            ml_rows,
            ml_img,
        )
        _render_table_image(
            "Over/Under Picks",
            ["Game Time (ET)", "Away", "Home", "Line", "Proj", "Pick", "Edge", "Confidence"],
            ou_rows,
            ou_img,
        )

        for title, img_path, attachment_name in [
            ("Moneyline Picks", ml_img, "moneyline-picks.png"),
            ("Over/Under Picks", ou_img, "totals-picks.png"),
        ]:
            with img_path.open("rb") as file_obj:
                payload_json = {
                    "embeds": [
                        {
                            "title": title,
                            "color": 3447003,
                            "image": {"url": f"attachment://{attachment_name}"},
                        }
                    ]
                }
                upload = requests.post(
                    webhook,
                    data={"payload_json": json.dumps(payload_json)},
                    files={"file": (attachment_name, file_obj, "image/png")},
                    timeout=30,
                )
                if upload.status_code >= 400:
                    raise RuntimeError(f"Discord image post failed ({upload.status_code}): {upload.text}")

    print("Posted picks to Discord with embedded table images.")


if __name__ == "__main__":
    main()
