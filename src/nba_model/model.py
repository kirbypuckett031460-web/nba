from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .odds import american_to_implied_prob
from .stats import (
    add_elo_features,
    build_team_form,
    fetch_game_logs,
    fetch_team_season_snapshot,
    normalize_team_name,
    team_lookup,
    to_game_level,
)


ID_COLUMNS = {
    "GAME_ID",
    "GAME_DATE",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "season",
}


@dataclass
class TrainingBundle:
    classifier: Pipeline
    total_regressor: Pipeline
    feature_columns: list[str]
    residual_total_std: float
    team_state: pd.DataFrame
    metrics: dict[str, float]
    generated_at: str

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @staticmethod
    def load(path: Path) -> "TrainingBundle":
        return joblib.load(path)


def _season_stats_for_training(seasons: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frame = fetch_team_season_snapshot(season)
            frame["season"] = season
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        raise ValueError("Unable to fetch any team season snapshots for model training.")
    return pd.concat(frames, ignore_index=True)


def _merge_game_with_stats(games: pd.DataFrame, seasonal_stats: pd.DataFrame) -> pd.DataFrame:
    home = seasonal_stats.add_prefix("home_").rename(columns={"home_TEAM_ID": "home_team_id", "home_season": "season"})
    away = seasonal_stats.add_prefix("away_").rename(columns={"away_TEAM_ID": "away_team_id", "away_season": "season"})

    merged = games.merge(home, on=["home_team_id", "season"], how="left")
    merged = merged.merge(away, on=["away_team_id", "season"], how="left")
    return merged


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in ID_COLUMNS or col in {"home_points", "away_points", "home_win", "total_points"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return sorted(cols)


def _build_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    skip_home_cols = {"home_points", "home_team_id"}
    home_metric_cols = [
        c
        for c in df.columns
        if c.startswith("home_") and c not in skip_home_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    for home_col in home_metric_cols:
        away_col = home_col.replace("home_", "away_", 1)
        if away_col in {"away_points", "away_team_id"}:
            continue
        if away_col in df.columns and pd.api.types.is_numeric_dtype(df[away_col]):
            diff_col = home_col.replace("home_", "diff_", 1)
            df[diff_col] = df[home_col] - df[away_col]
    return df


def _build_team_state(
    team_form: pd.DataFrame,
    seasonal_stats: pd.DataFrame,
    latest_elos: dict[int, float],
    current_season: str,
) -> pd.DataFrame:
    latest_form = (
        team_form.sort_values("GAME_DATE")
        .groupby("TEAM_ID", as_index=False)
        .tail(1)
        .copy()
    )
    latest_form = latest_form[
        [
            "TEAM_ID",
            "TEAM_NAME",
            "GAME_DATE",
            "rolling_pts_for",
            "rolling_pts_against",
            "rolling_margin",
            "rolling_possessions",
            "days_rest",
            "is_b2b",
        ]
    ]

    season_slice = seasonal_stats[seasonal_stats["season"] == current_season].copy()
    state = season_slice.merge(latest_form, on=["TEAM_ID", "TEAM_NAME"], how="left")
    state["elo_pre"] = state["TEAM_ID"].map(latest_elos).fillna(1500.0)
    state.rename(columns={"GAME_DATE": "latest_game_date", "is_b2b": "b2b"}, inplace=True)
    return state


def build_training_data(seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = fetch_game_logs(seasons)
    available_seasons = sorted(logs["SEASON"].dropna().astype(str).unique().tolist())
    if not available_seasons:
        raise ValueError("No valid seasons available from game logs.")
    form = build_team_form(logs, lookback_games=10)
    games = to_game_level(form)
    elo_result = add_elo_features(games)
    games = elo_result.game_frame

    seasonal_stats = _season_stats_for_training(available_seasons)
    merged = _merge_game_with_stats(games, seasonal_stats)
    merged = _build_differential_features(merged)
    merged = merged.sort_values("GAME_DATE").reset_index(drop=True)

    team_state = _build_team_state(
        team_form=form,
        seasonal_stats=seasonal_stats,
        latest_elos=elo_result.latest_elos,
        current_season=available_seasons[-1],
    )
    return merged, team_state


def _build_numeric_pipeline(model: Any, numeric_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric, numeric_features)],
        remainder="drop",
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def train_models(seasons: list[str]) -> TrainingBundle:
    frame, team_state = build_training_data(seasons)
    frame = frame.dropna(subset=["home_win", "total_points"]).reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError("Not enough training data returned from NBA stats API to build stable models.")

    feature_columns = _numeric_feature_columns(frame)
    split_idx = max(int(len(frame) * 0.8), 100)
    train_df = frame.iloc[:split_idx].copy()
    test_df = frame.iloc[split_idx:].copy()
    if test_df.empty:
        test_df = train_df.tail(min(100, len(train_df))).copy()

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train_cls = train_df["home_win"].astype(int)
    y_test_cls = test_df["home_win"].astype(int)
    y_train_tot = train_df["total_points"].astype(float)
    y_test_tot = test_df["total_points"].astype(float)

    cls_model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=350,
        min_samples_leaf=20,
        random_state=42,
    )
    reg_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=350,
        min_samples_leaf=20,
        random_state=42,
    )

    cls_pipeline = _build_numeric_pipeline(cls_model, feature_columns)
    reg_pipeline = _build_numeric_pipeline(reg_model, feature_columns)

    cls_pipeline.fit(X_train, y_train_cls)
    reg_pipeline.fit(X_train, y_train_tot)

    cls_probs = cls_pipeline.predict_proba(X_test)[:, 1]
    reg_preds = reg_pipeline.predict(X_test)
    residuals = y_test_tot - reg_preds
    residual_std = float(np.nanstd(residuals))
    if not np.isfinite(residual_std) or residual_std < 6.0:
        residual_std = 11.5

    moneyline_auc = float("nan")
    moneyline_logloss = float("nan")
    if len(np.unique(y_test_cls)) == 2:
        moneyline_auc = float(roc_auc_score(y_test_cls, cls_probs))
        moneyline_logloss = float(log_loss(y_test_cls, cls_probs))

    metrics = {
        "moneyline_auc": moneyline_auc,
        "moneyline_log_loss": moneyline_logloss,
        "total_mae": float(mean_absolute_error(y_test_tot, reg_preds)),
        "sample_games": float(len(frame)),
    }

    return TrainingBundle(
        classifier=cls_pipeline,
        total_regressor=reg_pipeline,
        feature_columns=feature_columns,
        residual_total_std=residual_std,
        team_state=team_state,
        metrics=metrics,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _team_state_row(team_state: pd.DataFrame, team_name: str) -> pd.Series | None:
    lookup = team_lookup()
    key = normalize_team_name(team_name)
    team = lookup.get(key)
    if not team:
        return None

    row = team_state[team_state["TEAM_ID"] == team["id"]]
    if row.empty:
        return None
    return row.iloc[0]


def _matchup_feature_row(
    home_team: str,
    away_team: str,
    commence_time: pd.Timestamp,
    team_state: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    home = _team_state_row(team_state, home_team)
    away = _team_state_row(team_state, away_team)
    if home is None or away is None:
        return {col: np.nan for col in feature_columns}

    numeric_state_cols = [
        c
        for c in team_state.columns
        if c not in {"TEAM_ID", "TEAM_NAME", "latest_game_date", "season"} and pd.api.types.is_numeric_dtype(team_state[c])
    ]
    feature_row: dict[str, float] = {}
    for col in numeric_state_cols:
        feature_row[f"home_{col}"] = float(home[col]) if pd.notna(home[col]) else np.nan
        feature_row[f"away_{col}"] = float(away[col]) if pd.notna(away[col]) else np.nan
        feature_row[f"diff_{col}"] = feature_row[f"home_{col}"] - feature_row[f"away_{col}"]

    home_last = pd.to_datetime(home.get("latest_game_date"), utc=True, errors="coerce")
    away_last = pd.to_datetime(away.get("latest_game_date"), utc=True, errors="coerce")
    if pd.notna(home_last):
        feature_row["home_days_rest"] = float(max((commence_time - home_last).days, 0))
    if pd.notna(away_last):
        feature_row["away_days_rest"] = float(max((commence_time - away_last).days, 0))
    if "home_days_rest" in feature_row and "away_days_rest" in feature_row:
        feature_row["diff_days_rest"] = feature_row["home_days_rest"] - feature_row["away_days_rest"]
    if "home_elo_pre" in feature_row and "away_elo_pre" in feature_row:
        feature_row["elo_diff"] = feature_row["home_elo_pre"] - feature_row["away_elo_pre"]

    return {col: feature_row.get(col, np.nan) for col in feature_columns}


def predict_from_odds(bundle: TrainingBundle, odds_df: pd.DataFrame, min_edge_pct: float) -> pd.DataFrame:
    if odds_df.empty:
        return odds_df.copy()

    rows: list[dict[str, Any]] = []
    for _, game in odds_df.iterrows():
        commence_time = pd.to_datetime(game["commence_time"], utc=True)
        feature_row = _matchup_feature_row(
            home_team=game["home_team"],
            away_team=game["away_team"],
            commence_time=commence_time,
            team_state=bundle.team_state,
            feature_columns=bundle.feature_columns,
        )
        features_df = pd.DataFrame([feature_row])

        home_win_prob = float(bundle.classifier.predict_proba(features_df)[0, 1])
        projected_total = float(bundle.total_regressor.predict(features_df)[0])
        over_line = game.get("total_line")
        over_prob = np.nan
        under_prob = np.nan
        if pd.notna(over_line):
            over_prob = float(1.0 - norm.cdf(float(over_line), loc=projected_total, scale=bundle.residual_total_std))
            under_prob = float(1.0 - over_prob)

        home_market = american_to_implied_prob(game.get("home_moneyline"))
        away_market = american_to_implied_prob(game.get("away_moneyline"))
        over_market = american_to_implied_prob(game.get("over_price"))
        under_market = american_to_implied_prob(game.get("under_price"))

        home_edge = (home_win_prob - home_market) * 100.0 if home_market is not None else np.nan
        away_win_prob = 1.0 - home_win_prob
        away_edge = (away_win_prob - away_market) * 100.0 if away_market is not None else np.nan
        over_edge = (over_prob - over_market) * 100.0 if over_market is not None and pd.notna(over_prob) else np.nan
        under_edge = (under_prob - under_market) * 100.0 if under_market is not None and pd.notna(under_prob) else np.nan

        moneyline_pick = "Pass"
        best_moneyline_edge = np.nanmax([home_edge, away_edge]) if not (np.isnan(home_edge) and np.isnan(away_edge)) else np.nan
        if np.isfinite(home_edge) and home_edge >= min_edge_pct and home_edge >= away_edge:
            moneyline_pick = f"{game['home_team']} ML"
        elif np.isfinite(away_edge) and away_edge >= min_edge_pct:
            moneyline_pick = f"{game['away_team']} ML"

        total_pick = "Pass"
        if np.isfinite(over_edge) and over_edge >= min_edge_pct and over_edge >= under_edge:
            total_pick = "Over"
        elif np.isfinite(under_edge) and under_edge >= min_edge_pct:
            total_pick = "Under"

        rows.append(
            {
                **game.to_dict(),
                "model_home_win_prob": home_win_prob,
                "model_away_win_prob": away_win_prob,
                "model_projected_total": projected_total,
                "model_over_prob": over_prob,
                "model_under_prob": under_prob,
                "home_edge_pct": home_edge,
                "away_edge_pct": away_edge,
                "over_edge_pct": over_edge,
                "under_edge_pct": under_edge,
                "best_moneyline_edge_pct": best_moneyline_edge,
                "moneyline_pick": moneyline_pick,
                "total_pick": total_pick,
            }
        )

    out = pd.DataFrame(rows).sort_values("commence_time").reset_index(drop=True)
    return out
