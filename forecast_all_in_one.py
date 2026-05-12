"""Прогноз распределения плана отгрузки по направлениям - всё в одном файле.

Структура:
  1. Импорты и константы
  2. Config (параметры пайплайна)
  3. ВХОДНЫЕ ДАННЫЕ (df + plan_fact)         <-- сюда подставь свои датафреймы
  4. Препроцессинг (нормализация колонок)
  5. Панель и инженерия признаков
  6. Метрики (WAPE / MAPE / MAPE_TOP / coverage) и сборка отчёта
  7. Forecaster'ы: naive_last / weighted / linreg / hurdle_topk + shrinkage
  8. Бэктест и тюнинг
  9. Пайплайн (determine_months, run_pipeline)
 10. CLI и точка входа

Ожидаемые колонки:
  df (история):  origin, destination, cargo_type, speed, date, volume
  plan_fact:     origin, date, plan_volume
В обоих DataFrame'ах допустимы русские названия из словарей
_FACTS_RENAME / _PLANS_RENAME - они нормализуются автоматически.
"""

from __future__ import annotations

# =============================================================================
# 1. Импорты
# =============================================================================
import argparse
import itertools
import logging
import sys
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LinearRegression


LANE_COLS: list[str] = ["origin", "destination", "cargo_type", "speed"]
OTHER_LABEL: str = "прочее"

logger = logging.getLogger("lane_forecast")


# =============================================================================
# 2. Config
# =============================================================================
@dataclass
class Config:
    """Все параметры пайплайна в одном месте."""

    # ---- CatBoost ----
    recency_halflife_months: float = 6.0
    hurdle_iterations: int = 300
    hurdle_depth: int = 6
    hurdle_lr: float = 0.05
    volume_iterations: int = 300
    volume_depth: int = 6
    volume_lr: float = 0.05
    min_rows_for_volume_model: int = 50
    random_state: int = 0

    # ---- hurdle_topk ----
    use_other_bucket: bool = True
    coverage_target: float = 0.92
    min_p_active: float = 0.05
    ml_weight: float = 0.6

    # ---- shrinkage ----
    shrink_alpha: float = 0.55
    shrink_beta: float = 0.25
    shrink_gamma: float = 0.15
    shrink_delta: float = 0.05

    # ---- бэктест ----
    n_validation_months: int = 5
    holdout: bool = True

    # ---- forecast mode ----
    target_month: Optional[str] = None
    forecast_model: Optional[str] = None

    # ---- tuning ----
    tune: bool = False
    tune_grid_coverage: tuple[float, ...] = (0.90, 0.95)
    tune_grid_ml_weight: tuple[float, ...] = (0.5, 0.7)
    tune_grid_min_p_active: tuple[float, ...] = (0.05,)

    # ---- вывод ----
    output_dir: Path = field(default_factory=lambda: Path("forecast_output"))


# =============================================================================
# 3. ВХОДНЫЕ ДАННЫЕ
#    Сюда подставь свои DataFrame'ы. Они нормализуются автоматически.
#
#    df         - история отгрузок:  origin, destination, cargo_type, speed,
#                                    date, volume
#    plan_fact  - план:              origin, date, plan_volume
#
#    Пример (раскомментируй и адаптируй под себя):
#
#    import psycopg2
#    with psycopg2.connect(...) as conn:
#        df        = pd.read_sql('SELECT * )
#        plan_fact = pd.read_sql('SELECT * )
# =============================================================================
df: Optional[pd.DataFrame] = None         # <-- замени на свой DataFrame
plan_fact: Optional[pd.DataFrame] = None  # <-- замени на свой DataFrame


# =============================================================================
# 4. Препроцессинг
# =============================================================================
_FACTS_RENAME: dict[str, str] = {
    "Станция отправления": "origin",
    "Станцияотправления": "origin",
    "Направление": "destination",
    "Груз": "cargo_type",
    "Тип груза": "cargo_type",
    "спо": "speed",
    "СПО": "speed",
    "СПФ": "speed",
    "Объем факт": "volume",
    "ОбъемФакт": "volume",
    "Объем": "volume",
    "Date": "date",
    "Дата": "date",
}

_PLANS_RENAME: dict[str, str] = {
    "Станцияотправления": "origin",
    "Станция отправления": "origin",
    "Объем план": "plan_volume",
    "ОбъемПлан": "plan_volume",
    "volume": "plan_volume",
    "Дата календаря": "date",
    "ДатаКалендаря": "date",
}


def preprocess_facts(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Нормализуем колонки истории отгрузок."""
    out = df_raw.rename(columns=_FACTS_RENAME).copy()
    required = {"origin", "destination", "cargo_type", "speed", "date", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"В df нет колонок: {sorted(missing)}. "
            "Добавь алиасы в _FACTS_RENAME или переименуй заранее."
        )
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.to_period("M")
    for col in LANE_COLS:
        out[col] = out[col].astype(str)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    return out[LANE_COLS + ["date", "month", "volume"]]


def preprocess_plans(plan_raw: pd.DataFrame) -> pd.DataFrame:
    """Нормализуем колонки плана. Возвращает агрегат по (origin, month)."""
    plan = plan_raw.rename(columns=_PLANS_RENAME).copy()
    required = {"origin", "plan_volume", "date"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(
            f"В plan_fact нет колонок: {sorted(missing)}. "
            "Добавь алиасы в _PLANS_RENAME или переименуй заранее."
        )
    plan["date"] = pd.to_datetime(plan["date"])
    plan["month"] = plan["date"].dt.to_period("M")
    plan["origin"] = plan["origin"].astype(str)
    plan["plan_volume"] = pd.to_numeric(
        plan["plan_volume"], errors="coerce"
    ).fillna(0.0)
    plans = plan.groupby(["origin", "month"], as_index=False)["plan_volume"].sum()
    return plans[plans["plan_volume"] > 0].reset_index(drop=True)


# =============================================================================
# 5. Панель и признаки
# =============================================================================
NUMERIC_FEATURES: list[str] = [
    "vol_lag_1", "vol_lag_2", "vol_lag_3", "vol_lag_6", "vol_lag_12",
    "act_lag_1", "act_lag_2", "act_lag_3", "act_lag_6", "act_lag_12",
    "share_lag_1", "share_lag_2", "share_lag_3",
    "vol_roll_mean_3", "vol_roll_mean_6",
    "share_roll_mean_3", "share_roll_std_3",
    "act_count_3", "act_count_6", "act_count_12",
    "mean_vol_when_active", "cv_vol_6m", "growth_1_3",
    "origin_total_lag_1", "origin_total_lag_3_mean", "origin_n_active_lag_1",
    "months_since_last", "sin_m", "cos_m",
    "origin_plan_next",
]
CAT_FEATURES: list[str] = list(LANE_COLS)
ALL_FEATURES: list[str] = NUMERIC_FEATURES + CAT_FEATURES


def build_panel(facts: pd.DataFrame, up_to_month: pd.Period) -> pd.DataFrame:
    """Полная сетка lane x month с нулями + флагом active."""
    history = facts[facts["month"] <= up_to_month].copy()
    if history.empty:
        return history

    lanes = history[LANE_COLS].drop_duplicates()
    months = pd.period_range(history["month"].min(), up_to_month, freq="M")

    lanes["_k"] = 1
    months_df = pd.DataFrame({"month": months, "_k": 1})
    grid = lanes.merge(months_df, on="_k").drop(columns="_k")

    actual = history.groupby(LANE_COLS + ["month"], as_index=False)["volume"].sum()
    panel = grid.merge(actual, on=LANE_COLS + ["month"], how="left")
    panel["volume"] = panel["volume"].fillna(0.0)
    panel["active"] = (panel["volume"] > 0).astype(int)
    return panel.sort_values(LANE_COLS + ["month"]).reset_index(drop=True)


def recency_weights(months: pd.Series, halflife_months: float) -> np.ndarray:
    """Экспоненциально затухающие веса по свежести месяца."""
    unique_months = sorted(months.unique())
    age_map = {m: i for i, m in enumerate(unique_months)}
    age = months.map(age_map).to_numpy(dtype=float)
    max_age = age.max() if len(age) else 0.0
    return np.power(0.5, (max_age - age) / halflife_months)


def normalize_to_plan(
    scores: pd.DataFrame, plan_by_origin: dict[str, float],
) -> pd.DataFrame:
    """Внутри каждого origin нормализуем score и умножаем на план origin."""
    chunks: list[pd.DataFrame] = []
    for origin, total in plan_by_origin.items():
        sub = scores[scores["origin"] == origin].copy()
        if sub.empty or total <= 0:
            continue
        s = sub["score"].sum()
        if s <= 0:
            sub["forecast_volume"] = total / len(sub)
        else:
            sub["forecast_volume"] = sub["score"] / s * total
        chunks.append(sub[LANE_COLS + ["forecast_volume"]])
    if not chunks:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])
    return pd.concat(chunks, ignore_index=True)


def _months_since_last_active(s: pd.Series) -> pd.Series:
    out = np.full(len(s), np.nan)
    last = -1
    for i, a in enumerate(s.to_numpy()):
        if last >= 0:
            out[i] = i - last
        if a > 0:
            last = i
    return pd.Series(out, index=s.index)


def make_features(panel: pd.DataFrame, plans: pd.DataFrame) -> pd.DataFrame:
    """Признаки на основе панели + плана.

    План t+1 (известный в момент t) добавляется как фича origin_plan_next.
    """
    p = panel.sort_values(LANE_COLS + ["month"]).reset_index(drop=True).copy()
    for c in CAT_FEATURES:
        p[c] = p[c].astype(str)

    origin_total_per_month = p.groupby(["origin", "month"])["volume"].transform("sum")
    p["lane_share"] = np.where(
        origin_total_per_month > 0, p["volume"] / origin_total_per_month, 0.0
    )

    g = p.groupby(LANE_COLS, sort=False)

    for lag in (1, 2, 3, 6, 12):
        p[f"vol_lag_{lag}"] = g["volume"].shift(lag)
        p[f"act_lag_{lag}"] = g["active"].shift(lag)
    for lag in (1, 2, 3):
        p[f"share_lag_{lag}"] = g["lane_share"].shift(lag)

    p["vol_roll_mean_3"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    p["vol_roll_mean_6"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(6).mean()
    )
    p["share_roll_mean_3"] = g["lane_share"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    p["share_roll_std_3"] = g["lane_share"].transform(
        lambda s: s.shift(1).rolling(3).std()
    )
    p["act_count_3"] = g["active"].transform(lambda s: s.shift(1).rolling(3).sum())
    p["act_count_6"] = g["active"].transform(lambda s: s.shift(1).rolling(6).sum())
    p["act_count_12"] = g["active"].transform(lambda s: s.shift(1).rolling(12).sum())

    p["mean_vol_when_active"] = g["volume"].transform(
        lambda s: s.where(s > 0).shift(1).expanding().mean()
    )
    roll_mean_6 = g["volume"].transform(lambda s: s.shift(1).rolling(6).mean())
    roll_std_6 = g["volume"].transform(lambda s: s.shift(1).rolling(6).std())
    p["cv_vol_6m"] = roll_std_6 / roll_mean_6.replace(0, np.nan)
    p["growth_1_3"] = p["vol_lag_1"] / p["vol_roll_mean_3"].replace(0, np.nan)
    p["months_since_last"] = g["active"].transform(_months_since_last_active)

    # ---- origin-level признаки ----
    origin_month = (
        p.groupby(["origin", "month"])
        .agg(
            origin_total_vol=("volume", "sum"),
            origin_n_active=("active", "sum"),
        )
        .reset_index()
        .sort_values(["origin", "month"])
    )
    og = origin_month.groupby("origin")
    origin_month["origin_total_lag_1"] = og["origin_total_vol"].shift(1)
    origin_month["origin_total_lag_3_mean"] = og["origin_total_vol"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    origin_month["origin_n_active_lag_1"] = og["origin_n_active"].shift(1)
    p = p.merge(
        origin_month[
            [
                "origin", "month",
                "origin_total_lag_1",
                "origin_total_lag_3_mean",
                "origin_n_active_lag_1",
            ]
        ],
        on=["origin", "month"],
        how="left",
    )

    # ---- план на t+1 как фича ----
    if plans is not None and not plans.empty:
        plan_for_next = plans[["origin", "month", "plan_volume"]].copy()
        plan_for_next["origin"] = plan_for_next["origin"].astype(str)
        plan_for_next["month"] = plan_for_next["month"] - 1
        plan_for_next = plan_for_next.rename(
            columns={"plan_volume": "origin_plan_next"}
        )
        p = p.merge(plan_for_next, on=["origin", "month"], how="left")
    else:
        p["origin_plan_next"] = np.nan

    # ---- сезонность ----
    month_num = p["month"].dt.month.astype(int)
    p["sin_m"] = np.sin(2 * np.pi * month_num / 12)
    p["cos_m"] = np.cos(2 * np.pi * month_num / 12)

    return p


# =============================================================================
# 6. Метрики и отчёт
# =============================================================================
@dataclass
class Scores:
    """Метрики на одном месяце."""

    wape: float
    mape: float
    mape_top: float
    coverage: float


def _collapse_fact_to_pred(fact: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Если в pred есть 'прочее' для какого-то origin, фактические lane
    этого origin, которых нет в pred, схлопываются в тот же бакет."""
    fact = fact.copy()
    pred_keys = set(map(tuple, pred[LANE_COLS].values))
    fact_keys = list(map(tuple, fact[LANE_COLS].values))
    in_pred = np.array([k in pred_keys for k in fact_keys])

    origins_with_other = set(pred.loc[pred["destination"] == OTHER_LABEL, "origin"])
    has_other = fact["origin"].isin(origins_with_other).values

    mask = (~in_pred) & has_other
    if mask.any():
        fact.loc[mask, "destination"] = OTHER_LABEL
        fact.loc[mask, "cargo_type"] = OTHER_LABEL
        fact.loc[mask, "speed"] = OTHER_LABEL
        fact = fact.groupby(LANE_COLS, as_index=False)["volume"].sum()
    return fact


def evaluate(fact: pd.DataFrame, pred: pd.DataFrame) -> Scores:
    """Считает WAPE, MAPE, MAPE_TOP и coverage."""
    fact = _collapse_fact_to_pred(fact, pred)
    f = fact.groupby(LANE_COLS, as_index=False)["volume"].sum()
    p = pred.groupby(LANE_COLS, as_index=False)["forecast_volume"].sum()
    m = f.merge(p, on=LANE_COLS, how="outer").fillna(0.0)

    y = m["volume"].to_numpy()
    yhat = m["forecast_volume"].to_numpy()

    abs_err = np.abs(y - yhat)
    total = float(y.sum()) if y.sum() > 0 else 1.0
    wape = float(abs_err.sum() / total)

    mask = y > 0
    mape = float(np.mean(abs_err[mask] / y[mask])) if mask.any() else float("nan")

    sorted_idx = np.argsort(-y)
    cum = np.cumsum(y[sorted_idx]) / total
    top_n = int(np.searchsorted(cum, 0.80) + 1)
    top_idx = sorted_idx[:top_n]
    y_top = y[top_idx]
    yhat_top = yhat[top_idx]
    mask_top = y_top > 0
    mape_top = (
        float(np.mean(np.abs(y_top[mask_top] - yhat_top[mask_top]) / y_top[mask_top]))
        if mask_top.any()
        else float("nan")
    )

    covered = float(m.loc[m["forecast_volume"] > 0, "volume"].sum())
    coverage = float(covered / total)

    return Scores(wape=wape, mape=mape, mape_top=mape_top, coverage=coverage)


def build_forecast_report(
    facts: pd.DataFrame,
    target_month: pd.Period,
    forecast: pd.DataFrame,
    plan_origins: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Прогноз + факт + ошибка построчно."""
    fact = facts[facts["month"] == target_month]
    if plan_origins is not None:
        fact = fact[fact["origin"].isin(set(plan_origins))]

    if fact.empty:
        report = forecast.copy()
        report["fact_volume"] = np.nan
        report["error"] = np.nan
        report["abs_error"] = np.nan
        return report.sort_values(
            ["origin", "forecast_volume"], ascending=[True, False]
        )

    fact_agg = (
        fact[LANE_COLS + ["volume"]]
        .groupby(LANE_COLS, as_index=False)["volume"]
        .sum()
    )
    fact_collapsed = _collapse_fact_to_pred(fact_agg, forecast).rename(
        columns={"volume": "fact_volume"}
    )
    report = forecast.merge(fact_collapsed, on=LANE_COLS, how="outer").fillna(0.0)
    report["error"] = report["forecast_volume"] - report["fact_volume"]
    report["abs_error"] = report["error"].abs()
    return report.sort_values(
        ["origin", "forecast_volume"], ascending=[True, False]
    )


# =============================================================================
# 7. Forecaster'ы
# =============================================================================
ForecasterFn = Callable[
    [pd.DataFrame, pd.Period, dict[str, float], pd.DataFrame], pd.DataFrame
]


def naive_last_month(
    history: pd.DataFrame,
    target_month: pd.Period,
    plan_by_origin: dict[str, float],
    plans: pd.DataFrame,  # noqa: ARG001
) -> pd.DataFrame:
    """Прогноз = факт последнего месяца, отнормированный на план."""
    h = history[history["month"] < target_month]
    if h.empty:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])
    last_m = h["month"].max()
    last = (
        h[h["month"] == last_m]
        .groupby(LANE_COLS, as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "score"})
    )
    return normalize_to_plan(last, plan_by_origin)


def weighted_baseline(
    history: pd.DataFrame,
    target_month: pd.Period,
    plan_by_origin: dict[str, float],
    plans: pd.DataFrame,  # noqa: ARG001
    weights: tuple[float, ...] = (0.1, 0.1, 0.8),
) -> pd.DataFrame:
    """Текущая прод-схема: взвешенное среднее по последним 3 месяцам."""
    h = history[history["month"] < target_month]
    if h.empty:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])

    last3 = sorted(h["month"].unique())[-3:]
    w = np.array(weights[-len(last3):])
    w = w / w.sum()

    agg = (
        h[h["month"].isin(last3)]
        .groupby(LANE_COLS + ["month"], as_index=False)["volume"]
        .sum()
        .pivot_table(
            index=LANE_COLS, columns="month", values="volume", fill_value=0.0
        )
    )
    agg = agg.reindex(columns=last3, fill_value=0.0)
    agg["weighted"] = (agg.values * w).sum(axis=1)
    out = (
        agg.reset_index()[LANE_COLS + ["weighted"]]
        .rename(columns={"weighted": "score"})
    )
    return normalize_to_plan(out, plan_by_origin)


def linreg_baseline(
    history: pd.DataFrame,
    target_month: pd.Period,
    plan_by_origin: dict[str, float],
    plans: pd.DataFrame,  # noqa: ARG001
    r2_threshold: float = 0.3,
    err_threshold: float = 0.3,
) -> pd.DataFrame:
    """Линейная регрессия по тренду с фоллбэком на weighted 0.1/0.1/0.8."""
    h = history[history["month"] < target_month]
    if h.empty:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])

    rows: list[dict] = []
    for keys, g in h.groupby(LANE_COLS):
        g = g.sort_values("month")
        y = g["volume"].to_numpy()
        n = len(y)
        if n < 3:
            w = np.array([0.1, 0.1, 0.8][-n:])
            w = w / w.sum()
            pred = float((y * w).sum())
        else:
            X = np.arange(n).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            pred_last = float(model.predict([[n - 1]])[0])
            actual_last = float(y[-1])
            err = (
                0.0 if actual_last == 0
                else abs(pred_last - actual_last) / actual_last
            )
            if (r2 < r2_threshold) or (err > err_threshold):
                w = np.array([0.1, 0.1, 0.8])
                pred = float((y[-3:] * w).sum())
            else:
                pred = float(model.predict([[n]])[0])
                pred = max(pred, 0.0)
                pred = min(pred, float(y.mean() * 2))
        rows.append({**dict(zip(LANE_COLS, keys)), "score": pred})

    return normalize_to_plan(pd.DataFrame(rows), plan_by_origin)


def shrunk_volume(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Сглаженная оценка объёма lane как смесь:
       recent(lane) / mean(origin+destination) / mean(origin+cargo) / origin prior."""
    last3 = sorted(panel["month"].unique())[-3:]
    p3 = panel[panel["month"].isin(last3)]

    lane = (
        p3.groupby(LANE_COLS, as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": "vol_lane"})
    )
    od = (
        p3.groupby(["origin", "destination", "month"], as_index=False)["volume"]
        .sum()
        .groupby(["origin", "destination"], as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": "vol_od"})
    )
    oc = (
        p3.groupby(["origin", "cargo_type", "month"], as_index=False)["volume"]
        .sum()
        .groupby(["origin", "cargo_type"], as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": "vol_oc"})
    )

    n_lanes_per_origin = (
        p3.groupby("origin")[LANE_COLS]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_lanes")
        .reset_index()
    )
    origin_total = (
        p3.groupby(["origin", "month"], as_index=False)["volume"]
        .sum()
        .groupby("origin", as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": "origin_avg_total"})
    )
    origin_prior = origin_total.merge(n_lanes_per_origin, on="origin", how="left")
    origin_prior["vol_prior"] = (
        origin_prior["origin_avg_total"] / origin_prior["n_lanes"].clip(lower=1)
    )

    od_n = (
        p3.groupby(["origin", "destination"])[LANE_COLS]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_od")
        .reset_index()
    )
    oc_n = (
        p3.groupby(["origin", "cargo_type"])[LANE_COLS]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_oc")
        .reset_index()
    )

    out = lane.merge(od, on=["origin", "destination"], how="left")
    out = out.merge(oc, on=["origin", "cargo_type"], how="left")
    out = out.merge(origin_prior[["origin", "vol_prior"]], on="origin", how="left")
    out = out.merge(od_n, on=["origin", "destination"], how="left")
    out = out.merge(oc_n, on=["origin", "cargo_type"], how="left")

    out["vol_od_per_lane"] = out["vol_od"] / out["n_od"].clip(lower=1)
    out["vol_oc_per_lane"] = out["vol_oc"] / out["n_oc"].clip(lower=1)

    for col in ("vol_lane", "vol_od_per_lane", "vol_oc_per_lane", "vol_prior"):
        out[col] = out[col].fillna(0.0)

    out["shrunk_volume"] = (
        cfg.shrink_alpha * out["vol_lane"]
        + cfg.shrink_beta * out["vol_od_per_lane"]
        + cfg.shrink_gamma * out["vol_oc_per_lane"]
        + cfg.shrink_delta * out["vol_prior"]
    )
    return out[LANE_COLS + ["shrunk_volume"]]


def _train_hurdle_and_volume(
    feat: pd.DataFrame, cfg: Config,
) -> tuple[Optional[CatBoostClassifier], Optional[CatBoostRegressor]]:
    """Учим P(active_{t+1}) и log(volume_{t+1}) при условии активности."""
    feat = feat.copy()
    feat["target_active"] = feat.groupby(LANE_COLS)["active"].shift(-1)
    feat["target_volume"] = feat.groupby(LANE_COLS)["volume"].shift(-1)

    h_train = feat.dropna(subset=["target_active"])
    hurdle: Optional[CatBoostClassifier] = None
    if not h_train.empty and h_train["target_active"].nunique() >= 2:
        weights = recency_weights(h_train["month"], cfg.recency_halflife_months)
        hurdle = CatBoostClassifier(
            iterations=cfg.hurdle_iterations,
            depth=cfg.hurdle_depth,
            learning_rate=cfg.hurdle_lr,
            loss_function="Logloss",
            cat_features=CAT_FEATURES,
            random_seed=cfg.random_state,
            verbose=False,
            allow_writing_files=False,
        )
        hurdle.fit(
            h_train[ALL_FEATURES],
            h_train["target_active"].astype(int).to_numpy(),
            sample_weight=weights,
        )
    else:
        logger.warning("Hurdle: недостаточно данных или один класс - пропускаем.")

    v_train = feat[(feat["target_active"] == 1) & (feat["target_volume"] > 0)]
    vol_model: Optional[CatBoostRegressor] = None
    if len(v_train) >= cfg.min_rows_for_volume_model:
        weights = recency_weights(v_train["month"], cfg.recency_halflife_months)
        y = np.log1p(v_train["target_volume"].to_numpy())
        vol_model = CatBoostRegressor(
            iterations=cfg.volume_iterations,
            depth=cfg.volume_depth,
            learning_rate=cfg.volume_lr,
            loss_function="RMSE",
            cat_features=CAT_FEATURES,
            random_seed=cfg.random_state,
            verbose=False,
            allow_writing_files=False,
        )
        vol_model.fit(v_train[ALL_FEATURES], y, sample_weight=weights)
    else:
        logger.warning(
            "Volume model: <%d строк - пропускаем.", cfg.min_rows_for_volume_model,
        )

    return hurdle, vol_model


def _allocate_plan(
    sub: pd.DataFrame,
    origin: str,
    total_vol: float,
    *,
    use_other_bucket: bool,
    coverage_target: float,
) -> list[pd.DataFrame]:
    """Распределение плана origin между его lane.

    use_other_bucket=True: top-K покрывают coverage_target, остальное -> 'прочее'.
    use_other_bucket=False: все кандидаты получают долю плана.
    """
    out: list[pd.DataFrame] = []
    if sub.empty or total_vol <= 0:
        return out

    sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
    score_total = float(sub["score"].sum())

    if score_total <= 0:
        sub["forecast_volume"] = total_vol / len(sub)
        out.append(sub[LANE_COLS + ["forecast_volume"]])
        return out

    if not use_other_bucket:
        sub["forecast_volume"] = sub["score"] / score_total * total_vol
        out.append(sub[LANE_COLS + ["forecast_volume"]])
        return out

    sub["share_norm"] = sub["score"] / score_total
    sub["cum"] = sub["share_norm"].cumsum()
    keep = sub[sub["cum"] <= coverage_target].copy()
    if keep.empty:
        keep = sub.head(1).copy()

    kept_sum = float(keep["share_norm"].sum())
    if kept_sum <= 0:
        keep["share_final"] = 1.0 / len(keep)
    else:
        keep["share_final"] = keep["share_norm"] / kept_sum * coverage_target
    keep["forecast_volume"] = keep["share_final"] * total_vol
    out.append(keep[LANE_COLS + ["forecast_volume"]])

    if len(sub) > len(keep):
        tail_volume = (1.0 - coverage_target) * total_vol
        out.append(
            pd.DataFrame([{
                "origin": origin,
                "destination": OTHER_LABEL,
                "cargo_type": OTHER_LABEL,
                "speed": OTHER_LABEL,
                "forecast_volume": tail_volume,
            }])
        )
    return out


def hurdle_topk(
    history: pd.DataFrame,
    target_month: pd.Period,
    plan_by_origin: dict[str, float],
    plans: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Hurdle + CatBoost-объём + shrinkage + опциональный top-K с 'прочее'."""
    panel = build_panel(history, up_to_month=target_month - 1)
    if panel.empty:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])

    feat = make_features(panel, plans)
    hurdle, vol_model = _train_hurdle_and_volume(feat, cfg)

    last_month = panel["month"].max()
    cur = feat[feat["month"] == last_month].copy()
    if cur.empty:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])

    X_cur = cur[ALL_FEATURES]
    cur["p_active"] = (
        hurdle.predict_proba(X_cur)[:, 1] if hurdle is not None else 0.5
    )
    cur["vol_ml"] = (
        np.expm1(vol_model.predict(X_cur)) if vol_model is not None else np.nan
    )
    cur["vol_ml"] = cur["vol_ml"].clip(lower=0)

    shrunk = shrunk_volume(panel, cfg)
    cand = cur[LANE_COLS + ["p_active", "vol_ml"]].merge(
        shrunk, on=LANE_COLS, how="left"
    )
    cand["shrunk_volume"] = cand["shrunk_volume"].fillna(0.0)
    cand["vol_ml"] = cand["vol_ml"].fillna(cand["shrunk_volume"])

    cand["expected_volume"] = (
        cfg.ml_weight * cand["vol_ml"]
        + (1 - cfg.ml_weight) * cand["shrunk_volume"]
    )
    cand["score"] = (
        cand["p_active"].clip(lower=cfg.min_p_active) * cand["expected_volume"]
    )

    chunks: list[pd.DataFrame] = []
    for origin, total_vol in plan_by_origin.items():
        sub = cand[cand["origin"] == origin]
        chunks.extend(
            _allocate_plan(
                sub,
                origin=origin,
                total_vol=total_vol,
                use_other_bucket=cfg.use_other_bucket,
                coverage_target=cfg.coverage_target,
            )
        )

    if not chunks:
        return pd.DataFrame(columns=LANE_COLS + ["forecast_volume"])
    return pd.concat(chunks, ignore_index=True)


def build_forecasters(cfg: Config) -> dict[str, ForecasterFn]:
    return {
        "naive_last":  naive_last_month,
        "weighted":    weighted_baseline,
        "linreg":      linreg_baseline,
        "hurdle_topk": partial(hurdle_topk, cfg=cfg),
    }


# =============================================================================
# 8. Бэктест и тюнинг
# =============================================================================
def rolling_backtest(
    facts: pd.DataFrame,
    plans: pd.DataFrame,
    eval_months: Iterable[pd.Period],
    forecasters: dict[str, ForecasterFn],
) -> pd.DataFrame:
    """По каждому месяцу прогоняем все модели и собираем метрики."""
    eval_months = list(eval_months)
    rows: list[dict] = []
    for m in eval_months:
        plan_m = plans[plans["month"] == m]
        plan_by_origin = dict(zip(plan_m["origin"], plan_m["plan_volume"]))
        if not plan_by_origin:
            logger.warning("На %s нет плана - пропускаем", m)
            continue
        fact_m = facts[(facts["month"] == m) & facts["origin"].isin(plan_by_origin)]
        fact_agg = fact_m.groupby(LANE_COLS, as_index=False)["volume"].sum()
        history = facts[facts["month"] < m]

        for name, fn in forecasters.items():
            pred = fn(history, m, plan_by_origin, plans)
            scores = evaluate(fact_agg, pred)
            rows.append({
                "month": str(m),
                "model": name,
                "wape": scores.wape,
                "mape": scores.mape,
                "mape_top": scores.mape_top,
                "coverage": scores.coverage,
                "plan_total": plan_m["plan_volume"].sum(),
                "fact_total": fact_agg["volume"].sum(),
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Среднее по месяцам + ratio относительно naive_last."""
    agg = results.groupby("model", as_index=False).agg(
        wape=("wape", "mean"),
        mape=("mape", "mean"),
        mape_top=("mape_top", "mean"),
        coverage=("coverage", "mean"),
    )
    if "naive_last" in agg["model"].values:
        baseline = agg.loc[agg["model"] == "naive_last", "wape"].iloc[0]
        agg["ratio_vs_naive"] = agg["wape"] / baseline
    return agg.sort_values("wape").reset_index(drop=True)


def tune_hyperparams(
    facts: pd.DataFrame,
    plans: pd.DataFrame,
    val_months: Iterable[pd.Period],
    cfg: Config,
) -> tuple[Config, list[dict]]:
    """Grid search по coverage_target / ml_weight / min_p_active."""
    val_months = list(val_months)
    grid = list(itertools.product(
        cfg.tune_grid_coverage,
        cfg.tune_grid_ml_weight,
        cfg.tune_grid_min_p_active,
    ))
    if not grid:
        logger.warning("Пустая сетка тюнинга. Возвращаем исходный Config.")
        return cfg, []

    logger.info("Запуск тюнинга: %d комбинаций на %d месяцах",
                len(grid), len(val_months))

    trials: list[dict] = []
    best_cfg = cfg
    best_wape = float("inf")
    for cov, w, mpa in grid:
        trial_cfg = replace(cfg, coverage_target=cov, ml_weight=w, min_p_active=mpa)
        forecasters = {"hurdle_topk": build_forecasters(trial_cfg)["hurdle_topk"]}
        res = rolling_backtest(facts, plans, val_months, forecasters)
        wape = float(res["wape"].mean()) if not res.empty else float("nan")
        trials.append({
            "coverage_target": cov, "ml_weight": w, "min_p_active": mpa, "wape": wape,
        })
        logger.info(
            "  coverage=%.2f ml_weight=%.2f min_p=%.2f -> WAPE=%.4f",
            cov, w, mpa, wape,
        )
        if wape < best_wape:
            best_wape = wape
            best_cfg = trial_cfg

    logger.info(
        "Лучшие параметры: coverage=%.2f ml_weight=%.2f min_p=%.2f WAPE=%.4f",
        best_cfg.coverage_target, best_cfg.ml_weight, best_cfg.min_p_active,
        best_wape,
    )
    return best_cfg, trials


# =============================================================================
# 9. Пайплайн
# =============================================================================
def determine_months(
    facts: pd.DataFrame, plans: pd.DataFrame, cfg: Config,
) -> tuple[list[pd.Period], pd.Period, bool]:
    """Возвращает (validation_months, target_month, target_has_fact)."""
    fact_months = set(facts["month"].unique())
    plan_months = set(plans["month"].unique())
    months_with_both = sorted(fact_months & plan_months)
    if not months_with_both:
        raise RuntimeError("Не найдено ни одного месяца, где есть и факт, и план.")

    if cfg.target_month is not None:
        target_month = pd.Period(cfg.target_month, freq="M")
        if target_month not in plan_months:
            raise ValueError(
                f"Для целевого месяца {target_month} нет плана в plan_fact."
            )
        target_has_fact = target_month in fact_months
        eligible = [m for m in months_with_both if m < target_month]
    else:
        target_month = months_with_both[-1]
        target_has_fact = True
        eligible = months_with_both[:-1]

    if not eligible:
        raise RuntimeError(
            "Нет ни одного исторического месяца с фактом и планом до target_month "
            "- нечем валидировать модели."
        )

    n = min(cfg.n_validation_months, len(eligible))
    return eligible[-n:], target_month, target_has_fact


def _log_new_directions_share(facts: pd.DataFrame) -> None:
    seen: set[tuple] = set()
    rows: list[dict] = []
    for m, sub in facts.groupby("month"):
        keys = set(map(tuple, sub[LANE_COLS].values))
        new = keys - seen
        rows.append({
            "month": str(m),
            "n_lanes": len(keys),
            "n_new": len(new),
            "new_share": len(new) / max(len(keys), 1),
        })
        seen |= keys
    info = pd.DataFrame(rows)
    if not info.empty:
        info = info.iloc[6:]
    if not info.empty:
        logger.info(
            "Доля новых направлений по месяцам (среднее: %.1f%%, диапазон %.1f-%.1f%%)",
            info["new_share"].mean() * 100,
            info["new_share"].min() * 100,
            info["new_share"].max() * 100,
        )


def run_pipeline(
    facts: pd.DataFrame, plans_raw: pd.DataFrame, cfg: Config,
) -> dict:
    """Полный прогон: препроцессинг -> валидация -> финальный прогноз.

    facts:     сырой DataFrame с историей (как в df)
    plans_raw: сырой DataFrame с планом (как в plan_fact)
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Нормализация данных...")
    facts_n = preprocess_facts(facts)
    plans_n = preprocess_plans(plans_raw)
    logger.info("Фактов: %d строк, плана: %d строк", len(facts_n), len(plans_n))
    _log_new_directions_share(facts_n)

    validation_months, target_month, target_has_fact = determine_months(
        facts_n, plans_n, cfg,
    )
    logger.info(
        "Валидация: %s ... %s (n=%d). Прогноз: %s (факт %s)",
        validation_months[0], validation_months[-1], len(validation_months),
        target_month, "есть" if target_has_fact else "будущий месяц",
    )

    if cfg.tune:
        cfg, _ = tune_hyperparams(facts_n, plans_n, validation_months, cfg)

    forecasters = build_forecasters(cfg)

    logger.info("Бэктест на валидации...")
    val_results = rolling_backtest(facts_n, plans_n, validation_months, forecasters)
    val_summary = summarize(val_results)
    val_summary.to_csv(
        cfg.output_dir / "validation_summary.csv", index=False, encoding="utf-8-sig",
    )
    val_results.to_csv(
        cfg.output_dir / "validation_by_month.csv", index=False, encoding="utf-8-sig",
    )
    logger.info("Сводка по валидации:\n%s", val_summary.to_string(index=False))

    chosen = cfg.forecast_model or val_summary.iloc[0]["model"]
    if chosen not in forecasters:
        raise ValueError(
            f"forecast_model={chosen!r} не найден среди {list(forecasters)}"
        )
    logger.info("Финальная модель для прогноза: %s", chosen)

    plan_target = plans_n[plans_n["month"] == target_month]
    plan_by_origin = dict(zip(plan_target["origin"], plan_target["plan_volume"]))
    history = facts_n[facts_n["month"] < target_month]
    final_pred = forecasters[chosen](history, target_month, plan_by_origin, plans_n)
    report = build_forecast_report(
        facts_n, target_month, final_pred, plan_origins=plan_by_origin.keys(),
    )

    suffix = "with_fact" if target_has_fact else "future"
    out_path = cfg.output_dir / f"forecast_{target_month}_{suffix}.xlsx"
    report.to_excel(out_path, index=False)
    logger.info("Отчёт сохранён: %s", out_path)

    result: dict = {
        "target_month": target_month,
        "target_has_fact": target_has_fact,
        "validation_summary": val_summary,
        "forecast": final_pred,
        "report": report,
        "report_path": out_path,
        "chosen_model": chosen,
    }
    if target_has_fact:
        fact_target = facts_n[
            (facts_n["month"] == target_month) & facts_n["origin"].isin(plan_by_origin)
        ]
        fact_agg = fact_target.groupby(LANE_COLS, as_index=False)["volume"].sum()
        holdout = evaluate(fact_agg, final_pred)
        logger.info(
            "HOLDOUT %s [%s]: WAPE=%.4f MAPE=%.4f MAPE_TOP=%.4f COV=%.4f",
            target_month, chosen,
            holdout.wape, holdout.mape, holdout.mape_top, holdout.coverage,
        )
        result["holdout"] = holdout
    return result


# =============================================================================
# 10. CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Прогноз отгрузки по направлениям. df и plan_fact задаются "
                    "в начале файла."
    )
    g = p.add_argument_group("Модель")
    g.add_argument("--no-other-bucket", action="store_true",
                   help="Не использовать бакет 'прочее'")
    g.add_argument("--coverage-target", type=float,
                   help="Доля плана, покрытая top-K lane (по умолчанию 0.92)")
    g.add_argument("--ml-weight", type=float,
                   help="Вес ML-объёма vs shrinkage в score (0..1)")
    g.add_argument("--min-p-active", type=float,
                   help="Нижний порог для P(active)")

    g2 = p.add_argument_group("Пайплайн")
    g2.add_argument("--n-validation-months", type=int,
                    help="Сколько месяцев брать на валидацию")
    g2.add_argument("--target-month",
                    help="Месяц для прогноза в формате YYYY-MM")
    g2.add_argument("--forecast-model",
                    help="Какой моделью считать финальный прогноз")
    g2.add_argument("--tune", action="store_true",
                    help="Запустить grid-search по hurdle_topk")
    g2.add_argument("--output-dir", help="Куда складывать csv/xlsx")

    p.add_argument("--log-level", default="INFO",
                   help="DEBUG / INFO / WARNING / ERROR")
    return p


def _apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if args.no_other_bucket:
        cfg.use_other_bucket = False
    if args.coverage_target is not None:
        cfg.coverage_target = args.coverage_target
    if args.ml_weight is not None:
        cfg.ml_weight = args.ml_weight
    if args.min_p_active is not None:
        cfg.min_p_active = args.min_p_active
    if args.n_validation_months is not None:
        cfg.n_validation_months = args.n_validation_months
    if args.target_month:
        cfg.target_month = args.target_month
    if args.forecast_model:
        cfg.forecast_model = args.forecast_model
    if args.tune:
        cfg.tune = True
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if df is None or plan_fact is None:
        logger.error(
            "Не заданы df / plan_fact. Подставь свои DataFrame'ы в начало файла "
            "(секция ВХОДНЫЕ ДАННЫЕ) и запусти снова."
        )
        return 1

    cfg = _apply_args(Config(), args)
    try:
        run_pipeline(df, plan_fact, cfg)
    except Exception as e:  # noqa: BLE001
        logger.error("Пайплайн упал: %s", e, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
