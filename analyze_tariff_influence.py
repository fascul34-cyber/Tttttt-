#!/usr/bin/env python3
"""Анализ коэффициентов влияния показателей на тариф перевозки."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_COLUMNS = {
    "tariff": "тариф",
    "group": "группность",
    "rate": "ставка",
    "distance": "расстояние",
    "weight_per_wagon": "вес_на_вагон",
    "total_weight": "сумма_веса",
    "tariff_sum": "сумма_тарифа",
}

FEATURE_LABELS = {
    "group": "группность",
    "distance": "расстояние",
    "weight_per_wagon": "вес_на_вагон",
    "total_weight": "сумма_веса",
    "rate": "ставка",
}

MIN_SAMPLE_WARNING = 30
VIF_WARNING_THRESHOLD = 10.0


def parse_column_overrides(pairs: list[str] | None) -> dict[str, str]:
    """Parse CLI overrides like 'группность=вагонов'."""
    overrides: dict[str, str] = {}
    if not pairs:
        return overrides
    key_map = {v: k for k, v in DEFAULT_COLUMNS.items()}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Неверный формат --columns: '{pair}'. Ожидается key=value")
        src, dst = pair.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if src in DEFAULT_COLUMNS:
            overrides[src] = dst
        elif src in key_map:
            overrides[key_map[src]] = dst
        else:
            raise ValueError(
                f"Неизвестный ключ колонки '{src}'. "
                f"Доступны: {', '.join(DEFAULT_COLUMNS.keys())} или русские имена."
            )
    return overrides


def load_data(path: Path, column_map: dict[str, str]) -> pd.DataFrame:
    """Load CSV or Excel and rename columns to internal names."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

    rename: dict[str, str] = {}
    for internal, default_name in DEFAULT_COLUMNS.items():
        file_col = column_map.get(internal, default_name)
        if file_col in df.columns:
            rename[file_col] = internal

    missing = [
        column_map.get(k, DEFAULT_COLUMNS[k])
        for k in ("group", "distance", "weight_per_wagon", "total_weight")
        if column_map.get(k, DEFAULT_COLUMNS[k]) not in df.columns
    ]
    if missing:
        raise ValueError(
            f"В файле отсутствуют обязательные колонки: {', '.join(missing)}. "
            f"Найденные колонки: {', '.join(df.columns.astype(str))}"
        )

    df = df.rename(columns=rename)
    for col in df.columns:
        if col in DEFAULT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_tariff(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tariff as tariff_sum / total_weight when tariff is missing."""
    result = df.copy()
    if "tariff_sum" in result.columns and result["tariff_sum"].notna().any():
        mask = result["tariff"].isna() | (result["tariff"] <= 0)
        valid = result["tariff_sum"].notna() & (result["total_weight"] > 0)
        result.loc[mask & valid, "tariff"] = (
            result.loc[mask & valid, "tariff_sum"] / result.loc[mask & valid, "total_weight"]
        )
    if result["tariff"].isna().all() or (result["tariff"] <= 0).all():
        raise ValueError(
            "Не удалось определить тариф. Укажите колонку 'тариф' или пару 'сумма_тарифа' + 'сумма_веса'."
        )
    return result


def filter_outliers_iqr(df: pd.DataFrame, column: str = "tariff", factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers by IQR on the target column."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()


def select_features(include_rate: bool) -> list[str]:
    features = ["group", "distance", "weight_per_wagon", "total_weight"]
    if include_rate:
        features.append("rate")
    return features


def prepare_model_data(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    cols = features + ["tariff"]
    clean = df[cols].dropna()
    clean = clean[(clean["tariff"] > 0)]
    for feat in features:
        if feat == "group":
            clean = clean[clean[feat] >= 1]
        else:
            clean = clean[clean[feat] > 0]
    if len(clean) < 5:
        raise ValueError(f"Слишком мало наблюдений после очистки: {len(clean)}")
    return clean[features], clean["tariff"]


def calc_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for multicollinearity check."""
    x_const = sm.add_constant(X.astype(float))
    rows = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(x_const.values, i + 1)
        rows.append(
            {
                "признак": FEATURE_LABELS.get(col, col),
                "VIF": float(vif),
                "предупреждение": "да" if vif > VIF_WARNING_THRESHOLD else "нет",
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def standardized_coefficients(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, float]:
    """OLS on z-scored features; coefficients are standardized betas."""
    z_x = (X - X.mean()) / X.std(ddof=0)
    z_y = (y - y.mean()) / y.std(ddof=0)
    z_x = z_x.replace([np.inf, -np.inf], np.nan).dropna()
    z_y = z_y.loc[z_x.index]

    model = sm.OLS(z_y, sm.add_constant(z_x)).fit()
    rows = []
    for col in X.columns:
        rows.append(
            {
                "признак": FEATURE_LABELS.get(col, col),
                "beta": model.params[col],
                "std_err": model.bse[col],
                "p_value": model.pvalues[col],
                "ci_low": model.conf_int().loc[col, 0],
                "ci_high": model.conf_int().loc[col, 1],
            }
        )
    return pd.DataFrame(rows).sort_values("beta", key=abs, ascending=False), float(model.rsquared)


def elasticity_coefficients(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, float]:
    """Log-log OLS; coefficients are elasticities."""
    log_x = np.log(X.astype(float))
    log_y = np.log(y.astype(float))
    valid = log_x.replace([np.inf, -np.inf], np.nan).dropna().index.intersection(
        log_y.replace([np.inf, -np.inf], np.nan).dropna().index
    )
    log_x = log_x.loc[valid]
    log_y = log_y.loc[valid]

    model = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    rows = []
    for col in X.columns:
        alpha = model.params[col]
        rows.append(
            {
                "признак": FEATURE_LABELS.get(col, col),
                "эластичность": alpha,
                "pct_изменение_тарифа_на_10pct_признака": alpha * 10,
                "std_err": model.bse[col],
                "p_value": model.pvalues[col],
                "ci_low": model.conf_int().loc[col, 0],
                "ci_high": model.conf_int().loc[col, 1],
            }
        )
    return pd.DataFrame(rows).sort_values("эластичность", key=abs, ascending=False), float(model.rsquared)


def shap_analysis(
    X: pd.DataFrame, y: pd.Series, output_dir: Path, random_state: int = 42
) -> tuple[pd.DataFrame, float]:
    """Random Forest + SHAP mean absolute importance."""
    display_names = [FEATURE_LABELS.get(c, c) for c in X.columns]
    X_display = X.copy()
    X_display.columns = display_names

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=max(2, len(X) // 20),
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    r2 = float(rf.score(X, y))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"признак": display_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, max(4, len(display_names) * 0.6)))
    shap.summary_plot(shap_values, X_display, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, max(3, len(display_names) * 0.5)))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    return importance, r2


def format_pvalue(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def print_report(
    std_df: pd.DataFrame,
    std_r2: float,
    elast_df: pd.DataFrame,
    elast_r2: float,
    shap_df: pd.DataFrame,
    shap_r2: float,
    vif_df: pd.DataFrame,
    n_obs: int,
    include_rate: bool,
) -> None:
    print("\n" + "=" * 60)
    print("КОЭФФИЦИЕНТЫ ВЛИЯНИЯ ПОКАЗАТЕЛЕЙ НА ТАРИФ")
    print("=" * 60)
    print(f"Наблюдений в модели: {n_obs}")

    if n_obs < MIN_SAMPLE_WARNING:
        print(
            f"\n[!] Предупреждение: выборка < {MIN_SAMPLE_WARNING} — p-value могут быть ненадёжными."
        )

    if not include_rate:
        print(
            "\n[i] Ставка исключена из модели (мультиколлинеарность с тарифом). "
            "Используйте --include-rate для включения."
        )
    else:
        print("\n[!] Ставка включена — интерпретируйте коэффициенты с осторожностью.")

    high_vif = vif_df[vif_df["VIF"] > VIF_WARNING_THRESHOLD]
    if not high_vif.empty:
        print("\n[!] Высокий VIF (>10) — возможна мультиколлинеарность:")
        for _, row in high_vif.iterrows():
            print(f"   {row['признак']}: VIF = {row['VIF']:.1f}")

    print(f"\n--- Стандартизированные коэффициенты (R2 = {std_r2:.3f}) ---")
    for _, row in std_df.iterrows():
        sign_note = ""
        if row["признак"] == "группность":
            if row["beta"] < 0:
                sign_note = " -> отрицательное влияние OK (больше вагонов - дешевле)"
            else:
                sign_note = " -> неожиданный положительный знак"
        print(
            f"  {row['признак']:16s}  beta = {row['beta']:+.3f}  "
            f"(p={format_pvalue(row['p_value'])}){sign_note}"
        )

    print(f"\n--- Эластичность log-log (R2 = {elast_r2:.3f}) ---")
    for _, row in elast_df.iterrows():
        print(
            f"  {row['признак']:16s}  alpha = {row['эластичность']:+.3f}  "
            f"-> +10% признака ~ {row['pct_изменение_тарифа_на_10pct_признака']:+.1f}% тарифа  "
            f"(p={format_pvalue(row['p_value'])})"
        )

    print(f"\n--- SHAP - нелинейный вклад (Random Forest R2 = {shap_r2:.3f}) ---")
    top = shap_df.iloc[0]["признак"] if len(shap_df) else "-"
    for _, row in shap_df.iterrows():
        note = " (наибольший вклад)" if row["признак"] == top else ""
        print(f"  {row['признак']:16s}  |SHAP| = {row['mean_abs_shap']:.4f}{note}")

    print("\n" + "=" * 60)


def export_report(
    output_dir: Path,
    std_df: pd.DataFrame,
    elast_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    vif_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    std_df.to_csv(output_dir / "standardized_coefficients.csv", index=False, encoding="utf-8-sig")
    elast_df.to_csv(output_dir / "elasticity_coefficients.csv", index=False, encoding="utf-8-sig")
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
    vif_df.to_csv(output_dir / "vif.csv", index=False, encoding="utf-8-sig")
    print(f"\nОтчёты сохранены в: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Расчёт коэффициентов влияния показателей на тариф перевозки.",
    )
    parser.add_argument("input", type=Path, help="Путь к CSV или Excel файлу с данными")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Папка для отчётов (по умолчанию: output)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        metavar="KEY=FILE_COL",
        help="Переопределение имён колонок, напр. group=вагонов distance=км",
    )
    parser.add_argument(
        "--include-rate",
        action="store_true",
        help="Включить ставку в модель (по умолчанию исключена из-за мультиколлинеарности)",
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Не удалять выбросы по тарифу (метод IQR)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed для Random Forest")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Ошибка: файл не найден: {args.input}", file=sys.stderr)
        return 1

    column_map = DEFAULT_COLUMNS.copy()
    overrides = parse_column_overrides(args.columns)
    column_map.update(overrides)

    try:
        df = load_data(args.input, column_map)
        df = compute_tariff(df)

        if not args.no_outlier_filter:
            before = len(df)
            df = filter_outliers_iqr(df)
            removed = before - len(df)
            if removed:
                print(f"Удалено выбросов по тарифу (IQR): {removed} из {before}")

        features = select_features(args.include_rate)
        X, y = prepare_model_data(df, features)

        vif_df = calc_vif(X)
        std_df, std_r2 = standardized_coefficients(X, y)
        elast_df, elast_r2 = elasticity_coefficients(X, y)
        shap_df, shap_r2 = shap_analysis(X, y, args.output, random_state=args.seed)

        export_report(args.output, std_df, elast_df, shap_df, vif_df)
        print_report(
            std_df, std_r2, elast_df, elast_r2, shap_df, shap_r2, vif_df, len(X), args.include_rate
        )
    except (ValueError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
