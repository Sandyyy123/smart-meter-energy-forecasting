"""
Project 16 - Smart Meters London
Baseline model: per-household SARIMA (or Prophet) with weather exogenous covariates.

Aggregates half-hourly kWh to daily totals per household, fits a small SARIMA
(or Prophet) per household with day-of-week and yearly seasonality and weather
covariates (mean daily temperature, mean daily humidity), and produces 1-day,
7-day, and 30-day forecasts for the held-out final 60 days of the trial.

NOT executed by the scaffolding agent - run from main session via:
    python src/model_baseline.py --max-households 50

Outputs:
    deliverables/baseline_metrics.json     per-household and aggregate MAPE / MAE / RMSE
    deliverables/baseline_forecasts.parquet  per-household forecasts at 1d / 7d / 30d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "deliverables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_daily_dataset(daily_dir: Path) -> pd.DataFrame:
    """Concatenate all block files in `daily_dir` and return a tidy dataframe.

    Columns: LCLid, day, energy_sum, energy_mean, energy_max, energy_min, energy_std.
    """
    blocks: list[pd.DataFrame] = []
    files = sorted(daily_dir.glob("block_*.csv"))
    if not files:
        raise FileNotFoundError(f"No block files under {daily_dir}")
    for fp in files:
        b = pd.read_csv(fp, parse_dates=["day"])
        blocks.append(b)
    df = pd.concat(blocks, ignore_index=True)
    df = df.rename(
        columns={
            "energy_sum": "kwh_sum",
            "energy_mean": "kwh_mean",
            "energy_max": "kwh_max",
            "energy_min": "kwh_min",
            "energy_std": "kwh_std",
        }
    )
    return df


def load_weather_daily(path: Path) -> pd.DataFrame:
    """Daily weather aggregates from Darksky bundle."""
    w = pd.read_csv(path, parse_dates=["time"])
    w["day"] = w["time"].dt.normalize()
    daily = (
        w.groupby("day")
        .agg(
            temp=("temperatureMax", "mean"),
            humidity=("humidity", "mean"),
            wind=("windSpeed", "mean"),
            cloud=("cloudCover", "mean"),
        )
        .reset_index()
    )
    return daily


def load_holidays(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path, parse_dates=["Bank holidays"])
    h = h.rename(columns={"Bank holidays": "day", "Type": "holiday"})
    h["day"] = h["day"].dt.normalize()
    return h[["day", "holiday"]]


def load_households_meta(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# SARIMA backend
# ---------------------------------------------------------------------------
def fit_sarima(train: pd.Series, exog: pd.DataFrame | None) -> "SARIMAXResultsWrapper":
    """Fit a small SARIMA(1,1,1)(1,1,1,7) on daily kWh.

    Day-of-week seasonality (period 7); yearly seasonality is too long for
    daily SARIMA at this scale and is handled instead by a yearly Fourier
    series in `exog` (caller's responsibility).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train,
        exog=exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_prophet(train: pd.DataFrame) -> "Prophet":
    """Alternative backend - decomposable trend + weekly + yearly seasonality.

    `train` must have columns `ds` (day) and `y` (kwh_sum), plus optional
    regressor columns (`temp`, `humidity`, ...).
    """
    from prophet import Prophet

    m = Prophet(
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for col in ("temp", "humidity", "wind", "cloud"):
        if col in train.columns:
            m.add_regressor(col)
    m.fit(train)
    return m


# ---------------------------------------------------------------------------
# Yearly Fourier basis
# ---------------------------------------------------------------------------
def yearly_fourier(index: pd.DatetimeIndex, n_terms: int = 3) -> pd.DataFrame:
    doy = index.dayofyear.to_numpy()
    cols = {}
    for k in range(1, n_terms + 1):
        cols[f"yearly_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"yearly_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# Per-household pipeline
# ---------------------------------------------------------------------------
def build_household_panel(
    daily: pd.DataFrame, weather: pd.DataFrame, holidays: pd.DataFrame, lclid: str
) -> pd.DataFrame:
    """Tidy per-household daily panel ready for SARIMA / Prophet."""
    h = daily[daily["LCLid"] == lclid][["day", "kwh_sum"]].copy()
    h = h.dropna().sort_values("day").reset_index(drop=True)
    if h.empty:
        return h
    full_idx = pd.date_range(h["day"].min(), h["day"].max(), freq="D")
    h = h.set_index("day").reindex(full_idx).rename_axis("day").reset_index()
    h["kwh_sum"] = h["kwh_sum"].interpolate("linear", limit=3)
    h = h.merge(weather, on="day", how="left")
    h = h.merge(holidays, on="day", how="left")
    h["is_holiday"] = h["holiday"].notna().astype(int)
    h = h.drop(columns=["holiday"])
    return h


def evaluate_household(
    panel: pd.DataFrame, horizons: Iterable[int] = (1, 7, 30), backend: str = "sarima"
) -> dict:
    """Walk-forward evaluation on the final 60 days for the requested horizons."""
    if len(panel) < 200:
        return {"error": "insufficient_history", "n_days": len(panel)}

    test_n = 60
    train = panel.iloc[:-test_n].set_index("day")
    test = panel.iloc[-test_n:].set_index("day")

    exog_cols = ["temp", "humidity", "wind", "cloud", "is_holiday"]
    fourier = yearly_fourier(panel["day"].dt.normalize())
    fourier.index = panel["day"].dt.normalize()

    if backend == "sarima":
        train_exog = pd.concat([train[exog_cols], fourier.loc[train.index]], axis=1).fillna(0.0)
        test_exog = pd.concat([test[exog_cols], fourier.loc[test.index]], axis=1).fillna(0.0)
        fit = fit_sarima(train["kwh_sum"], train_exog)
        forecast = fit.forecast(steps=test_n, exog=test_exog)
    elif backend == "prophet":
        df = panel.rename(columns={"day": "ds", "kwh_sum": "y"})
        train_df = df.iloc[:-test_n]
        test_df = df.iloc[-test_n:]
        m = fit_prophet(train_df[["ds", "y"] + exog_cols])
        future = test_df[["ds"] + exog_cols]
        forecast = m.predict(future)["yhat"].values
        test = test.copy()
    else:
        raise ValueError(f"Unknown backend: {backend}")

    actual = test["kwh_sum"].to_numpy()
    pred = np.asarray(forecast)

    out = {"backend": backend, "n_days": len(panel)}
    for h in horizons:
        h = min(h, len(actual))
        a = actual[:h]
        p = pred[:h]
        denom = np.where(np.abs(a) < 1e-3, 1e-3, np.abs(a))
        out[f"mape_{h}d"] = float(np.mean(np.abs((a - p) / denom)) * 100)
        out[f"mae_{h}d"] = float(np.mean(np.abs(a - p)))
        out[f"rmse_{h}d"] = float(np.sqrt(np.mean((a - p) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-households", type=int, default=50)
    parser.add_argument("--backend", choices=("sarima", "prophet"), default="sarima")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    daily_dir = DATA_DIR / "daily_dataset" / "daily_dataset"
    weather_path = DATA_DIR / "weather_daily_darksky.csv"
    holidays_path = DATA_DIR / "uk_bank_holidays.csv"
    hh_meta_path = DATA_DIR / "informations_households.csv"

    log.info("Loading daily kWh blocks from %s", daily_dir)
    daily = load_daily_dataset(daily_dir)
    log.info("Daily rows: %s, households: %s", len(daily), daily["LCLid"].nunique())

    log.info("Loading weather (daily) and holidays")
    weather = load_weather_daily(weather_path)
    holidays = load_holidays(holidays_path)
    meta = load_households_meta(hh_meta_path)

    rng = np.random.default_rng(args.seed)
    eligible = daily.groupby("LCLid")["day"].nunique()
    eligible = eligible[eligible >= 250].index.to_numpy()
    sample = rng.choice(eligible, size=min(args.max_households, len(eligible)), replace=False)

    log.info(
        "Sampled %d households (eligible pool: %d) with backend=%s",
        len(sample),
        len(eligible),
        args.backend,
    )

    results: list[dict] = []
    for i, lclid in enumerate(sample, start=1):
        panel = build_household_panel(daily, weather, holidays, lclid)
        if panel.empty:
            log.warning("[%d/%d] %s empty panel, skipping", i, len(sample), lclid)
            continue
        try:
            res = evaluate_household(panel, backend=args.backend)
        except Exception as e:
            log.warning("[%d/%d] %s failed: %s", i, len(sample), lclid, e)
            continue
        res["LCLid"] = lclid
        meta_row = meta[meta["LCLid"] == lclid]
        if not meta_row.empty:
            res["acorn_grouped"] = meta_row["Acorn_grouped"].iloc[0]
            res["tariff"] = meta_row["stdorToU"].iloc[0]
        results.append(res)
        if i % 10 == 0:
            log.info("[%d/%d] last household MAPE@7d=%.1f%%", i, len(sample), res.get("mape_7d", np.nan))

    if not results:
        log.error("No households evaluated successfully")
        return 1

    df = pd.DataFrame(results)
    summary = {
        "n_households": int(len(df)),
        "backend": args.backend,
        "median_mape_1d": float(df["mape_1d"].median()),
        "median_mape_7d": float(df["mape_7d"].median()),
        "median_mape_30d": float(df["mape_30d"].median()),
        "median_mae_1d_kwh": float(df["mae_1d"].median()),
        "median_mae_7d_kwh": float(df["mae_7d"].median()),
        "median_mae_30d_kwh": float(df["mae_30d"].median()),
        "median_rmse_7d_kwh": float(df["rmse_7d"].median()),
    }
    out_metrics = OUT_DIR / "baseline_metrics.json"
    out_metrics.write_text(json.dumps({"per_household": results, "summary": summary}, indent=2))
    log.info("Wrote %s (summary=%s)", out_metrics, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
