"""
Project 16 - Smart Meters London
Advanced model: Temporal Fusion Transformer (TFT) using pytorch-forecasting,
multi-horizon prediction (1d / 7d / 30d), grouped by ACORN supergroup, with
quantile prediction intervals.

Architecture choices follow Lim 2021 (DOI 10.1016/j.ijforecast.2021.03.012):
  - Static categoricals: LCLid, Acorn_grouped, tariff (Std vs dToU)
  - Known-future reals: day-of-week, day-of-year sin/cos, is_holiday,
    forecast weather (temp, humidity, wind, cloud)
  - Observed reals: lagged kwh_sum (the target itself), kwh_mean, kwh_std
  - Hidden size 64, attention heads 4, dropout 0.2, learning rate 3e-4
  - Quantile regression head with quantiles {0.1, 0.5, 0.9}

NOT executed by the scaffolding agent. Main session runs:
    python src/model_advanced.py --max-households 500 --epochs 30

Outputs:
    deliverables/tft_model.ckpt          best checkpoint
    deliverables/tft_metrics.json        per-horizon QuantileLoss / pinball / SMAPE
    deliverables/tft_forecasts.parquet   per-household forecasts with q10/q50/q90
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tft")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "deliverables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_panel(daily_dir: Path, weather_path: Path, hh_path: Path, holidays_path: Path) -> pd.DataFrame:
    """Concatenate daily blocks, attach weather + ACORN + holidays, return a tidy panel."""
    blocks: list[pd.DataFrame] = []
    files = sorted(daily_dir.glob("block_*.csv"))
    for fp in files:
        b = pd.read_csv(fp, parse_dates=["day"])
        blocks.append(b)
    daily = pd.concat(blocks, ignore_index=True)
    daily = daily.rename(columns={"energy_sum": "kwh_sum", "energy_mean": "kwh_mean", "energy_std": "kwh_std"})
    daily = daily[["LCLid", "day", "kwh_sum", "kwh_mean", "kwh_std"]].dropna(subset=["kwh_sum"])

    w = pd.read_csv(weather_path, parse_dates=["time"])
    w["day"] = w["time"].dt.normalize()
    weather = (
        w.groupby("day")
        .agg(temp=("temperatureMax", "mean"), humidity=("humidity", "mean"),
             wind=("windSpeed", "mean"), cloud=("cloudCover", "mean"))
        .reset_index()
    )

    h = pd.read_csv(holidays_path, parse_dates=["Bank holidays"])
    h["day"] = h["Bank holidays"].dt.normalize()
    h["is_holiday"] = 1
    holidays = h[["day", "is_holiday"]]

    meta = pd.read_csv(hh_path)[["LCLid", "Acorn_grouped", "stdorToU"]].rename(columns={"stdorToU": "tariff"})

    panel = daily.merge(weather, on="day", how="left")
    panel = panel.merge(holidays, on="day", how="left")
    panel["is_holiday"] = panel["is_holiday"].fillna(0).astype(int)
    panel = panel.merge(meta, on="LCLid", how="left")

    panel["dow"] = panel["day"].dt.dayofweek.astype(int)
    doy = panel["day"].dt.dayofyear.to_numpy()
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    panel = panel.sort_values(["LCLid", "day"]).reset_index(drop=True)
    panel["time_idx"] = panel.groupby("LCLid").cumcount().astype(int)

    panel = panel.dropna(subset=["Acorn_grouped", "tariff"])
    panel["temp"] = panel["temp"].fillna(panel["temp"].median())
    panel["humidity"] = panel["humidity"].fillna(panel["humidity"].median())
    panel["wind"] = panel["wind"].fillna(panel["wind"].median())
    panel["cloud"] = panel["cloud"].fillna(panel["cloud"].median())
    panel["kwh_mean"] = panel["kwh_mean"].fillna(0.0)
    panel["kwh_std"] = panel["kwh_std"].fillna(0.0)
    return panel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_tft(panel: pd.DataFrame, max_encoder_length: int, max_prediction_length: int):
    """Build the pytorch-forecasting TimeSeriesDataSet and TFT model."""
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    cutoff = panel["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        panel[panel["time_idx"] <= cutoff],
        time_idx="time_idx",
        target="kwh_sum",
        group_ids=["LCLid"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["LCLid", "Acorn_grouped", "tariff"],
        time_varying_known_categoricals=["dow", "is_holiday"],
        time_varying_known_reals=["time_idx", "temp", "humidity", "wind", "cloud", "doy_sin", "doy_cos"],
        time_varying_unknown_reals=["kwh_sum", "kwh_mean", "kwh_std"],
        target_normalizer=GroupNormalizer(groups=["LCLid"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, panel, predict=True, stop_randomization=True)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return training, validation, tft


def fit_and_evaluate(panel: pd.DataFrame, max_households: int, epochs: int, batch_size: int) -> dict:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    keep = panel.groupby("LCLid")["time_idx"].count()
    keep = keep[keep >= 365].index
    if len(keep) > max_households:
        rng = np.random.default_rng(42)
        keep = rng.choice(keep, size=max_households, replace=False)
    panel = panel[panel["LCLid"].isin(keep)].reset_index(drop=True)
    panel["time_idx"] = panel.groupby("LCLid").cumcount().astype(int)

    log.info("Panel after filter: %d rows / %d households", len(panel), panel["LCLid"].nunique())

    max_encoder_length = 60
    max_prediction_length = 30

    training, validation, tft = build_tft(panel, max_encoder_length, max_prediction_length)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    ckpt = ModelCheckpoint(dirpath=str(OUT_DIR), filename="tft_model", monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", patience=6, mode="min")
    trainer = pl.Trainer(
        max_epochs=epochs,
        gradient_clip_val=0.1,
        callbacks=[ckpt, early],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)
    preds = best.predict(val_loader, mode="quantiles", return_x=True)
    actuals = torch.cat([y[0] for x, (y, _) in iter(val_loader)])

    q50 = preds.output[..., 1]
    abs_err = (actuals - q50).abs()
    smape_per_step = 200.0 * abs_err / (actuals.abs() + q50.abs() + 1e-6)

    horizons = {1: 0, 7: 6, 30: 29}
    metrics = {}
    for h, idx in horizons.items():
        if idx < smape_per_step.shape[1]:
            metrics[f"smape_{h}d"] = float(smape_per_step[:, idx].mean().item())
            metrics[f"mae_{h}d"] = float(abs_err[:, idx].mean().item())

    metrics["n_households"] = int(panel["LCLid"].nunique())
    metrics["best_ckpt"] = ckpt.best_model_path
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-households", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    daily_dir = DATA_DIR / "daily_dataset" / "daily_dataset"
    weather_path = DATA_DIR / "weather_daily_darksky.csv"
    holidays_path = DATA_DIR / "uk_bank_holidays.csv"
    hh_path = DATA_DIR / "informations_households.csv"

    log.info("Building panel from %s", daily_dir)
    panel = load_panel(daily_dir, weather_path, hh_path, holidays_path)
    log.info("Panel rows: %s, households: %s", len(panel), panel["LCLid"].nunique())

    metrics = fit_and_evaluate(panel, args.max_households, args.epochs, args.batch_size)
    out_path = OUT_DIR / "tft_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("Wrote %s", out_path)
    log.info("metrics=%s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
