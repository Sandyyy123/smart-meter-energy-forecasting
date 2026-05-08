# Project 16 - Smart Meters London Energy Forecasting

**Track:** Time-series forecasting (energy demand, household-level)
**Domain:** Energy / utilities (DACH twin: smart-meter rollout under BSI Smart Meter Gateway, EnWG Section 14a flexible loads)
**Difficulty:** 8/10
**Status:** Phase 1 - scaffolded code only

## Goal

Forecast half-hourly household electricity consumption from the London Smart Meter trial (5,567 households, Nov 2011 - Feb 2014), producing per-household and aggregated multi-horizon forecasts at 1-day, 7-day, and 30-day horizons. Evaluate against weather covariates and the ACORN demographic clustering provided with the dataset.

## Why this project

Smart-meter forecasting is the operational core of demand response, dynamic tariffs, and distribution-grid planning. The same toolkit transfers directly to DACH utilities: Stadtwerke and Verteilnetzbetreiber (VNB) need household-level half-hourly forecasts under the rollout mandated by the Messstellenbetriebsgesetz (MsbG), and Section 14a EnWG creates incentives for steerable-load forecasting (heat pumps, wallboxes, battery storage).

This project is the energy twin of project #9 (Favorita retail sales). Both are hierarchical multi-series forecasting tasks; the contrast is that retail series are dominated by promotional and calendar shocks while energy series are dominated by weather and intra-day occupancy structure.

## Data

| Source | Scope | Size | Use |
|--------|-------|------|-----|
| [Kaggle - Smart meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) | 5,567 London households, half-hourly Nov 2011 - Feb 2014 | ~1.25 GB compressed | Primary dataset |
| Darksky weather (bundled) | Hourly London weather (temp, humidity, wind, cloud) | included | Exogenous covariates |
| ACORN demographic | UK demographic segmentation per household | included | Grouping variable |
| UK national holidays | bundled | small | Calendar covariate |

Data acquisition is documented only (1.25 GB places it in the judgment-call zone; default per agent rules is to document, not download). See `data/README.md` for the Kaggle CLI command and post-download layout.

## Target

Half-hourly household kWh consumption (`energy(kWh/hh)` column). Aggregated to daily for SARIMA/Prophet baseline; kept at half-hourly for the TFT advanced model.

## Methodology

- **Baseline (`src/model_baseline.py`):** per-household SARIMA or Prophet, weather (temp, humidity) as exogenous covariates, day-of-week and yearly seasonality, aggregated to daily.
- **Advanced (`src/model_advanced.py`):** Temporal Fusion Transformer (Lim 2021) using `pytorch-forecasting`, multi-horizon prediction (1d/7d/30d), grouped by ACORN cluster, quantile regression for prediction intervals, with weather and calendar known-future covariates plus household ID as static categorical.

## Deliverables

- `brief.md` (this file)
- `data/README.md` (Kaggle download command + setup notes)
- `notebooks/01_EDA.ipynb` (raw, not executed)
- `reports/references.md` (20+ verified references)
- `src/model_baseline.py` (SARIMA / Prophet, runnable, not executed)
- `src/model_advanced.py` (TFT via pytorch-forecasting, runnable, not executed)
- `manuscripts/manuscript.md` (IMRaD, 4000-5000 words)
- `deliverables/presentation.html` (self-contained, inline CSS)
- `checkpoint.json` (status JSON)

## Open questions for follow-up phase

1. Hierarchical reconciliation: should aggregated forecasts (ACORN cluster, full feeder) be reconciled with per-household forecasts via MinT or bottom-up methods?
2. Cold-start households (less than 3 months of history) - how to handle in TFT; transfer learning from similar ACORN clusters?
3. Counterfactual evaluation for the dynamic tariff (dToU) cohort embedded in the dataset; treat as natural experiment for treatment-effect modelling?
4. DACH transfer: how do German Lastprofile (BDEW H0/G0/L0) compare to ACORN clusters for cold-start segmentation?
