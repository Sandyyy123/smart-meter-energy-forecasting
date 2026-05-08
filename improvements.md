# Improvements - Project 16 (Smart Meters London)

**Role:** IMPROVER
**Scope:** scaffold review, no file modifications.

## Top recommendation

**Add a LightGBM tabular comparator (lag-1 / lag-7 / lag-30 + weather + calendar + ACORN one-hots) as a third headline model alongside SARIMA and TFT.** M5 retrospective evidence (Makridakis 2022) shows boosted-tree ensembles often match or beat deep architectures on short-horizon multi-series load forecasting at a fraction of training cost. A LightGBMRegressor on the same daily panel takes ~10 minutes on CPU vs 4-6 hours for TFT on the planned RTX 5090, gives SHAP-based feature attribution that complements TFT's variable-selection-network weights, and provides the operational baseline most utilities can actually deploy. Implementation: extend `src/model_baseline.py` with a `--backend lightgbm` branch using lag features and `early_stopping_rounds=50` on a per-ACORN-supergroup time-block CV split. Without this comparator the SARIMA-vs-TFT contrast is incomplete and the manuscript's M5 framing is unsupported by an in-paper experiment.

---

## Weaknesses and actionable improvements

### 1. Missing gradient-boosting comparator (HIGH)
The methodology jumps from per-household SARIMA to a global TFT, skipping the strongest pragmatic baseline in the operational utility stack. Add LightGBM on lagged features (lag-1, lag-7, lag-14, lag-30, rolling-7-day mean, rolling-30-day mean, weather, dow, doy_sin/cos, is_holiday, ACORN one-hots, tariff). Train globally with `LCLid` target-encoded; evaluate at the same 1d/7d/30d horizons via recursive forecasting. This satisfies the [Massaoudi 2021] reference already in the bibliography, which is currently cited but not implemented.

### 2. Calibration not enforced in the loss (HIGH)
The TFT QuantileLoss at q10/q50/q90 trains for pinball-loss minimisation but does not guarantee calibrated coverage. Add a post-hoc conformal calibration step (split-conformal or CQR per Romano 2019) on a held-out calibration window of 14 days per household, producing PI's with finite-sample 80% coverage guarantee. Implementation: `MapieRegressor` wrapper or a 30-line custom conformal layer on top of `predict(mode="quantiles")`. Report empirical coverage and width side-by-side with raw quantile coverage in Table 2.

### 3. No proper time-series cross-validation (HIGH)
Current protocol holds out the final 60 days (baseline) or final 30 days (TFT) as a single test slice. This produces a high-variance estimate that is sensitive to seasonal alignment of the cutoff. Add expanding-window CV with at least 3 folds (cuts at -150d, -90d, -30d) and report median+IQR of MAPE/SMAPE across folds. The `scikit-learn` `TimeSeriesSplit` or `mlforecast` rolling-origin evaluation handles the bookkeeping. Currently the manuscript's expected-MAPE ranges (25-35% baseline, 15-30% reduction) are unsupported by a CV protocol.

### 4. dToU treatment-effect arm not modelled (MEDIUM)
The brief and Section 5 explicitly flag the dToU sub-cohort as a randomised natural experiment, but neither script estimates the treatment effect. Add a dedicated section/script implementing a difference-in-differences estimator: regress half-hourly kWh on `treated x event_window` interaction with household + half-hour-of-day fixed effects on the dToU vs Std cohorts, matched on pre-trial mean kWh and ACORN supergroup. This is the highest-value non-forecasting result the dataset can produce and would differentiate the project from the existing literature on LCL.

### 5. Cold-start and transfer-learning untested (MEDIUM)
TFT excludes households with <365 days of history from training (line 147 in `model_advanced.py`) but the brief and manuscript both motivate cold-start handling as a Phase-2 deliverable. Implement two ablations: (a) train on full-history households, fine-tune on 30-day cold-start households via a learning-rate-divided-by-10 schedule for 5 epochs; (b) leave-one-ACORN-out CV that holds out one supergroup at training and tests transfer to the held-out group. Without these the "DACH transfer" framing in Section 6 lacks empirical backing.

### 6. Hierarchical reconciliation cited but not run (MEDIUM)
[Wickramasuriya 2019] MinT is cited and is the operational target for substation-level forecasting, yet the headline only produces per-household forecasts. Add a small final step that aggregates predictions to ACORN-supergroup and trial-total levels and applies MinT reconciliation via `hierarchicalforecast` or `scikit-hts`. Reports the trace-of-error-covariance reduction. ~50 lines of code, high reviewer-impact.

### 7. Reproducibility gaps (MEDIUM)
No `requirements.txt`, `pyproject.toml`, or `environment.yml` exists in the project root. The TFT script seeds the household sample (`rng = np.random.default_rng(42)`) but does not seed `pytorch-lightning`, `torch.cuda`, or the dataloader. Add: (a) `requirements.txt` pinning `statsmodels==0.14.x`, `prophet==1.1.x`, `pytorch-forecasting==1.0.x`, `lightning==2.x`, `pyarrow`, `lightgbm`; (b) `pl.seed_everything(42, workers=True)` at the top of `fit_and_evaluate`; (c) a `Makefile` or `run.sh` that documents the canonical invocation order. Also pin Python (3.11.x recommended for `pytorch-forecasting` compat). Currently a re-run on a different machine cannot be guaranteed bit-exact.

### 8. Weather-as-known-future is unrealistic (MEDIUM)
Both models treat weather as a perfect known-future input (acknowledged in Section 5 but not tested). Add a sensitivity analysis: re-run TFT inference twice, once with true weather and once with weather + N(0, sigma) noise where sigma is calibrated to UK Met Office day-ahead forecast error (~1.5 degC for temperature, ~10% for humidity at 24h). Report SMAPE degradation per horizon. This is a single-day extension and would close the gap between published TFT benchmarks and deployment reality.

### 9. Notebook is sample-only, not full-dataset (LOW-MEDIUM)
`01_EDA.ipynb` only loads `block_0` for missingness, target distribution, weekly seasonality, and cold-start flagging (cells 8 onward). The cold-start counts and missingness rates reported in the manuscript Section 2 ("1% to 8% per household", "fewer than 90 distinct days") are not derivable from a single block. Add a final notebook cell that streams all 112 blocks via `pyarrow.dataset` and produces the per-household history-length histogram and missingness-by-block heatmap. Otherwise the manuscript narrative is unsupported by the EDA artefact.

### 10. Presentation audience mismatch (LOW)
`deliverables/presentation.html` not yet inspected by this reviewer (scope: scaffold review without modification), but the brief positions the project as DACH-utility-facing. Verify the deck closes with one slide of operational implications: cost per forecast in EUR, tariff-design economic value, and the BSI Smart Meter Gateway / Section 14a EnWG regulatory hook. If absent, add a final slide that translates SMAPE@1d into EUR/MWh balancing-cost reduction at typical EPEX day-ahead spreads.

---

## Priority summary

| # | Improvement | Priority |
|---|-------------|---------:|
| 1 | LightGBM tabular comparator | HIGH |
| 2 | Conformal calibration of TFT quantiles | HIGH |
| 3 | Expanding-window time-series CV | HIGH |
| 4 | dToU difference-in-differences arm | MEDIUM |
| 5 | Cold-start fine-tuning + leave-one-ACORN-out | MEDIUM |
| 6 | MinT hierarchical reconciliation run | MEDIUM |
| 7 | requirements.txt + full seeding | MEDIUM |
| 8 | Weather-noise sensitivity ablation | MEDIUM |
| 9 | Full-corpus EDA streaming via pyarrow | LOW-MEDIUM |
| 10 | Operational/EUR-impact closing slide | LOW |

---

## Compact summary

Project 16 has a credible scaffold (TFT + SARIMA, 27 verified refs, 4133-word IMRaD manuscript, exhaustive Section 5 limitations). Top gaps are experimental rigor, not architecture: missing LightGBM headline comparator (M5 evidence demands it), no conformal calibration to back the q10/q90 coverage claim, no rolling-origin CV (single 60-day cutoff is high-variance), unbuilt dToU treatment-effect arm despite explicit randomised-trial framing, and absent reproducibility scaffolding (no requirements.txt, partial seeding). Ten ranked improvements provided. Top single change: add LightGBM as a third headline model in `src/model_baseline.py` with `--backend lightgbm`, since it costs ~10 minutes of compute, satisfies the already-cited Massaoudi 2021 reference, and gives utilities a deployable baseline without GPU. Role B (IMPROVER) complete.
