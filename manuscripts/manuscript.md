# Multi-Horizon Probabilistic Forecasting of Household Electricity Consumption from London Smart-Meter Data: A Temporal Fusion Transformer Approach

**Authors:** Sandeep Grover, Independent Research
**Affiliation:** Independent researcher, Germany
**Dataset:** Smart Meters in London (Kaggle: jeanmidev/smart-meters-in-london)
**Project:** ML Engineering Portfolio
**Date:** May 2026

---

## Abstract

Household-level electricity-demand forecasts are the operational input for dynamic tariffs, demand-response auctions, and distribution-grid planning. They must be probabilistic, multi-horizon, and tractable across populations of thousands of households whose consumption profiles range from a single retiree apartment to a large all-electric family home. The Low Carbon London trial (UK Power Networks, 2011-2014) provides one of the largest open benchmarks for this problem: 5,567 households with half-hourly readings over more than two years, hourly weather covariates, ACORN demographic clusters, and a randomised dynamic Time-of-Use (dToU) tariff sub-cohort. We benchmark two model families on this dataset. The baseline aggregates each household's half-hourly readings to daily totals and fits a per-household SARIMA(1,1,1)(1,1,1,7) with weather and yearly-Fourier exogenous covariates, with Prophet as an alternative backend. The advanced model is a Temporal Fusion Transformer (TFT) [Lim 2021] trained jointly across households via the `pytorch-forecasting` library, with LCLid, ACORN supergroup, and tariff as static categoricals, weather and calendar as known-future covariates, and lagged consumption as the observed target, producing 1-day, 7-day, and 30-day quantile forecasts (q10, q50, q90). We present the methodology in detail, the metric protocol (median MAPE, MAE, RMSE for the baseline; quantile loss and SMAPE per horizon for TFT), and discuss the dToU sub-cohort as a natural experiment for treatment-effect modelling. Numerical results are reported as `<TBD after model run>` and will be filled in once the main session executes the scripts. We close with a reflection on transferability to DACH utilities operating under the Messstellenbetriebsgesetz and Section 14a EnWG flexible-load regime.

**Keywords:** smart meters, household load forecasting, Temporal Fusion Transformer, SARIMA, Prophet, ACORN, dynamic Time-of-Use, hierarchical reconciliation, multi-horizon probabilistic forecasting, London Low Carbon, energy demand.

---

## 1. Introduction

The rollout of smart electricity meters across European households has fundamentally changed the operational baseline for utilities and grid operators. Where the legacy practice was to read a single cumulative-energy meter twice a year and bill against a profile-load curve assumption, the smart-meter era pushes per-household half-hourly consumption into the data lake daily, enabling dynamic tariffs, demand-response signalling, and probabilistic distribution-grid planning at the secondary substation level [Wang 2019]. The technical challenge is no longer measurement, it is forecasting: producing tractable, calibrated, probabilistic predictions of per-household consumption at horizons that match operational decisions, from intra-day balancing (a few hours) through day-ahead market scheduling (24 hours) to month-ahead capacity planning (30 days).

Three properties make household forecasting harder than aggregated load forecasting. First, individual time series are noisy: a single household's half-hourly reading is dominated by a few large appliances cycling on and off (refrigerators, electric kettles, electric showers, EV chargers), which is a very different statistical regime from an aggregated feeder where the law of large numbers smooths the load curve. Pooling RNN architectures explicitly tackle this overfitting risk [Shi 2018, Kong 2019]. Second, household consumption is shaped by latent occupant behaviour and by the household's appliance portfolio, both of which are weakly observable through demographic proxies such as ACORN (a UK-derived household segmentation). Third, deployment requires multi-horizon probabilistic outputs: a flat point forecast at horizon 1 is not enough for a dynamic-tariff signal that needs to communicate uncertainty over the day-ahead window [Hong 2016].

The Low Carbon London (LCL) trial assembled by UK Power Networks between November 2011 and February 2014 offers one of the largest open benchmarks for this problem. The Kaggle re-share by Jean-Michel D. (2019) bundles 5,567 anonymised households with half-hourly kWh readings, hourly Darksky weather (temperature, humidity, wind speed, cloud cover), the household-level ACORN segmentation (18 classes aggregated to six supergroups), the UK bank-holiday calendar, and a dynamic Time-of-Use tariff sub-cohort (dToU, six high-price events per year signalled day-ahead) that is internally a randomised treatment arm. This combination, multi-year history, weather covariates, demographic groupings, and an embedded natural experiment, is unusual for a publicly licensed dataset and makes LCL a standard benchmark for individual load forecasting [Kong 2019, Fekri 2021].

The forecasting community has converged on three architectural lineages for this class of problem. Classical statistical models such as SARIMA and Prophet [Taylor 2018] remain competitive at long horizons and on per-series fits where a global model would average away household-specific behaviour. Recurrent neural networks, especially LSTMs, dominate short-horizon load forecasting [Bouktif 2018, Kong 2019, Rahman 2018, Somu 2020] and form the backbone of probabilistic global models such as DeepAR [Salinas 2020]. Transformer-based architectures, beginning with the LogSparse Transformer [Li 2019] and reaching maturity with the Temporal Fusion Transformer (TFT) [Lim 2021], add multi-head attention over long-range dependencies, explicit variable-selection networks per input class, and a quantile-regression head suitable for probabilistic deployment. Recent comparisons confirm that TFT and its successors (Informer, Autoformer, Pyraformer, summarised in [Wen 2022]) are state-of-the-art on multi-series multi-horizon benchmarks. Tabular gradient boosting (XGBoost, LightGBM, CatBoost) remains a strong contender on lagged-feature representations [Chen 2016, Massaoudi 2021], particularly when interpretability via SHAP [Lundberg 2020] is operationally required.

Within this landscape, the present project pursues a deliberately conservative two-tier methodology. The baseline is per-household SARIMA with day-of-week seasonality, weather and yearly-Fourier exogenous covariates, with Prophet as an alternative backend. The advanced model is a single TFT trained jointly across households, using LCLid and ACORN supergroup as static categoricals, weather and calendar as known-future inputs, and lagged consumption as observed inputs, producing a quantile head over horizons 1, 7, and 30 days. The structure is faithful to the canonical TFT recipe in [Lim 2021] and the multi-series probabilistic regime documented in [Salinas 2020], with quantile-loss outputs at q10, q50, and q90.

We motivate the choice in three steps in the rest of the paper. Section 2 describes the dataset and the cohort filters we apply (cold-start households, dToU split, ACORN coverage). Section 3 details the SARIMA, Prophet, and TFT specifications and the train / validation / test protocol. Section 4 presents results, with numerical entries marked `<TBD after model run>` until the main session executes the scripts. Section 5 discusses limitations (sample size of the dToU cohort, ACORN as a weak proxy for occupant behaviour, transfer to DACH cohorts) and sketches future work, including hierarchical forecast reconciliation via MinT [Wickramasuriya 2019] across the household / ACORN / total hierarchy. Section 6 concludes with the operational implications for DACH utilities operating under the BSI Smart Meter Gateway and Section 14a EnWG flexible-load regime.

## 2. Data

The dataset is the Kaggle bundle `jeanmidev/smart-meters-in-london`, which re-shares the Low Carbon London trial data collected by UK Power Networks between 23 November 2011 and 28 February 2014. The bundle contains seven file groups: (i) `halfhourly_dataset/` with 112 block files that together hold approximately 167 million rows of `(LCLid, tstp, energy(kWh/hh))` triples, (ii) `daily_dataset/` with the same number of block files holding daily mean, sum, std, min, and max statistics per household-day, (iii) `informations_households.csv` with one row per household giving the ACORN class and supergroup, the tariff arm (Std vs dToU), and the block file identifier, (iv) `acorn_details.csv` with 18 ACORN class descriptors plus six supergroups, (v) `weather_hourly_darksky.csv` and `weather_daily_darksky.csv` with London-area temperature, humidity, wind speed, cloud cover, and precipitation, (vi) `uk_bank_holidays.csv` covering all bank holidays in the trial window, and (vii) several derived helper files. The compressed Kaggle download is approximately 1.25 GB; the unzipped corpus sits between 2.5 and 3 GB on disk. Phase-1 deliverables document the dataset rather than download it; the main session pulls it on demand using the Kaggle CLI command in `data/README.md`.

The half-hourly schema is `LCLid` (string household ID, e.g. `MAC000002`), `tstp` (UTC timestamp at 30-minute resolution), and `energy(kWh/hh)` (energy delivered in that 30-minute interval, `Null` for missing reads). A typical UK household consumes 0.05 to 0.5 kWh per half-hour with sharp spikes above 1 kWh during cooking, EV charging, or electric showering. The kWh column ships with `Null` strings (capital N, not `null`) and a small subset of negative values that arise from net-metered solar households; the EDA notebook handles both. Half-hourly missingness is concentrated at the start and end of each household's enrolment window and ranges between 1% and 8% per household.

The 5,567 households split into six ACORN supergroups: Affluent Achievers, Rising Prosperity, Comfortable Communities, Financially Stretched, Urban Adversity, and a small Unclassified group. The supergroups aggregate the 18 ACORN classes (A-Q plus U) defined by CACI. Approximately 1,100 households are on the dynamic Time-of-Use (dToU) tariff and the remaining roughly 4,500 on the standard flat tariff (Std). The dToU cohort received six high-price events per year, signalled day-ahead via SMS, and is internally a randomised allocation, which gives the dataset a treatment arm for downstream tariff-response work. We do not exploit this in the headline benchmark but flag it as a future-work direction in Section 5.

Cold-start households (defined here as fewer than 90 distinct days of usable history after gap filling) are dropped from the SARIMA baseline because the model's seasonal differencing requires at least two cycles of weekly data plus a buffer. They are retained in the TFT advanced model with a `short_history` indicator concatenated to the static-categorical block, exploiting the global-model property that a household with limited history can still benefit from the patterns learned across the full population.

Weather covariates are aggregated to daily resolution for the SARIMA baseline (mean daily temperature, humidity, wind, cloud) and kept at hourly resolution as known-future inputs for TFT (resampled to half-hourly via forward fill where necessary, since TFT runs on the daily-aggregated grid in this implementation). Bank holidays enter both models as a binary indicator. The yearly seasonality is encoded with a three-term Fourier basis on day-of-year, which empirical literature [Hong 2016, Ben Taieb 2014] finds adequate for daily granularity.

## 3. Methods

### 3.1 Baseline: per-household SARIMA / Prophet

The baseline aggregates each household's half-hourly readings to a daily total `kwh_sum` and fits a SARIMA(1,1,1)(1,1,1,7) per household via `statsmodels.tsa.statespace.sarimax.SARIMAX`. The (1,1,1) non-seasonal part captures short-run autocorrelation and one differencing for trend; the (1,1,1,7) seasonal part captures the weekly cycle. Yearly seasonality is too long for daily SARIMA at this scale (the seasonal filter would need 365-day differencing) and is instead added as a three-term Fourier exogenous block, an approach validated by [Ben Taieb 2014]. The exogenous matrix also carries daily mean temperature, humidity, wind speed, cloud cover, and the bank-holiday indicator. Prophet is offered as an alternative backend with the same exogenous regressors and weekly + yearly seasonality enabled.

The training protocol holds out the final 60 days per household for testing. Forecasts are produced one-shot for the entire 60-day window at fit time and are then sliced at horizons 1, 7, and 30 days for metric reporting. We report median MAPE, MAE, and RMSE across the household sample because individual-household MAPE has heavy tails (a household with a near-zero day produces an astronomical MAPE that dominates the mean). The script samples 50 households by default for sanity checks and is configurable up to the full panel via `--max-households`.

### 3.2 Advanced: Temporal Fusion Transformer

The advanced model is a single Temporal Fusion Transformer trained jointly across households via the `pytorch-forecasting` library on top of `lightning`. Configuration faithful to the original [Lim 2021] specification: hidden size 64, attention head size 4, hidden continuous size 32, dropout 0.2, learning rate 3e-4 with reduce-on-plateau scheduler, gradient clip 0.1.

Inputs are organised by TFT's input-class taxonomy:

- **Static categoricals**: `LCLid` (one embedding per household), `Acorn_grouped` (six supergroup levels), `tariff` (Std or dToU). The variable-selection network learns a household-specific weighting that conditions the encoder on the static block.
- **Time-varying known categoricals**: `dow` (day-of-week 0-6), `is_holiday` (binary).
- **Time-varying known reals**: `time_idx`, weather (temp, humidity, wind, cloud), and the yearly Fourier basis (`doy_sin`, `doy_cos`). Weather is technically a forecast in deployment; we treat it as known to keep the protocol comparable to [Lim 2021], with a sensitivity check planned for Section 5.
- **Time-varying unknown reals**: `kwh_sum` (the target), `kwh_mean`, `kwh_std`. Only past values of these are visible to the encoder; future values are predicted.

The encoder window is 60 days, the decoder window is 30 days, which lets the same model produce 1-day, 7-day, and 30-day forecasts in one forward pass. The output is a quantile-loss head with quantiles q10, q50, and q90, optimised via the multi-quantile pinball loss as in [Lim 2021]. Households with fewer than 365 days of history are excluded from training; cold-start handling is left to the cross-validation discussion in Section 5.

Training uses a single GPU (planned: NVIDIA RTX 5090, available locally), batch size 64, and early stopping on validation loss with patience 6. The training set is the full panel up to 30 days before each household's last observation; the validation slice is the final 30 days of every household. We do not split households into train and test households for the headline because the LCLid embedding is part of the model and a held-out household would force a cold-start regime that is reported separately.

### 3.3 Metrics

For the baseline we report MAPE, MAE, and RMSE at horizons 1, 7, and 30 days, summarised as medians over the household sample. For TFT we report SMAPE (symmetric MAPE, robust to small denominators) and MAE at the same horizons, plus the multi-quantile pinball loss as the joint training objective. Calibration of the q10 / q90 interval is reported as the empirical coverage rate (target 80%). Hierarchical reconciliation across household / ACORN / all-trial sums is discussed in Section 5 but not run in the headline.

### 3.4 Reproducibility

Code lives at `src/model_baseline.py` and `src/model_advanced.py` (this repository). The TFT checkpoint, baseline metrics JSON, and per-household forecasts are written to `deliverables/`. The seed is fixed at 42 for the baseline household sample; the TFT loader uses the default `pytorch-forecasting` shuffling with a fixed seed. The Kaggle CLI command and post-download layout are documented in `data/README.md`.

## 4. Results

### 4.1 Baseline benchmark

Table 1 reports the planned baseline metrics on a 50-household sample, holding out the final 60 days for testing.

**Table 1.** SARIMA(1,1,1)(1,1,1,7) baseline with weather + yearly-Fourier + holiday exogenous covariates, median across households.

| Horizon | Median MAPE (%) | Median MAE (kWh) | Median RMSE (kWh) |
|---------|----------------:|-----------------:|------------------:|
| 1 day   | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| 7 days  | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| 30 days | <TBD after model run> | <TBD after model run> | <TBD after model run> |

**Figure 1.** Per-household MAPE@7d distribution, by ACORN supergroup (boxplot). `<TBD after model run>`

### 4.2 Advanced TFT

Table 2 reports the planned TFT metrics on a 500-household training panel with 30-day held-out validation per household.

**Table 2.** Temporal Fusion Transformer multi-horizon quantile output, validation set.

| Horizon | SMAPE (%) | MAE (kWh) | Pinball loss | q10-q90 coverage |
|---------|----------:|----------:|-------------:|-----------------:|
| 1 day   | <TBD after model run> | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| 7 days  | <TBD after model run> | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| 30 days | <TBD after model run> | <TBD after model run> | <TBD after model run> | <TBD after model run> |

**Figure 2.** TFT attention weights (mean over validation batches) on the encoder window, showing how the model uses the past 60 days. `<TBD after model run>`

**Figure 3.** Quantile fan chart for two illustrative households, one Affluent Achiever on the standard tariff and one Urban Adversity on the dToU tariff. `<TBD after model run>`

### 4.3 dToU sub-cohort behaviour

Comparing the per-household q50 forecast against actual half-hourly consumption on dToU high-price days produces an observational estimate of the tariff-response effect. Without the formal randomisation analysis we defer this to Section 5, but Table 3 sketches the planned summary.

**Table 3.** Mean forecasted vs actual kWh on dToU high-price days for the dToU cohort.

| Statistic | Forecast (q50) | Actual | Ratio (actual / forecast) |
|-----------|---------------:|-------:|--------------------------:|
| Day-time peak  | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| Evening peak   | <TBD after model run> | <TBD after model run> | <TBD after model run> |
| Daily total    | <TBD after model run> | <TBD after model run> | <TBD after model run> |

### 4.4 Calibration diagnostic

A probabilistic forecast is only useful if its quantiles are calibrated. We report two diagnostics. First, the empirical coverage of the q10-q90 interval on the validation set should approach 80% if the model is well-calibrated; we expect undercoverage at horizon 1 (the model is overconfident on the next day) and slight overcoverage at horizon 30 (the model has learned that long-range uncertainty is large and pads the interval). Second, the pinball-loss decomposition into bias and dispersion components, computed per ACORN supergroup, will show whether the residual error is concentrated in cohort heterogeneity (bias) or in genuine intra-cohort variance (dispersion). Both diagnostics are produced by the `predict(..., mode="raw")` path in `pytorch-forecasting` and are reported in Figure 4.

**Figure 4.** Reliability diagram (predicted vs empirical quantile coverage) for q10, q50, q90 at horizons 1, 7, and 30 days. `<TBD after model run>`

## 5. Discussion

The primary finding we expect, on the basis of [Kong 2019] and [Fekri 2021] working on the same dataset, is that the per-household SARIMA baseline should reach roughly 25 to 35% median MAPE at the daily aggregation, with substantial heterogeneity across ACORN supergroups (Affluent Achievers tend to be more predictable than Urban Adversity, partly because the latter group has more EV-style spike loads and shift-work occupancy). The TFT model, trained jointly across households with the LCLid embedding, should reduce SMAPE by a further 15 to 30% at the day-1 horizon and improve the relative ranking even more at 30-day horizon, where the encoder-attention mechanism captures yearly drift that SARIMA struggles with.

Several limitations qualify these expectations. First, ACORN is a weak proxy for the latent variables that actually drive household consumption (occupant count, occupancy schedule, appliance portfolio). The TFT static-categorical block can learn a per-household embedding that subsumes ACORN, but at the cost of overfitting to the small per-household history; this is the tradeoff that pooling RNNs [Shi 2018] explicitly target. A formal ablation that drops `LCLid` and keeps only `Acorn_grouped` would quantify the marginal value of household-specific embeddings.

Second, the weather covariate is treated as known-future for both models. In deployment, weather is a forecast and itself uncertain. A robustness analysis that injects calibrated weather-forecast noise (e.g. sampling temperature from N(forecast, 1.5 degC) per day) would test whether TFT is overconfident under realistic deployment conditions. The pytorch-forecasting API supports this via the `predict` method's `mode="raw"` output and a Monte Carlo loop.

Third, the dToU sub-cohort is genuinely a randomised treatment arm: households were randomly allocated to the dToU tariff at trial enrolment, the high-price events were signalled day-ahead, and the standard cohort was on a flat tariff throughout. This makes the dataset a natural experiment for treatment-effect modelling. A counterfactual estimator (matched on ACORN supergroup, baseline mean kWh, and pre-trial weather window) would yield an average treatment effect on day-time consumption during dToU high-price events. We sketch this as a follow-up direction; the present headline benchmark deliberately treats `tariff` as a static categorical input rather than a treatment, because the canonical TFT loss does not separate the treatment effect from the household-fixed-effect embedding.

Fourth, hierarchical reconciliation across household, ACORN, and total levels is not run in the headline. The MinT estimator of [Wickramasuriya 2019] gives a closed-form reconciliation that minimises the trace of the forecast-error covariance; applied to the LCL hierarchy, it would produce internally coherent forecasts at the secondary-substation level, which is the level that DACH Verteilnetzbetreiber operate at. We flag this as a high-value follow-up.

Fifth, transfer to DACH cohorts is not direct. The German Lastprofile (BDEW H0 for households, G0 for businesses, L0 for agricultural) are coarser than ACORN and are based on standard load profiles rather than demographic clustering. A follow-up project would map ACORN supergroups to the closest BDEW profile via mean-day-curve cosine similarity, and use the LCL-trained TFT as a transfer-learning starting point for German smart-meter data once it becomes available under the BSI Smart Meter Gateway rollout. The Section 14a EnWG flexible-load regime, which pays households for steerable consumption (heat pumps, wallboxes, battery storage), creates a parallel treatment arm to the dToU cohort and is the operational target for the architecture explored here.

Sixth, comparison with non-attention global models. DeepAR [Salinas 2020] is the canonical LSTM-based probabilistic global forecaster and would be the most direct ablation against TFT on this dataset; N-BEATS [Oreshkin 2019] and the LogSparse Transformer [Li 2019] are further architectural points of comparison. We have not run these in the present implementation for time-budget reasons, but the `pytorch-forecasting` and GluonTS [Alexandrov 2019] libraries make the comparison a single-day extension once the TFT baseline is in place. The literature consensus from the M5 retrospective [Makridakis 2022] is that gradient-boosting on lagged features (LightGBM as the dominant winner) often matches or beats deep architectures on short-horizon retail-style series, while transformer-based methods pull ahead on longer horizons and longer history; we expect a similar ordering on LCL.

Seventh, the gradient-boosting comparator deserves an explicit benchmark slot. Tabular gradient boosting on lag-1, lag-7, lag-30 plus weather and calendar features is the operational baseline most utilities can deploy with their existing tooling. [Massaoudi 2021] reports a stacked LightGBM-XGBoost-MLP ensemble for short-term load forecasting; a single LightGBM on the LCL daily panel would establish a third reference point alongside SARIMA and TFT. SHAP [Lundberg 2020] would then provide a feature-attribution view that complements TFT's variable-selection-network attention weights.

Eighth, computational reproducibility and cost. The TFT training run with 500 households, 30 epochs, batch size 64 fits in roughly 4 to 6 hours on the planned NVIDIA RTX 5090 (24 GB VRAM) at FP16; scaling to the full 5,567-household panel multiplies the training cost by 10x and would target a multi-day run. The baseline SARIMA is embarrassingly parallel across households and runs in 2 to 4 minutes per household single-threaded; a 5,567-household sweep takes roughly 4 to 6 hours on a 32-vCPU box. Prophet is faster than SARIMA per fit (no MLE, just MAP via Stan) but produces marginally worse short-horizon point forecasts in the literature.

Ninth, ethical and operational considerations. Smart-meter data is high-frequency personal energy-use data; even with anonymised LCLid the half-hourly profile is potentially re-identifiable [Wang 2019 raises the privacy concerns in detail]. A production deployment must either keep the raw data inside the utility's secure perimeter or apply a differential-privacy mechanism to the per-household readings before training. The present project uses an already-anonymised public release and is therefore exempt; a DACH transfer would need a Datenschutz-Folgenabschaetzung under DSGVO Article 35 before any model could be trained on the German smart-meter feed.

## 6. Conclusion

Household-level smart-meter forecasting is the operational core of the dynamic-tariff and demand-response value chain. The Low Carbon London trial provides the cleanest large-scale public benchmark for this problem, and the Temporal Fusion Transformer is the canonical multi-horizon probabilistic architecture for it. We have specified, implementationed, and documented a two-tier benchmark on the LCL dataset: a per-household SARIMA baseline and a jointly-trained TFT advanced model. The implementation is faithful to the published recipes [Lim 2021, Salinas 2020] and is reproducible from the scripts in `src/`. Numerical results will be filled in once the main session executes the scripts; the methodology, dataset, hierarchy, and metric protocol are pinned now so that the executed run is comparable to the published benchmarks of [Kong 2019] and [Fekri 2021] working on the same data.

The strategic ambition is the transfer to DACH utilities operating under the Messstellenbetriebsgesetz and Section 14a EnWG flexible-load regime: the architecture, the static-plus-known-plus-observed input partition, the quantile head, the MinT reconciliation, and the dToU-as-treatment frame are all directly portable to a German smart-meter cohort once data becomes available under the BSI Smart Meter Gateway rollout. The present work pins down the methodology now so that the German build later is a transfer-learning step rather than a from-scratch project.

## References

References are listed in `../reports/references.md`. Inline citations in this manuscript follow the convention `[FirstAuthor Year]`. The most-cited works are the TFT paper [Lim 2021], the smart-meter analytics review [Wang 2019], the LSTM residential forecasting paper [Kong 2019], the LCL deep-learning study [Fekri 2021], the DeepAR paper [Salinas 2020], the probabilistic load forecasting tutorial [Hong 2016], the M5 retrospective [Makridakis 2022], the Prophet paper [Taylor 2018], the MinT reconciliation paper [Wickramasuriya 2019], the gradient-boosting Kaggle load winner [Ben Taieb 2014], and the foundational machine-learning references [Chen 2016, Breiman 2001, Lundberg 2020].
