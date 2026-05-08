# Validation Report - Project #16 (Smart Meters London)

**Role:** A (Validator)
**Project:** 16_smart_meters - Multi-Horizon Probabilistic Forecasting of Household Electricity Consumption (Low Carbon London)
**Phase:** Scaffold-only (Phase 1)
**Overall:** PASS-WITH-WARNINGS

## Compact Summary (under 150 words)

All structural artefacts present and parseable: notebook JSON valid, both Python scripts compile, manuscript follows IMRaD with all 8 expected sections, all 22 inline citations resolve to entries in `reports/references.md`, em-dash count is zero across the seven scanned files, no AI-tell phrases detected, and 5/5 sampled DOIs resolve live on CrossRef with matching titles. Methods listed in manuscript section 3 (SARIMA, Prophet, yearly Fourier, TFT with QuantileLoss, pytorch-forecasting) all appear in `src/model_baseline.py` or `src/model_advanced.py`. Two non-blocking warnings: (1) manuscript word count is 4133 vs. target 4000-5000 (inside band, but on the lower edge - flagged for visibility); (2) presentation HTML contains 5 outbound `https://doi.org/...` href links - inline-only deliverable contract is met (no remote CSS/JS/images), but DOI links technically count as external `href="http`.

---

## Findings (one per line)

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as JSON via `json.load`.

### 2. Python script syntax
- [PASS] `src/model_baseline.py` passes `ast.parse` without errors.
- [PASS] `src/model_advanced.py` passes `ast.parse` without errors.

### 3. Manuscript word count
- [WARN] `wc -w manuscripts/manuscript.md` returned 4133 words. Target band is 4000-5000; value is inside the band but only 133 words above the floor. Recommend expanding Discussion subsections (esp. ablation sketches, hierarchical reconciliation) for safety margin.

### 4. Self-contained HTML
- [WARN] `grep -E 'href="http|src="http' deliverables/presentation.html` returned 5 hits. All 5 are `https://doi.org/...` reference DOI hyperlinks, no remote CSS/JS/images/fonts/CDN. The presentation is functionally self-contained for offline viewing (no rendering breaks without network); DOI links are semantic-only outbound references. Strict reading of "0 hits" rule = WARN; inline-resource compliance = PASS.

### 5. IMRaD completeness
- [PASS] Manuscript contains all required sections: Title (line 1), Abstract, Introduction, Methods (3), Results (4), Discussion (5), Conclusion (6), References. Section 2 (Data) is an additional split of the Methods/Materials block, which is standard IMRaD-extended practice.

### 6. Method drift
- [PASS] Methods named in `## 3. Methods` all map to code:
  - SARIMA(1,1,1)(1,1,1,7), Prophet, yearly Fourier basis -> `src/model_baseline.py` (functions `fit_sarima`, `fit_prophet`, `yearly_fourier`).
  - Temporal Fusion Transformer, QuantileLoss(quantiles=[0.1, 0.5, 0.9]), pytorch-forecasting `TimeSeriesDataSet`, hidden_size=64, attention_head_size=4, dropout=0.2, lr=3e-4 -> `src/model_advanced.py` (function `build_tft`).
  - Metrics named (MAPE, MAE, RMSE, SMAPE, pinball/quantile loss, q10/q90 coverage) all implemented or returned by the metrics dicts in both scripts.
  - No drift detected.

### 7. Citation drift
- [PASS] 22 distinct inline `[Author Year]` tokens extracted from manuscript: Alexandrov 2019, Ben Taieb 2014, Bouktif 2018, Breiman 2001, Chen 2016, Fekri 2021, Hong 2016, Kong 2019, Li 2019, Lim 2021, Lundberg 2020, Makridakis 2022, Massaoudi 2021, Oreshkin 2019, Rahman 2018, Salinas 2020, Shi 2018, Somu 2020, Taylor 2018, Wang 2019, Wen 2022, Wickramasuriya 2019. Each maps to an entry in `reports/references.md` (matched by surname proximity to year). Zero orphan citations.

### 8. Re-verify 5 random references (live CrossRef)
- [PASS] 10.1016/j.ijforecast.2021.03.012 -> HTTP 200, title "Temporal Fusion Transformers for interpretable multi-horizon time series..." (2021). Match.
- [PASS] 10.1016/j.ijforecast.2019.07.001 -> HTTP 200, title "DeepAR: Probabilistic forecasting with autoregressive recurrent networks" (2020 publication, 2019 ref tag is preprint year - acceptable). Match.
- [PASS] 10.1109/TSG.2017.2753802 -> HTTP 200, title "Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural..." (2019). Match.
- [PASS] 10.1080/01621459.2018.1448825 -> HTTP 200, title "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series..." (2018; ref lists 2019 publication year - JASA assigned-issue lag, acceptable). Match.
- [PASS] 10.1016/j.apenergy.2020.116177 -> HTTP 200, title "Deep learning for load forecasting with smart meter data: Online Adaptive..." (2021 publication, 2021 ref tag matches). Match.

### 9. Em-dash scan
- [PASS] Total em-dash (U+2014) count across `brief.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, both `src/*.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html` = 0.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` over the project folder returned no hits.

### 11. Checkpoint schema
- [PASS] `checkpoint.json` keys: `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`. All four required fields present (`project_number`, `title`, `methodology`, `status`). `phase` ("Phase 1 - scaffolded code-only") is the canonical equivalent of `status`-style metadata and is harmless. No missing fields.

### Bonus: scaffold-completeness
- [PASS] All scaffold artefacts listed in `brief.md` Deliverables block are present on disk: `brief.md`, `data/README.md`, `notebooks/01_EDA.ipynb`, `reports/references.md`, `src/model_baseline.py`, `src/model_advanced.py`, `manuscripts/manuscript.md`, `deliverables/presentation.html`, `checkpoint.json`.
- [PASS] Per QA rules, projects #1-#8 may have saved model artefacts under `deliverables/`. Project #16 is in the #9-#21 scaffold-only bucket; absence of `.pkl`/`.ckpt`/`.png` is expected and not flagged. `deliverables/` currently contains only `presentation.html`, consistent with scaffold phase.

---

## Blockers
None. All checks completed within a single tool run; CrossRef API was reachable.

## One-line status
Role A (Validator) complete - PASS-WITH-WARNINGS (manuscript word count near floor, HTML carries 5 DOI outbound href links; both non-blocking).
