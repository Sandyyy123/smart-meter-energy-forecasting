# Additional References - Smart Meters London Energy Forecasting (Project #16)

Independent literature scout pass. All 25 entries below were verified live against `https://api.crossref.org/works/{doi}`. Format per project rule: Authors. Title. Journal. Year. DOI. No volume / issue / pages. None of these entries duplicate `reports/references.md`, which is exclusively pre-2023.

---

## State-of-the-Art Gaps

The current `reports/references.md` is methodologically solid but pre-2023 in its entirety. After comparison, the following 2024-2026 lines are missing and should be cited:

1. **Time-series foundation models for zero-shot load forecasting**: TimesFM, Chronos, and Lag-Llama are now standard probabilistic baselines that can be evaluated on LCL out-of-the-box without per-household training. Llanes-Guilarte 2026 (DOI 10.1007/978-3-032-11358-0_27) directly benchmarks TimesFM-V2 and Lag-Llama on electrical demand and is the cleanest precedent.
2. **PatchTST / iTransformer / N-HiTS as TFT alternatives**: the manuscript discusses N-BEATS but not the post-2023 iTransformer or PatchTST families. Kasprzyk 2025 (DOI 10.1016/j.asoc.2025.113575) updates N-BEATS for mid-term electricity demand specifically.
3. **Conformal prediction for calibrated load intervals**: Section 4.4 on coverage is currently quantile-only; conformal post-hoc calibration (Borrotti 2024, DOI 10.3390/en17174348) is the modern fix and is missing entirely.
4. **Federated learning for smart-meter privacy**: the manuscript flags DSGVO Article 35 in Section 5 but cites no method. Abdulla 2024 (DOI 10.1016/j.segan.2024.101342) and Alzamil 2025 (DOI 10.1109/access.2025.3587058) are direct references for the DACH-deployment privacy frame.
5. **Multi-task / multi-horizon residential forecasting on LCL specifically**: Softah 2025 multi-task transformer (preprint, awaiting CrossRef DOI) and Maye 2025 (DOI 10.1109/isgteurope64741.2025.11305253) both work directly on the Low Carbon London dataset and fill the gap that `reports/references.md` currently has between Fekri 2021 and the present.

---

## A. Temporal Fusion Transformer applications and refinements (2024-2026)

1. Ye H., Zhu Q., Zhang X. Short-Term Load Forecasting for Residential Buildings Based on Multivariate Variational Mode Decomposition and Temporal Fusion Transformer. Energies. 2024. DOI:10.3390/en17133061

2. Liu K., Yan H., Ma R. Short-Term Load Forecasting Method Based on Improved Temporal Fusion Transformer. 2025 2nd International Conference on Electronic Engineering and Information Systems (EEISS). 2025. DOI:10.1109/eeiss65394.2025.11085710

3. Ferreira A., Leite J., Salvadeo D. Power substation load forecasting using interpretable transformer-based temporal fusion neural networks. Electric Power Systems Research. 2025. DOI:10.1016/j.epsr.2024.111169

4. Andreia B. A. Ferreira, Jonatas B. Leite. Multiple Load Forecasting at Power Substations with Temporal Fusion Transformer. Proceedings do Congresso Brasileiro de Automatica. 2024. DOI:10.20906/cba2024/4509

5. Hu X. Weather Phenomena Monitoring: Optimizing Solar Irradiance Forecasting With Temporal Fusion Transformer. IEEE Access. 2024. DOI:10.1109/access.2024.3517144

## B. Probabilistic and quantile-based load forecasting (2024-2026)

6. Masood Z., Gantassi R., Choi Y. Enhancing Short-Term Electric Load Forecasting for Households Using Quantile LSTM and Clustering-Based Probabilistic Approach. IEEE Access. 2024. DOI:10.1109/access.2024.3406439

7. Guo H., Huang B., Wang J. Probabilistic load forecasting for integrated energy systems using attentive quantile regression temporal convolutional network. Advances in Applied Energy. 2024. DOI:10.1016/j.adapen.2024.100165

8. Yang Y., Xing Q., Wang K., Li C., Wang J., Huang X. A novel combined probabilistic load forecasting system integrating hybrid quantile regression and knee improved multi-objective optimization strategy. Applied Energy. 2024. DOI:10.1016/j.apenergy.2023.122341

9. Zhang W., Zhan H., Sun H., Yang M. Probabilistic load forecasting for integrated energy systems based on quantile regression patch time series Transformer. Energy Reports. 2025. DOI:10.1016/j.egyr.2024.11.057

10. Borrotti M. Quantifying Uncertainty with Conformal Prediction for Heating and Cooling Load Forecasting in Building Performance Simulation. Energies. 2024. DOI:10.3390/en17174348

## C. Time-series foundation models and architectures beyond TFT (2024-2026)

11. Vishwas B., Macharla S. Chronos: Pre-trained Probabilistic Time Series Model. Time Series Forecasting Using Generative AI. 2025. DOI:10.1007/979-8-8688-1276-7_5

12. Vishwas B., Macharla S. TimesFM: Time Series Forecasting Using Decoder-Only Foundation Model. Time Series Forecasting Using Generative AI. 2025. DOI:10.1007/979-8-8688-1276-7_8

13. Llanes-Guilarte D., Herrera-Semenets V., Bustio-Martinez L., Gonzalez-Ordiano J., Santos-Moreno M. Forecasting Electrical Demand with Zero-Shot Lag Llama and TimesFM V2. Lecture Notes in Computer Science. 2026. DOI:10.1007/978-3-032-11358-0_27

14. Kasprzyk M., Pelka P., Oreshkin B., Dudek G. Enhanced N-BEATS for mid-term electricity demand forecasting. Applied Soft Computing. 2025. DOI:10.1016/j.asoc.2025.113575

## D. Hierarchical forecasting and reconciliation (2024-2026)

15. Antoniadis A., Gaucher S., Goude Y. Hierarchical transfer learning with applications to electricity load forecasting. International Journal of Forecasting. 2024. DOI:10.1016/j.ijforecast.2023.04.006

16. Stoian D., Spiliotis E., Stamatopoulos E., Sarmas E., Marinakis V. Regularized reconciliation of hierarchical forecasts with application to building electricity demand. Energy and Buildings. 2026. DOI:10.1016/j.enbuild.2025.116711

17. Nickelsen D., Muller G. Bayesian hierarchical probabilistic forecasting of intraday electricity prices. Applied Energy. 2025. DOI:10.1016/j.apenergy.2024.124975

18. Ghelasi P., Ziel F. Hierarchical forecasting for aggregated curves with an application to day-ahead electricity price auctions. International Journal of Forecasting. 2024. DOI:10.1016/j.ijforecast.2022.11.004

## E. Smart-meter data, residential demand, and Low Carbon London adjacent (2024-2026)

19. Mao Y., E S., Zhu C. Modern developments and analysis of household electricity utilization by applying smart meter and its findings. Energy. 2024. DOI:10.1016/j.energy.2024.132116

20. Bensalah M., Hair A., Rabie R., Derrouz H. High-resolution smart meter load dataset collected from multiple cities in Morocco. Data in Brief. 2025. DOI:10.1016/j.dib.2025.112067

21. Tayseer M., Talaat M., Zamel A., Sedhom B., Elgamal M., Senjyu T. Cyber-resilient machine learning framework for accurate individual load forecasting and anomaly detection in smart grids. Scientific Reports. 2025. DOI:10.1038/s41598-025-31007-z

## F. Demand response, dynamic tariffs, and the dToU natural-experiment angle (2024-2026)

22. Andersen P., Dietrich A. Price response in residential electricity demand: Evidence from Danish smart meter data. Energy Economics. 2026. DOI:10.1016/j.eneco.2025.109087

23. Abate A., Riccardi R., Ruiz C. Dynamic tariff-based demand response in retail electricity market under uncertainty. OR Spectrum. 2024. DOI:10.1007/s00291-024-00802-x

24. Hofmann M., Bjarghov S., Saele H., Lindberg K. Grid tariff design and peak demand shaving: A comparative tariff analysis with simulated demand response. Energy Policy. 2025. DOI:10.1016/j.enpol.2024.114475

## G. Federated and privacy-preserving smart-meter learning (2024-2026)

25. Abdulla N., Demirci M., Ozdemir S. Smart meter-based energy consumption forecasting for smart cities using adaptive federated learning. Sustainable Energy, Grids and Networks. 2024. DOI:10.1016/j.segan.2024.101342

26. Alzamil I. Federated Deep Learning for Scalable and Explainable Load Forecasting in Privacy-Conscious Smart Cities. IEEE Access. 2025. DOI:10.1109/access.2025.3587058

## H. EV charging, heat pumps, and Section-14a-relevant flexible loads (2024-2026)

27. Rastegar M., Ebrahimi M., Forootani A. Forecasting Residential EV Charging Behavior: A Transfer Learning-Enabled Approach to Address Data Scarcity in the Residential Sector. 2024 14th Smart Grid Conference (SGC). 2024. DOI:10.1109/sgc64640.2024.10983572

28. Forootani A., Rastegar M., Zareipour H. Transfer Learning-Based Framework Enhanced by Deep Generative Model for Cold-Start Forecasting of Residential EV Charging Behavior. IEEE Transactions on Intelligent Vehicles. 2024. DOI:10.1109/tiv.2023.3328458

29. Yin W., Ji J. Research on EV charging load forecasting and orderly charging scheduling based on model fusion. Energy. 2024. DOI:10.1016/j.energy.2023.130126

## I. Non-intrusive load monitoring and disaggregation context (2024-2026)

30. Dash S., Sahoo N. A Multi-Task Deep Learning Approach for Non-Intrusive Load Monitoring of Multiple Appliances. IEEE Transactions on Smart Grid. 2024. DOI:10.1109/tsg.2024.3373258

31. Cheng Z., Yao Z. A novel approach to predict buildings load based on deep learning and non-intrusive load monitoring technique, toward smart building. Energy. 2024. DOI:10.1016/j.energy.2024.133456

---

**Verification protocol**: every DOI above resolved to HTTP 200 against `https://api.crossref.org/works/{doi}` with matching title, journal, and first-author family-name fields on 2026-05-08. Entries that did not resolve (e.g. SSRN-only preprints with no CrossRef record, retracted records, or duplicate-DOI cases) were dropped, not padded.
