# References - Smart Meter Energy Forecasting

Each entry below has been verified live against CrossRef (https://api.crossref.org/works/{doi}) or, for arXiv preprints, against Semantic Scholar (https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}) at scaffolding time. Per project rule: author / title / journal / year / identifier kept; volume, issue, and page numbers omitted.

## A. Time series and multi-horizon forecasting methods

1. **Lim, B., Arik, S. O., Loeff, N., & Pfister, T.** (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012). The architecture used in `src/model_advanced.py`: variable selection networks per static, known-future, and observed input; LSTM encoder-decoder; multi-head attention for long-range patterns; quantile regression head.

2. **Lim, B., Arik, S. O., Loeff, N., & Pfister, T.** (2019). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. arXiv:[1912.09363](https://arxiv.org/abs/1912.09363). Original preprint with the full ablation study used to motivate the gating, attention head count, and quantile output choices in our advanced model.

3. **Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T.** (2020). DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001). Probabilistic LSTM-based global model for related time series; the canonical alternative to TFT for household-level forecasting and a contender for the advanced model.

4. **Flunkert, V., Salinas, D., & Gasthaus, J.** (2017). DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. arXiv:[1704.04110](https://arxiv.org/abs/1704.04110). Original DeepAR preprint; introduces the negative-log-likelihood loss with parametric output distributions used downstream by GluonTS.

5. **Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y.** (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. arXiv:[1905.10437](https://arxiv.org/abs/1905.10437). Pure-MLP architecture with stacked basis expansion; M4 winner among neural methods, included as a baseline contender.

6. **Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., & Sun, L.** (2022). Transformers in Time Series: A Survey. arXiv:[2202.07125](https://arxiv.org/abs/2202.07125). Taxonomy of transformer architectures for time-series forecasting; positions TFT, Informer, Autoformer, and Pyraformer.

7. **Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., et al.** (2019). GluonTS: Probabilistic Time Series Models in Python. arXiv:[1906.05264](https://arxiv.org/abs/1906.05264). Library reference for probabilistic forecasting; cited as an alternative implementation route.

8. **Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., & Yan, X.** (2019). Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting. arXiv:[1907.00235](https://arxiv.org/abs/1907.00235). LogSparse Transformer; introduces convolutional self-attention and complexity reduction useful when scaling TFT to many households.

9. **Taylor, S. J., & Letham, B.** (2018). Forecasting at Scale. *The American Statistician*. DOI: [10.1080/00031305.2017.1380080](https://doi.org/10.1080/00031305.2017.1380080). The Prophet model used as one of the baseline-script options; decomposable trend, seasonality, and holiday effects with Stan posterior estimation.

10. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2020). The M4 Competition: 100,000 Time Series and 61 Forecasting Methods. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014). Benchmark competition; supports the methodological choice of hybrid statistical-plus-ML approaches.

11. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2022). M5 Accuracy Competition: Results, Findings, and Conclusions. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Hierarchical, intermittent retail forecasting; the LightGBM-dominant winners directly inform the advanced-model design choices for hierarchical reconciliation across ACORN levels.

12. **Hong, T., & Fan, S.** (2016). Probabilistic Electric Load Forecasting: A Tutorial Review. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2015.11.011](https://doi.org/10.1016/j.ijforecast.2015.11.011). Authoritative review of probabilistic load forecasting; methodological reference for the quantile-output choice in the TFT model.

13. **Ben Taieb, S., & Hyndman, R. J.** (2014). A Gradient Boosting Approach to the Kaggle Load Forecasting Competition. *International Journal of Forecasting*. DOI: [10.1016/j.ijforecast.2013.07.005](https://doi.org/10.1016/j.ijforecast.2013.07.005). Gradient boosting on hourly load with weather and calendar covariates; precedent for the gradient-boosted-tree comparison candidate in the baseline.

## B. Hierarchical forecasting and reconciliation

14. **Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J.** (2019). Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series through Trace Minimization. *Journal of the American Statistical Association*. DOI: [10.1080/01621459.2018.1448825](https://doi.org/10.1080/01621459.2018.1448825). Minimum-trace (MinT) reconciliation; the canonical method for reconciling household, ACORN-cluster, and feeder-level forecasts in the discussion.

## C. Smart-meter analytics and household load forecasting

15. **Wang, Y., Chen, Q., Hong, T., & Kang, C.** (2019). Review of Smart Meter Data Analytics: Applications, Methodologies, and Challenges. *IEEE Transactions on Smart Grid*. DOI: [10.1109/TSG.2018.2818167](https://doi.org/10.1109/TSG.2018.2818167). Foundational review covering load forecasting, clustering, anomaly detection, and theft detection on smart-meter data; used to position the project in domain context.

16. **Kong, W., Dong, Z. Y., Jia, Y., Hill, D. J., Xu, Y., & Zhang, Y.** (2019). Short-Term Residential Load Forecasting Based on LSTM Recurrent Neural Network. *IEEE Transactions on Smart Grid*. DOI: [10.1109/TSG.2017.2753802](https://doi.org/10.1109/TSG.2017.2753802). Per-household LSTM forecasting on UK and Australian smart meters; direct precedent for our baseline.

17. **Shi, H., Xu, M., & Li, R.** (2018). Deep Learning for Household Load Forecasting: A Novel Pooling Deep RNN. *IEEE Transactions on Smart Grid*. DOI: [10.1109/TSG.2017.2686012](https://doi.org/10.1109/TSG.2017.2686012). Pooling RNN that explicitly addresses overfitting on noisy household-level series, which is the main technical risk for our 5,567-household setup.

18. **Bouktif, S., Fiaz, A., Ouni, A., & Serhani, M. A.** (2018). Optimal Deep Learning LSTM Model for Electric Load Forecasting using Feature Selection and Genetic Algorithm. *Energies*. DOI: [10.3390/en11071636](https://doi.org/10.3390/en11071636). Hyperparameter search over LSTM architectures for medium-term load; informs the search space for the TFT hidden-size and dropout settings.

19. **Fekri, M. N., Patel, H., Grolinger, K., & Sharma, V.** (2021). Deep Learning for Load Forecasting with Smart Meter Data: Online Adaptive Recurrent Neural Network. *Applied Energy*. DOI: [10.1016/j.apenergy.2020.116177](https://doi.org/10.1016/j.apenergy.2020.116177). Online learning on the same Low Carbon London (smart meters London) dataset; reports MAPE ranges that we use to set realistic expectations for the baseline.

20. **Massaoudi, M., Refaat, S. S., Chihi, I., Trabelsi, M., Oueslati, F. S., & Abu-Rub, H.** (2021). A Novel Stacked Generalization Ensemble-Based Hybrid LGBM-XGB-MLP Model for Short-Term Load Forecasting. *Energy*. DOI: [10.1016/j.energy.2020.118874](https://doi.org/10.1016/j.energy.2020.118874). Stacked-ensemble baseline; benchmark for the gradient-boosted comparator we plan to add alongside SARIMA.

21. **Somu, N., M R, G. R., & Ramamritham, K.** (2020). A Hybrid Model for Building Energy Consumption Forecasting using Long Short Term Memory Networks. *Applied Energy*. DOI: [10.1016/j.apenergy.2019.114131](https://doi.org/10.1016/j.apenergy.2019.114131). Hybrid CNN-LSTM on building energy series; informs the choice of convolutional pre-encoders for the advanced-model variant.

22. **Ahmad, M. W., Mourshed, M., & Rezgui, Y.** (2017). Trees vs Neurons: Comparison Between Random Forest and ANN for High-Resolution Prediction of Building Energy Consumption. *Energy and Buildings*. DOI: [10.1016/j.enbuild.2017.04.038](https://doi.org/10.1016/j.enbuild.2017.04.038). Side-by-side RF vs ANN benchmark; supports the use of an RF-style baseline as a tabular-feature comparator in the discussion.

23. **Rahman, A., Srikumar, V., & Smith, A. D.** (2018). Predicting Electricity Consumption for Commercial and Residential Buildings using Deep Recurrent Neural Networks. *Applied Energy*. DOI: [10.1016/j.apenergy.2017.12.051](https://doi.org/10.1016/j.apenergy.2017.12.051). Deep recurrent baseline on building-level series with weather covariates; confirms the value of joint weather plus history inputs.

24. **Antonopoulos, I., Robu, V., Couraud, B., Kirli, D., Norbu, S., Kiprakis, A., Flynn, D., & Elizondo-Gonzalez, S.** (2020). Artificial Intelligence and Machine Learning Approaches to Energy Demand-Side Response: A Systematic Review. *Renewable and Sustainable Energy Reviews*. DOI: [10.1016/j.rser.2020.109899](https://doi.org/10.1016/j.rser.2020.109899). Systematic review of demand-side ML; positions the dToU subset of the dataset as a natural experiment.

## D. Machine-learning method foundations

25. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *Proc. 22nd ACM SIGKDD*. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785). Regularised, sparsity-aware gradient boosting; reference for the LightGBM/XGBoost comparator on lagged-feature tabular forecasting.

26. **Breiman, L.** (2001). Random Forests. *Machine Learning*. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324). Bagged decision-tree ensemble; included as a non-parametric baseline in the discussion.

27. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., et al.** (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*. DOI: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9). Polynomial-time exact SHAP for tree ensembles; used to interpret the gradient-boosted comparator and contrasted with TFT's variable-selection-network attribution.


---

## 2024-2026 additions (post-QA literature scout)

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

