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
