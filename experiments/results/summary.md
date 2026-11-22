# Expanding-window CV — MAE ($) per test month

| experiment                                                   |    Jun |    Jul |    Aug |    Sep |    Oct |   Mean MAE |   Mean bias |
|:-------------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|-----------:|------------:|
| XGBoost + quote_signal + market_index                        | 135.5  |  91.61 | 170.3  | 108.3  | 122.74 |     125.69 |       -67.6 |
| LightGBM + quote_signal + market_index (previous approach) | 132.28 |  93.92 | 168.22 | 109.61 | 129.32 |     126.67 |       -73.1 |
| Blend: XGB + LGB + CatBoost, core features (FINAL)                  | 222.82 | 128.68 |  91.33 |  95.59 | 100.69 |     127.82 |       -80.2 |
| XGBoost + market_index (no quote_signal)                     | 141.8  | 100.01 | 117.08 | 152.77 | 129.57 |     128.25 |      -100.4 |
| XGBoost, core features                               | 218.73 | 129.2  |  94.08 |  97.42 | 102.48 |     128.38 |       -79.6 |
| XGBoost, core features, no lane priors                       | 221.31 | 129.09 |  92.78 |  97.25 | 102.12 |     128.51 |       -80.4 |
| CatBoost, core features                                      | 225.99 | 129.57 |  91.38 |  95.85 | 100.99 |     128.76 |       -80.9 |
| LightGBM, core features                                      | 224.19 | 129.39 |  93.59 |  97.01 | 101.9  |     129.22 |       -80.1 |
| XGBoost + month-relative market_index                        | 221.69 | 119.64 |  95.64 | 105.7  | 103.61 |     129.26 |       -78.8 |
| Baseline: lane median RPM x distance                         | 206.66 | 190.75 | 216.69 | 199.18 | 201.47 |     202.95 |         0.6 |

Bias = mean(prediction - actual). It is negative for every L1 model because the MAE-optimal prediction is the conditional *median*, and the corrupted labels are mostly far above it (2-5x lane rate).
