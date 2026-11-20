# Data Quality & Exploration Audit Report

## 1. Overview
- **Training dataset**: `train-test.csv` (48,000 loads, Jan 1, 2025 to Oct 31, 2025)
- **Validation dataset**: `validation.csv` (12,000 loads, Nov 1, 2025 to Dec 31, 2025)
- **Target**: `posted_rate` (USD)
- **Time sequence**: Dates are monotonically increasing throughout `train-test.csv`.

---

## 2. Missing Values Analysis
| Feature | Train Missing (Count / %) | Validation Missing (Count / %) | Nature & Imputation Strategy |
|---|---|---|---|
| `weight` | 300 / 0.625% | 165 / 1.375% | Missing completely at random (MCAR). Impute with equipment-specific median. |
| `market_index` | 374 / 0.779% | 249 / 2.075% | Continuous market sentiment index. Impute using rolling 7-day mean or date-matched median. |
| `quote_signal` | 0 / 0.0% | 0 / 0.0% | 100% complete across both splits. |
| `distance` | 0 / 0.0% | 0 / 0.0% | 100% complete across both splits. |
| Coordinates & Categories | 0 / 0.0% | 0 / 0.0% | 100% complete. |

---

## 3. Geographic & Route Coordinate Audit
- **City Coordinate Consistency**: 
  - There are exactly 64 unique pickup cities and 64 unique delivery cities.
  - Every city has a **single, perfectly identical (lat, lon)** across all rows in both pickup and delivery contexts.
  - Lexington: `(36.99152, -84.99876)`
  - Fort Wayne: `(41.31561, -85.36206)`
- **Route Geometry (Distance vs. Straight-Line Haversine)**:
  - Minimum circuity ratio (`distance / straight_dist`): **1.058** (no negative or impossible shortcuts).
  - Median circuity ratio: **1.182** (consistent with US highway network circuity of ~1.18–1.20).
  - Circuity > 2.0: Only **22 loads** out of 48,000 (< 0.05%), indicating highly reliable road network distance values.

---

## 4. Target & Rate-Per-Mile (RPM) Audit
- **Baseline Formula Discovery**:
  - The feature `quote_signal` represents the expected rate-per-mile baseline.
  - The product `base_quote = distance * quote_signal` correlates strongly with `posted_rate` ($r = 0.8987$).
  - The ratio `quote_ratio = posted_rate / (distance * quote_signal)` has an exact median of **1.0040** across all 48,000 loads.
- **Anomalies and Extreme Residuals**:
  - `quote_ratio > 3.0`: 334 loads (0.69%) where posted rate is 3× to 11× the baseline quote.
  - `quote_ratio < 0.3`: 169 loads (0.35%) where posted rate is < 0.3× the baseline quote.
  - These heavy tails explain why RMSE remained stubbornly high (~$635) across baseline models while MAE improved.
  - Training remediation: Target reformulation to rate-per-mile or quote ratio with robust loss functions (L1 / MAE) and Huber loss, preventing extreme outliers from distorting model gradients.

---

## 5. Categorical Integrity
- Equipment values are clean and uniform across both datasets:
  - `Dry Van`: 56.67%
  - `Reefer`: 25.10%
  - `Flatbed`: 18.23%
- City names have zero trailing/leading whitespace and consistent casing.
- Duplicate rows: Exactly **0 duplicate loads** in either train or validation data.
