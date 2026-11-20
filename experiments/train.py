"""
Freight Rate Prediction — Full Training Pipeline
Iterations: Baseline (Ridge) → LightGBM → XGBoost → Ensemble
Tracks all experiment results and writes outputs for scoring.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train-test.csv"
VAL_CSV = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DEC_CSV = ROOT / "december-chart-inputs.csv"
OUT_PRED = ROOT / "validation_predictions.csv"
RESULTS_JSON = ROOT / "experiment_results.json"


# ── Helpers ──────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"label": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 6)}


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    return train, val


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, fit_encoders: dict | None = None,
                       global_stats: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Date features ──
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["days_since_start"] = (df["date"] - pd.Timestamp("2025-01-01")).dt.days

    # ── Fill nulls using global stats (fit on train only) ──
    gstats = global_stats or {}
    for col in ["weight", "market_index", "quote_signal"]:
        if col not in gstats:
            gstats[col] = df[col].median()
        df[col] = df[col].fillna(gstats[col])

    # ── Lane feature ──
    df["lane"] = df["pickup"] + " → " + df["delivery"]

    # ── Interaction features ──
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["dist_x_quote"] = df["distance"] * df["quote_signal"]
    df["market_x_quote"] = df["market_index"] * df["quote_signal"]
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1)
    df["distance_sq"] = df["distance"] ** 2
    df["log_distance"] = np.log1p(df["distance"])
    df["log_weight"] = np.log1p(df["weight"])

    # ── Haversine distance (more accurate than raw distance col) ──
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # ── Label encode categoricals ──
    encoders = fit_encoders or {}
    for col in ["equipment", "pickup", "delivery", "lane"]:
        if col not in encoders:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col + "_enc"] = df[col].astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in known else -1
            )

    return df, encoders, gstats


FEATURE_COLS = [
    "distance", "log_distance", "haversine_km",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal",
    "dist_x_market", "dist_x_quote", "market_x_quote",
    "day_of_week", "month", "week_of_year", "day_of_year",
    "is_weekend", "quarter", "days_since_start",
    "equipment_enc", "pickup_enc", "delivery_enc", "lane_enc",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]

SIMPLE_COLS = [
    "distance", "log_distance", "weight", "market_index", "quote_signal",
    "dist_x_market", "dist_x_quote", "market_x_quote",
    "equipment_enc", "day_of_week", "month", "is_weekend", "haversine_km",
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = []
    print("=" * 60)
    print("FREIGHT RATE PREDICTION — EXPERIMENT TRACKER")
    print("=" * 60)

    train_raw, val_raw = load_data()
    print(f"\nTrain: {len(train_raw)} rows | Val: {len(val_raw)} rows")

    # Engineer features (fit stats/encoders on train only)
    train_fe, encoders, gstats = engineer_features(train_raw)
    val_fe, _, _ = engineer_features(val_raw, fit_encoders=encoders, global_stats=gstats)

    X_train = train_fe[FEATURE_COLS]
    y_train = train_fe["posted_rate"]
    X_val = val_fe[FEATURE_COLS]

    # Time-based CV split (last 20% of train = ~Sept-Oct 2025)
    n = len(X_train)
    cv_cutoff = int(n * 0.80)
    X_cv_tr = X_train.iloc[:cv_cutoff]
    X_cv_vl = X_train.iloc[cv_cutoff:]
    y_cv_tr = y_train.iloc[:cv_cutoff]
    y_cv_vl = y_train.iloc[cv_cutoff:]
    print(f"  CV split: {cv_cutoff} train / {n - cv_cutoff} val")

    # ── ITERATION 1: Ridge Baseline ───────────────────────────────────────────
    print("\n--- Iteration 1: Ridge Baseline ---")
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    ridge_pipe.fit(X_cv_tr[SIMPLE_COLS], y_cv_tr)
    r1_pred = ridge_pipe.predict(X_cv_vl[SIMPLE_COLS])
    r1 = metrics(y_cv_vl, r1_pred, "Ridge CV")
    results.append(r1)

    # Refit on full train
    ridge_full = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    ridge_full.fit(X_train[SIMPLE_COLS], y_train)
    ridge_val_pred = ridge_full.predict(X_val[SIMPLE_COLS])

    # ── ITERATION 2: LightGBM ────────────────────────────────────────────────
    print("\n--- Iteration 2: LightGBM ---")
    lgb_params = {
        "objective": "regression_l1",   # optimize MAE directly
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 15,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    lgb_cv = lgb.LGBMRegressor(**lgb_params)
    lgb_cv.fit(
        X_cv_tr[FEATURE_COLS], y_cv_tr,
        eval_set=[(X_cv_vl[FEATURE_COLS], y_cv_vl)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    lgb_cv_pred = lgb_cv.predict(X_cv_vl[FEATURE_COLS])
    r2 = metrics(y_cv_vl, lgb_cv_pred, "LightGBM CV")
    results.append(r2)
    best_lgb_iter = lgb_cv.best_iteration_
    print(f"  LightGBM best iteration: {best_lgb_iter}")

    lgb_full = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": best_lgb_iter + 50})
    lgb_full.fit(X_train[FEATURE_COLS], y_train)
    lgb_val_pred = lgb_full.predict(X_val[FEATURE_COLS])

    # ── ITERATION 3: XGBoost ─────────────────────────────────────────────────
    print("\n--- Iteration 3: XGBoost ---")
    xgb_params = {
        "objective": "reg:absoluteerror",   # MAE objective
        "eval_metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }

    xgb_cv = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    xgb_cv.fit(
        X_cv_tr[FEATURE_COLS], y_cv_tr,
        eval_set=[(X_cv_vl[FEATURE_COLS], y_cv_vl)],
        verbose=False,
    )
    xgb_cv_pred = xgb_cv.predict(X_cv_vl[FEATURE_COLS])
    r3 = metrics(y_cv_vl, xgb_cv_pred, "XGBoost CV")
    results.append(r3)
    best_xgb_iter = xgb_cv.best_iteration
    print(f"  XGBoost best iteration: {best_xgb_iter}")

    xgb_full = xgb.XGBRegressor(**{**xgb_params, "n_estimators": best_xgb_iter + 50})
    xgb_full.fit(X_train[FEATURE_COLS], y_train)
    xgb_val_pred = xgb_full.predict(X_val[FEATURE_COLS])

    # ── ITERATION 4: Optimal Ensemble ────────────────────────────────────────
    print("\n--- Iteration 4: Weighted Ensemble ---")
    best_w, best_mae_ens = 0.5, float("inf")
    for w in np.arange(0.3, 0.8, 0.05):
        blend = w * lgb_cv_pred + (1 - w) * xgb_cv_pred
        m = mean_absolute_error(y_cv_vl, blend)
        if m < best_mae_ens:
            best_mae_ens = m
            best_w = w
    print(f"  Optimal LGB weight: {best_w:.2f}")
    ensemble_cv_pred = best_w * lgb_cv_pred + (1 - best_w) * xgb_cv_pred
    r4 = metrics(y_cv_vl, ensemble_cv_pred, f"Ensemble CV (LGB={best_w:.2f})")
    results.append(r4)
    ensemble_val_pred = best_w * lgb_val_pred + (1 - best_w) * xgb_val_pred

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:<40} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R²={r['R2']:.4f}")

    # Pick best by CV MAE
    cv_maes = {r["label"]: r["MAE"] for r in results}
    best_label = min(cv_maes, key=cv_maes.get)
    print(f"\n  → Best: {best_label}")

    pred_map = {
        results[0]["label"]: ridge_val_pred,
        results[1]["label"]: lgb_val_pred,
        results[2]["label"]: xgb_val_pred,
        results[3]["label"]: ensemble_val_pred,
    }
    final_val_pred = np.clip(pred_map[best_label], 50, None)

    # ── Write validation_predictions.csv ─────────────────────────────────────
    template = pd.read_csv(TEMPLATE_CSV)
    template["predicted_rate"] = final_val_pred
    template.to_csv(OUT_PRED, index=False)
    print(f"\n  Wrote {OUT_PRED}")

    # ── December predictions ──────────────────────────────────────────────────
    dec_raw = pd.read_csv(DEC_CSV)
    # Add dummy cols present in train but missing in december
    dec_raw["load_id"] = "DEC"
    dec_raw["market_index"] = gstats["market_index"]
    dec_raw["quote_signal"] = gstats["quote_signal"]
    dec_raw["pickup_lat"] = 38.0495          # Lexington, KY approx
    dec_raw["pickup_lon"] = -84.4989
    dec_raw["delivery_lat"] = 41.0793        # Fort Wayne, IN approx
    dec_raw["delivery_lon"] = -85.1394

    dec_fe, _, _ = engineer_features(dec_raw, fit_encoders=encoders, global_stats=gstats)
    dec_lgb = lgb_full.predict(dec_fe[FEATURE_COLS])
    dec_xgb = xgb_full.predict(dec_fe[FEATURE_COLS])
    dec_pred = np.clip(best_w * dec_lgb + (1 - best_w) * dec_xgb, 50, None)

    dec_out = pd.read_csv(ROOT / "december-chart-inputs.csv")  # re-read clean version
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv(DEC_CSV, index=False)
    print(f"  Wrote {DEC_CSV}")

    # ── Save experiment results JSON ──────────────────────────────────────────
    summary = {
        "experiments": results,
        "best_model": best_label,
        "best_cv_mae": round(float(best_mae_ens), 4),
        "lgb_best_iter": int(best_lgb_iter),
        "xgb_best_iter": int(best_xgb_iter),
        "ensemble_lgb_weight": float(best_w),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {RESULTS_JSON}")
    print("\nDone! Run scorer to validate.")


if __name__ == "__main__":
    main()
