"""
Freight Rate Prediction — v5: Log-target + Target Encoding
Applies log1p transform to posted_rate, adds target-encoded lane feature,
and re-runs LightGBM + XGBoost + Ensemble.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train-test.csv"
VAL_CSV = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DEC_CSV_SRC = ROOT / "december-chart-inputs.csv"
OUT_PRED = ROOT / "validation_predictions.csv"
RESULTS_JSON = ROOT / "experiment_results.json"


def metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")
    return {"label": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 6)}


def load_data():
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    return train, val


def engineer_features(df, fit_encoders=None, global_stats=None, target_enc_map=None):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Date features
    df["day_of_week"]     = df["date"].dt.dayofweek
    df["month"]           = df["date"].dt.month
    df["week_of_year"]    = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"]     = df["date"].dt.dayofyear
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]         = df["date"].dt.quarter
    df["days_since_start"]= (df["date"] - pd.Timestamp("2025-01-01")).dt.days

    # Lane
    df["lane"] = df["pickup"] + " → " + df["delivery"]

    # Fill nulls
    gstats = global_stats or {}
    for col in ["weight", "market_index", "quote_signal"]:
        if col not in gstats:
            gstats[col] = df[col].median()
        df[col] = df[col].fillna(gstats[col])

    # Interactions
    df["dist_x_market"]   = df["distance"] * df["market_index"]
    df["dist_x_quote"]    = df["distance"] * df["quote_signal"]
    df["market_x_quote"]  = df["market_index"] * df["quote_signal"]
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1)
    df["log_distance"]    = np.log1p(df["distance"])
    df["log_weight"]      = np.log1p(df["weight"])

    # Haversine
    lat1 = np.radians(df["pickup_lat"]); lon1 = np.radians(df["pickup_lon"])
    lat2 = np.radians(df["delivery_lat"]); lon2 = np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["haversine_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # Label encoders
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

    # Target encoding for lane (mean posted_rate per lane on train)
    if target_enc_map is None:
        target_enc_map = {}
    if "lane_target_enc" not in target_enc_map:
        target_enc_map["lane_target_enc"] = {}  # will be filled externally
    global_mean = target_enc_map.get("global_mean", 0)
    df["lane_target_enc"] = df["lane"].map(target_enc_map.get("lane_mean", {})).fillna(global_mean)

    return df, encoders, gstats, target_enc_map


FEATURE_COLS = [
    "distance", "log_distance", "haversine_km",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal",
    "dist_x_market", "dist_x_quote", "market_x_quote",
    "day_of_week", "month", "week_of_year", "day_of_year",
    "is_weekend", "quarter", "days_since_start",
    "equipment_enc", "pickup_enc", "delivery_enc", "lane_enc",
    "lane_target_enc",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]


def build_target_enc(train_df, cutoff_idx, smoothing=10):
    """Out-of-fold target encoding to avoid leakage."""
    tr = train_df.iloc[:cutoff_idx].copy()
    tr["lane"] = tr["pickup"] + " → " + tr["delivery"]
    global_mean = tr["posted_rate"].mean()
    lane_stats = tr.groupby("lane")["posted_rate"].agg(["mean", "count"])
    # Smoothed target encoding
    lane_stats["smooth"] = (
        (lane_stats["mean"] * lane_stats["count"] + global_mean * smoothing)
        / (lane_stats["count"] + smoothing)
    )
    return {"lane_mean": lane_stats["smooth"].to_dict(), "global_mean": global_mean}


def main():
    print("=" * 60)
    print("FREIGHT RATE PREDICTION v5 — LOG-TARGET + TARGET ENCODING")
    print("=" * 60)

    train_raw, val_raw = load_data()
    n = len(train_raw)
    cv_cutoff = int(n * 0.80)
    print(f"\nTrain: {n} | CV cutoff: {cv_cutoff} | Val: {len(val_raw)}")

    # Build target encoding from TRAIN ONLY (before CV cutoff to avoid leakage)
    tenc_cv   = build_target_enc(train_raw, cv_cutoff)
    tenc_full = build_target_enc(train_raw, n)

    # Feature engineering
    train_fe, encoders, gstats, _ = engineer_features(train_raw, target_enc_map=tenc_full)
    val_fe, _, _, _                = engineer_features(val_raw, fit_encoders=encoders,
                                                       global_stats=gstats, target_enc_map=tenc_full)

    # For CV split, re-engineer with CV target encoding
    train_cv_fe, _, _, _ = engineer_features(train_raw, fit_encoders=encoders,
                                              global_stats=gstats, target_enc_map=tenc_cv)

    X_cv_tr = train_cv_fe[FEATURE_COLS].iloc[:cv_cutoff]
    X_cv_vl = train_cv_fe[FEATURE_COLS].iloc[cv_cutoff:]
    y_cv_tr = train_raw["posted_rate"].iloc[:cv_cutoff]
    y_cv_vl = train_raw["posted_rate"].iloc[cv_cutoff:]

    # Log-transform target
    log_y_cv_tr = np.log1p(y_cv_tr)

    X_train = train_fe[FEATURE_COLS]
    y_train = train_raw["posted_rate"]
    X_val   = val_fe[FEATURE_COLS]

    results = []

    # ── LightGBM on log target ────────────────────────────────────────────────
    print("\n--- v5a: LightGBM (log target + target enc) ---")
    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "num_leaves": 255,
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
        X_cv_tr, log_y_cv_tr,
        eval_set=[(X_cv_vl, np.log1p(y_cv_vl))],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)],
    )
    lgb_cv_pred = np.expm1(lgb_cv.predict(X_cv_vl))
    r_lgb = metrics(y_cv_vl, lgb_cv_pred, "v5a LightGBM log-target")
    results.append(r_lgb)
    best_lgb_iter = lgb_cv.best_iteration_
    print(f"  Best iter: {best_lgb_iter}")

    lgb_full = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": best_lgb_iter + 50})
    lgb_full.fit(X_train, np.log1p(y_train))
    lgb_val_pred = np.expm1(lgb_full.predict(X_val))

    # ── XGBoost on log target ─────────────────────────────────────────────────
    print("\n--- v5b: XGBoost (log target + target enc) ---")
    xgb_params = {
        "objective": "reg:absoluteerror",
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
        "early_stopping_rounds": 100,
    }
    xgb_cv = xgb.XGBRegressor(**xgb_params)
    xgb_cv.fit(
        X_cv_tr, log_y_cv_tr,
        eval_set=[(X_cv_vl, np.log1p(y_cv_vl))],
        verbose=False,
    )
    xgb_cv_pred = np.expm1(xgb_cv.predict(X_cv_vl))
    r_xgb = metrics(y_cv_vl, xgb_cv_pred, "v5b XGBoost log-target")
    results.append(r_xgb)
    best_xgb_iter = xgb_cv.best_iteration
    print(f"  Best iter: {best_xgb_iter}")

    xgb_full = xgb.XGBRegressor(**{**xgb_params, "n_estimators": best_xgb_iter + 50,
                                   "early_stopping_rounds": None})
    xgb_full.fit(X_train, np.log1p(y_train))
    xgb_val_pred = np.expm1(xgb_full.predict(X_val))

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n--- v5c: Ensemble ---")
    best_w, best_mae_ens = 0.5, float("inf")
    for w in np.arange(0.3, 0.8, 0.05):
        blend = w * lgb_cv_pred + (1 - w) * xgb_cv_pred
        m = mean_absolute_error(y_cv_vl, blend)
        if m < best_mae_ens:
            best_mae_ens, best_w = m, w
    print(f"  Optimal LGB weight: {best_w:.2f}")
    ens_cv_pred = best_w * lgb_cv_pred + (1 - best_w) * xgb_cv_pred
    r_ens = metrics(y_cv_vl, ens_cv_pred, f"v5c Ensemble (LGB={best_w:.2f})")
    results.append(r_ens)
    ens_val_pred = best_w * lgb_val_pred + (1 - best_w) * xgb_val_pred

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("v5 SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:<40} MAE=${r['MAE']:.2f}  RMSE=${r['RMSE']:.2f}  R²={r['R2']:.4f}")

    best_label = min(results, key=lambda x: x["MAE"])["label"]
    print(f"\n  → Best: {best_label}")

    pred_map = {
        results[0]["label"]: lgb_val_pred,
        results[1]["label"]: xgb_val_pred,
        results[2]["label"]: ens_val_pred,
    }
    final_val_pred = np.clip(pred_map[best_label], 50, None)

    # ── Write validation_predictions.csv ──────────────────────────────────────
    template = pd.read_csv(TEMPLATE_CSV)
    template["predicted_rate"] = final_val_pred
    template.to_csv(OUT_PRED, index=False)
    print(f"\n  Wrote {OUT_PRED}")

    # ── December predictions ──────────────────────────────────────────────────
    dec_raw = pd.read_csv(DEC_CSV_SRC)
    dec_raw["load_id"]       = "DEC"
    dec_raw["market_index"]  = gstats["market_index"]
    dec_raw["quote_signal"]  = gstats["quote_signal"]
    dec_raw["pickup_lat"]    = 38.0495
    dec_raw["pickup_lon"]    = -84.4989
    dec_raw["delivery_lat"]  = 41.0793
    dec_raw["delivery_lon"]  = -85.1394

    dec_fe, _, _, _ = engineer_features(dec_raw, fit_encoders=encoders,
                                         global_stats=gstats, target_enc_map=tenc_full)
    dec_lgb = np.expm1(lgb_full.predict(dec_fe[FEATURE_COLS]))
    dec_xgb = np.expm1(xgb_full.predict(dec_fe[FEATURE_COLS]))
    dec_pred = np.clip(best_w * dec_lgb + (1 - best_w) * dec_xgb, 50, None)

    dec_out = pd.read_csv(ROOT / "december-chart-inputs.csv")
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv(DEC_CSV_SRC, index=False)
    print(f"  Wrote {DEC_CSV_SRC}")

    # ── Update experiment_results.json ────────────────────────────────────────
    try:
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
    except Exception:
        all_results = {"experiments": []}

    all_results["experiments"].extend(results)
    all_results["v5_best"] = best_label
    all_results["v5_best_mae"] = round(float(min(r["MAE"] for r in results)), 4)
    all_results["v5_lgb_best_iter"] = int(best_lgb_iter)
    all_results["v5_xgb_best_iter"] = int(best_xgb_iter)
    all_results["v5_ensemble_lgb_weight"] = float(best_w)

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Updated {RESULTS_JSON}")
    print("\nDone! Run scorer to validate.")


if __name__ == "__main__":
    main()
