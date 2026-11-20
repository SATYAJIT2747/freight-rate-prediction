"""
Freight Rate Prediction — v6: CatBoost + LightGBM Refined Ensemble
Uses CatBoost's native categorical handling, deeper tuning.
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
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train-test.csv"
VAL_CSV   = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DEC_CSV   = ROOT / "december-chart-inputs.csv"
OUT_PRED  = ROOT / "validation_predictions.csv"
RESULTS_JSON = ROOT / "experiment_results.json"


def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")
    return {"label": label, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,6)}


def load_data():
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)


def engineer_features(df, fit_encoders=None, global_stats=None, lane_mean=None, global_mean=0):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"]      = df["date"].dt.dayofweek
    df["month"]            = df["date"].dt.month
    df["week_of_year"]     = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"]      = df["date"].dt.dayofyear
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]          = df["date"].dt.quarter
    df["days_since_start"] = (df["date"] - pd.Timestamp("2025-01-01")).dt.days

    df["lane"] = df["pickup"] + " → " + df["delivery"]

    gstats = global_stats or {}
    for col in ["weight", "market_index", "quote_signal"]:
        if col not in gstats:
            gstats[col] = df[col].median()
        df[col] = df[col].fillna(gstats[col])

    df["dist_x_market"]   = df["distance"] * df["market_index"]
    df["dist_x_quote"]    = df["distance"] * df["quote_signal"]
    df["market_x_quote"]  = df["market_index"] * df["quote_signal"]
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1)
    df["log_distance"]    = np.log1p(df["distance"])
    df["log_weight"]      = np.log1p(df["weight"])

    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["haversine_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # Smoothed target encoding for lane
    if lane_mean:
        df["lane_target_enc"] = df["lane"].map(lane_mean).fillna(global_mean)
    else:
        df["lane_target_enc"] = global_mean

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
    "lane_target_enc",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]

CAT_FEATURES_STR = ["equipment", "pickup", "delivery", "lane"]  # for CatBoost
CATBOOST_COLS = [c for c in FEATURE_COLS if c not in
                 ["equipment_enc","pickup_enc","delivery_enc","lane_enc"]] + CAT_FEATURES_STR


def build_lane_target_enc(df, smoothing=20):
    gm = df["posted_rate"].mean()
    stats = df.groupby("lane")["posted_rate"].agg(["mean","count"])
    stats["smooth"] = (stats["mean"]*stats["count"] + gm*smoothing) / (stats["count"]+smoothing)
    return stats["smooth"].to_dict(), gm


def main():
    print("="*60)
    print("FREIGHT RATE PREDICTION v6 — CatBoost + LGB Ensemble")
    print("="*60)

    train_raw, val_raw = load_data()
    n = len(train_raw)
    cv_cutoff = int(n * 0.80)
    print(f"\nTrain: {n} | CV: {cv_cutoff}/{n-cv_cutoff} | Val: {len(val_raw)}")

    # Build target encoding from pre-cutoff data only (no leakage)
    train_raw["lane"] = train_raw["pickup"] + " → " + train_raw["delivery"]
    lane_mean_cv, gm_cv     = build_lane_target_enc(train_raw.iloc[:cv_cutoff])
    lane_mean_full, gm_full = build_lane_target_enc(train_raw)

    # Feature engineering
    train_fe, encoders, gstats = engineer_features(
        train_raw, lane_mean=lane_mean_full, global_mean=gm_full)
    val_fe, _, _ = engineer_features(
        val_raw, fit_encoders=encoders, global_stats=gstats,
        lane_mean=lane_mean_full, global_mean=gm_full)

    # CV sets (use CV target enc to avoid leakage)
    train_cv_fe, _, _ = engineer_features(
        train_raw, fit_encoders=encoders, global_stats=gstats,
        lane_mean=lane_mean_cv, global_mean=gm_cv)

    X_cv_tr = train_cv_fe[FEATURE_COLS].iloc[:cv_cutoff]
    X_cv_vl = train_cv_fe[FEATURE_COLS].iloc[cv_cutoff:]
    y_cv_tr  = train_raw["posted_rate"].iloc[:cv_cutoff]
    y_cv_vl  = train_raw["posted_rate"].iloc[cv_cutoff:]
    X_train  = train_fe[FEATURE_COLS]
    y_train  = train_raw["posted_rate"]
    X_val    = val_fe[FEATURE_COLS]

    results = []

    # ── LightGBM v6 (more leaves, lower lr) ──────────────────────────────────
    print("\n--- v6a: LightGBM refined ---")
    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "num_leaves": 511,
        "min_child_samples": 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    lgb_cv = lgb.LGBMRegressor(**lgb_params)
    lgb_cv.fit(X_cv_tr, y_cv_tr,
               eval_set=[(X_cv_vl, y_cv_vl)],
               callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=0)])
    lgb_cv_pred = lgb_cv.predict(X_cv_vl)
    r_lgb = metrics(y_cv_vl, lgb_cv_pred, "v6a LightGBM refined")
    results.append(r_lgb)
    best_lgb_iter = lgb_cv.best_iteration_
    print(f"  Best iter: {best_lgb_iter}")

    lgb_full = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": best_lgb_iter + 50})
    lgb_full.fit(X_train, y_train)
    lgb_val_pred = lgb_full.predict(X_val)

    # ── CatBoost ─────────────────────────────────────────────────────────────
    print("\n--- v6b: CatBoost ---")
    # For CatBoost, pass string categoricals directly
    cb_tr_data = train_cv_fe.iloc[:cv_cutoff][FEATURE_COLS + CAT_FEATURES_STR].copy()
    cb_vl_data = train_cv_fe.iloc[cv_cutoff:][FEATURE_COLS + CAT_FEATURES_STR].copy()
    cb_full_data = train_fe[FEATURE_COLS + CAT_FEATURES_STR].copy()
    cb_val_data  = val_fe[FEATURE_COLS + CAT_FEATURES_STR].copy()

    for d in [cb_tr_data, cb_vl_data, cb_full_data, cb_val_data]:
        for c in CAT_FEATURES_STR:
            d[c] = d[c].astype(str)

    cat_idx = [cb_tr_data.columns.get_loc(c) for c in CAT_FEATURES_STR]

    cb_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=cat_idx,
    )
    cb_model.fit(
        Pool(cb_tr_data, y_cv_tr, cat_features=cat_idx),
        eval_set=Pool(cb_vl_data, y_cv_vl, cat_features=cat_idx),
    )
    cb_cv_pred = cb_model.predict(cb_vl_data)
    r_cb = metrics(y_cv_vl, cb_cv_pred, "v6b CatBoost")
    results.append(r_cb)
    print(f"  Best iter: {cb_model.best_iteration_}")

    cb_full = CatBoostRegressor(
        iterations=cb_model.best_iteration_ + 50,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=0,
        cat_features=cat_idx,
    )
    cb_full.fit(Pool(cb_full_data, y_train, cat_features=cat_idx))
    cb_val_pred = cb_full.predict(cb_val_data)

    # ── XGBoost (same as v4 best) ─────────────────────────────────────────────
    print("\n--- v6c: XGBoost (v4 best params) ---")
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
    xgb_cv.fit(X_cv_tr, y_cv_tr,
               eval_set=[(X_cv_vl, y_cv_vl)], verbose=False)
    xgb_cv_pred = xgb_cv.predict(X_cv_vl)
    r_xgb = metrics(y_cv_vl, xgb_cv_pred, "v6c XGBoost")
    results.append(r_xgb)
    best_xgb_iter = xgb_cv.best_iteration
    print(f"  Best iter: {best_xgb_iter}")

    xgb_full = xgb.XGBRegressor(**{**xgb_params,
                                   "n_estimators": best_xgb_iter + 50,
                                   "early_stopping_rounds": None})
    xgb_full.fit(X_train, y_train)
    xgb_val_pred = xgb_full.predict(X_val)

    # ── 3-way ensemble grid search ────────────────────────────────────────────
    print("\n--- v6d: 3-way Ensemble (LGB + CB + XGB) ---")
    best_combo = (0.2, 0.4, 0.4)
    best_mae_ens = float("inf")
    for w_lgb in np.arange(0.1, 0.5, 0.1):
        for w_cb in np.arange(0.1, 0.6, 0.1):
            w_xgb = round(1 - w_lgb - w_cb, 2)
            if w_xgb <= 0:
                continue
            blend = w_lgb*lgb_cv_pred + w_cb*cb_cv_pred + w_xgb*xgb_cv_pred
            m = mean_absolute_error(y_cv_vl, blend)
            if m < best_mae_ens:
                best_mae_ens = m
                best_combo = (w_lgb, w_cb, w_xgb)

    w_lgb, w_cb, w_xgb = best_combo
    print(f"  Best weights: LGB={w_lgb:.2f} CB={w_cb:.2f} XGB={w_xgb:.2f}")
    ens_cv_pred = w_lgb*lgb_cv_pred + w_cb*cb_cv_pred + w_xgb*xgb_cv_pred
    r_ens = metrics(y_cv_vl, ens_cv_pred, f"v6d Ensemble (L={w_lgb} C={w_cb} X={w_xgb})")
    results.append(r_ens)
    ens_val_pred = w_lgb*lgb_val_pred + w_cb*cb_val_pred + w_xgb*xgb_val_pred

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("v6 SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['label']:<46} MAE=${r['MAE']:.2f}  RMSE=${r['RMSE']:.2f}  R²={r['R2']:.4f}")

    best = min(results, key=lambda x: x["MAE"])
    print(f"\n  → Best: {best['label']}  MAE=${best['MAE']:.2f}")

    pred_map = {
        results[0]["label"]: lgb_val_pred,
        results[1]["label"]: cb_val_pred,
        results[2]["label"]: xgb_val_pred,
        results[3]["label"]: ens_val_pred,
    }
    final_val_pred = np.clip(pred_map[best["label"]], 50, None)

    # Write outputs
    template = pd.read_csv(TEMPLATE_CSV)
    template["predicted_rate"] = final_val_pred
    template.to_csv(OUT_PRED, index=False)
    print(f"\n  Wrote {OUT_PRED}")

    # December
    dec_raw = pd.read_csv(DEC_CSV)
    dec_raw["load_id"]       = "DEC"
    dec_raw["market_index"]  = gstats["market_index"]
    dec_raw["quote_signal"]  = gstats["quote_signal"]
    dec_raw["pickup_lat"]    = 38.0495
    dec_raw["pickup_lon"]    = -84.4989
    dec_raw["delivery_lat"]  = 41.0793
    dec_raw["delivery_lon"]  = -85.1394

    dec_fe, _, _ = engineer_features(dec_raw, fit_encoders=encoders,
                                      global_stats=gstats, lane_mean=lane_mean_full, global_mean=gm_full)
    dec_cb_data = dec_fe[FEATURE_COLS + CAT_FEATURES_STR].copy()
    for c in CAT_FEATURES_STR:
        dec_cb_data[c] = dec_cb_data[c].astype(str)

    dec_lgb = lgb_full.predict(dec_fe[FEATURE_COLS])
    dec_cb  = cb_full.predict(dec_cb_data)
    dec_xgb = xgb_full.predict(dec_fe[FEATURE_COLS])
    dec_pred = np.clip(w_lgb*dec_lgb + w_cb*dec_cb + w_xgb*dec_xgb, 50, None)

    dec_out = pd.read_csv(ROOT / "december-chart-inputs.csv")
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv(DEC_CSV, index=False)
    print(f"  Wrote {DEC_CSV}")

    # Update experiment results
    try:
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
    except Exception:
        all_results = {"experiments": []}

    all_results["experiments"].extend(results)
    all_results["v6_best"] = best["label"]
    all_results["v6_best_mae"] = best["MAE"]
    all_results["v6_weights"] = {"lgb": w_lgb, "cb": w_cb, "xgb": w_xgb}

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Updated {RESULTS_JSON}")
    print("\nDone!")


if __name__ == "__main__":
    main()
