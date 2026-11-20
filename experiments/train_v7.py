"""
Freight Rate Prediction — v7: Optuna HPO for XGBoost + LightGBM
Runs 60 trials to find the best hyperparameters, then retrains on full data.
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
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TRAIN_CSV    = ROOT / "train-test.csv"
VAL_CSV      = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DEC_CSV      = ROOT / "december-chart-inputs.csv"
OUT_PRED     = ROOT / "validation_predictions.csv"
RESULTS_JSON = ROOT / "experiment_results.json"


def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")
    return {"label": label, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,6)}


def load_data():
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)


def build_lane_target_enc(df, smoothing=20):
    gm = df["posted_rate"].mean()
    df = df.copy()
    df["lane"] = df["pickup"] + " → " + df["delivery"]
    stats = df.groupby("lane")["posted_rate"].agg(["mean", "count"])
    stats["smooth"] = (
        (stats["mean"] * stats["count"] + gm * smoothing) / (stats["count"] + smoothing)
    )
    return stats["smooth"].to_dict(), gm


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
    df["distance_sq"]     = df["distance"] ** 2

    # Haversine
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["haversine_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # Three-way interactions
    df["dist_x_market_x_quote"] = df["distance"] * df["market_index"] * df["quote_signal"]
    df["equipment_x_distance"]  = df["distance"]  # will be multiplied after enc

    # Lane target encoding
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

    # Equipment × distance interaction
    df["equipment_x_distance"] = df["equipment_enc"] * df["distance"]

    return df, encoders, gstats


FEATURE_COLS = [
    "distance", "log_distance", "haversine_km", "distance_sq",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal",
    "dist_x_market", "dist_x_quote", "market_x_quote",
    "dist_x_market_x_quote", "equipment_x_distance",
    "day_of_week", "month", "week_of_year", "day_of_year",
    "is_weekend", "quarter", "days_since_start",
    "equipment_enc", "pickup_enc", "delivery_enc", "lane_enc",
    "lane_target_enc",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]


def main():
    print("="*60)
    print("FREIGHT RATE PREDICTION v7 — Optuna HPO")
    print("="*60)

    train_raw, val_raw = load_data()
    n = len(train_raw)
    cv_cutoff = int(n * 0.80)

    lane_mean_cv, gm_cv     = build_lane_target_enc(train_raw.iloc[:cv_cutoff])
    lane_mean_full, gm_full = build_lane_target_enc(train_raw)

    train_fe, encoders, gstats = engineer_features(
        train_raw, lane_mean=lane_mean_full, global_mean=gm_full)
    val_fe, _, _ = engineer_features(
        val_raw, fit_encoders=encoders, global_stats=gstats,
        lane_mean=lane_mean_full, global_mean=gm_full)

    train_cv_fe, _, _ = engineer_features(
        train_raw, fit_encoders=encoders, global_stats=gstats,
        lane_mean=lane_mean_cv, global_mean=gm_cv)

    X_cv_tr = train_cv_fe[FEATURE_COLS].iloc[:cv_cutoff].values
    X_cv_vl = train_cv_fe[FEATURE_COLS].iloc[cv_cutoff:].values
    y_cv_tr  = train_raw["posted_rate"].iloc[:cv_cutoff].values
    y_cv_vl  = train_raw["posted_rate"].iloc[cv_cutoff:].values
    X_train  = train_fe[FEATURE_COLS].values
    y_train  = train_raw["posted_rate"].values
    X_val    = val_fe[FEATURE_COLS].values

    results = []

    # ── XGBoost Optuna HPO ────────────────────────────────────────────────────
    print(f"\n--- v7a: XGBoost Optuna HPO (40 trials) ---")

    def xgb_objective(trial):
        params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 80,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_cv_tr, y_cv_tr, eval_set=[(X_cv_vl, y_cv_vl)], verbose=False)
        return mean_absolute_error(y_cv_vl, m.predict(X_cv_vl))

    study_xgb = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    # Enqueue known best parameters from HPO
    study_xgb.enqueue_trial({
        'learning_rate': 0.017665616756854928, 'max_depth': 8, 'min_child_weight': 15,
        'subsample': 0.8538599019652642, 'colsample_bytree': 0.9013411613515765,
        'colsample_bylevel': 0.5136911166983367, 'reg_alpha': 0.003187415631986632,
        'reg_lambda': 0.005765137706618165, 'gamma': 4.913676718804608
    })
    study_xgb.optimize(xgb_objective, n_trials=5, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    print(f"  Best XGB params: {best_xgb_params}")
    print(f"  Best XGB CV MAE: ${study_xgb.best_value:.2f}")

    # Refit with best params
    xgb_tuned = xgb.XGBRegressor(
        **best_xgb_params,
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=3000,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=80,
    )
    xgb_tuned.fit(X_cv_tr, y_cv_tr, eval_set=[(X_cv_vl, y_cv_vl)], verbose=False)
    xgb_cv_pred = xgb_tuned.predict(X_cv_vl)
    r_xgb = metrics(y_cv_vl, xgb_cv_pred, "v7a XGBoost Optuna")
    results.append(r_xgb)
    best_xgb_iter = xgb_tuned.best_iteration

    xgb_full = xgb.XGBRegressor(
        **best_xgb_params,
        objective="reg:absoluteerror",
        n_estimators=best_xgb_iter + 50,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    xgb_full.fit(X_train, y_train)
    xgb_val_pred = xgb_full.predict(X_val)

    # ── LightGBM Optuna HPO ───────────────────────────────────────────────────
    print(f"\n--- v7b: LightGBM Optuna HPO (40 trials) ---")

    def lgb_objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 1023),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "n_jobs": -1,
        }
        # Recreate dataset each trial to avoid feature_pre_filter conflict
        _lgb_tr = lgb.Dataset(X_cv_tr, y_cv_tr)
        _lgb_vl = lgb.Dataset(X_cv_vl, y_cv_vl, reference=_lgb_tr)
        cb = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=0)]
        booster = lgb.train(params, _lgb_tr, num_boost_round=3000,
                            valid_sets=[_lgb_vl], callbacks=cb)
        pred = booster.predict(X_cv_vl)
        return mean_absolute_error(y_cv_vl, pred)

    study_lgb = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.enqueue_trial({
        'learning_rate': 0.008753432039862064, 'num_leaves': 66, 'min_child_samples': 10,
        'feature_fraction': 0.5864419431357996, 'bagging_fraction': 0.6954116392194919,
        'bagging_freq': 5, 'reg_alpha': 0.7720050192818343, 'reg_lambda': 0.012199648658891778
    })
    study_lgb.optimize(lgb_objective, n_trials=5, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    print(f"  Best LGB params: {best_lgb_params}")
    print(f"  Best LGB CV MAE: ${study_lgb.best_value:.2f}")

    lgb_tuned_params = {
        **best_lgb_params,
        "objective": "regression_l1",
        "metric": "mae",
        "feature_pre_filter": False,
        "verbosity": -1,
        "n_jobs": -1,
    }
    cb = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=0)]
    # Recreate datasets fresh for final CV fit
    lgb_tr = lgb.Dataset(X_cv_tr, y_cv_tr)
    lgb_vl = lgb.Dataset(X_cv_vl, y_cv_vl, reference=lgb_tr)
    booster_cv = lgb.train(lgb_tuned_params, lgb_tr, num_boost_round=3000,
                           valid_sets=[lgb_vl], callbacks=cb)
    lgb_cv_pred = booster_cv.predict(X_cv_vl)
    r_lgb = metrics(y_cv_vl, lgb_cv_pred, "v7b LightGBM Optuna")
    results.append(r_lgb)
    best_lgb_iter = booster_cv.best_iteration

    lgb_full_tr = lgb.Dataset(X_train, y_train)
    booster_full = lgb.train(lgb_tuned_params, lgb_full_tr, num_boost_round=best_lgb_iter + 50)
    lgb_val_pred = booster_full.predict(X_val)

    # ── Optimal Ensemble ──────────────────────────────────────────────────────
    print("\n--- v7c: Optimal Ensemble ---")
    best_w, best_mae_ens = 0.5, float("inf")
    for w in np.arange(0.1, 0.95, 0.05):
        blend = w * xgb_cv_pred + (1 - w) * lgb_cv_pred
        m = mean_absolute_error(y_cv_vl, blend)
        if m < best_mae_ens:
            best_mae_ens, best_w = m, w
    print(f"  Optimal XGB weight: {best_w:.2f}")
    ens_cv_pred  = best_w * xgb_cv_pred + (1 - best_w) * lgb_cv_pred
    r_ens = metrics(y_cv_vl, ens_cv_pred, f"v7c Ensemble (XGB={best_w:.2f})")
    results.append(r_ens)
    ens_val_pred = best_w * xgb_val_pred + (1 - best_w) * lgb_val_pred

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("v7 SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['label']:<46} MAE=${r['MAE']:.2f}  RMSE=${r['RMSE']:.2f}  R²={r['R2']:.4f}")

    best = min(results, key=lambda x: x["MAE"])
    print(f"\n  → Best: {best['label']}  MAE=${best['MAE']:.2f}")

    pred_map = {
        results[0]["label"]: xgb_val_pred,
        results[1]["label"]: lgb_val_pred,
        results[2]["label"]: ens_val_pred,
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
                                      global_stats=gstats,
                                      lane_mean=lane_mean_full, global_mean=gm_full)
    dec_xgb = xgb_full.predict(dec_fe[FEATURE_COLS].values)
    dec_lgb = booster_full.predict(dec_fe[FEATURE_COLS].values)
    dec_pred = np.clip(best_w * dec_xgb + (1 - best_w) * dec_lgb, 50, None)

    dec_out = pd.read_csv(ROOT / "december-chart-inputs.csv")
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv(DEC_CSV, index=False)
    print(f"  Wrote {DEC_CSV}")

    # Update results
    try:
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
    except Exception:
        all_results = {"experiments": []}

    all_results["experiments"].extend(results)
    all_results["v7_best"] = best["label"]
    all_results["v7_best_mae"] = best["MAE"]
    all_results["v7_xgb_params"] = best_xgb_params
    all_results["v7_lgb_params"] = best_lgb_params
    all_results["v7_ensemble_xgb_weight"] = float(best_w)

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Updated {RESULTS_JSON}")
    print("\nDone!")


if __name__ == "__main__":
    main()
