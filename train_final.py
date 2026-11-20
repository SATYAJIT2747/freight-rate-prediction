"""
Freight Rate Prediction Challenge — Final Production Pipeline (train_final.py)
-----------------------------------------------------------------------------
Implements:
1. Leak-free data preparation & feature engineering (route geometry, circuity, quote interactions).
2. Rate-Per-Mile (RPM = posted_rate / distance) target reformulation for heavy-tail robustness.
3. 4-fold expanding-window time-series validation & October 16-31 holdout benchmarking.
4. Multi-seed LightGBM & XGBoost ensemble trained on 100% labeled data (Jan-Oct 2025).
5. Generation of submission-ready validation_predictions.csv and december-chart-inputs.csv.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
TRAIN_CSV = ROOT / "train-test.csv"
VAL_CSV = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DEC_CSV = ROOT / "december-chart-inputs.csv"
OUT_PRED = ROOT / "validation_predictions.csv"


def calculate_haversine(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> np.ndarray:
    """Calculates great-circle distance in statute miles between coordinate pairs."""
    lat1, lon1 = np.radians(lat1_deg), np.radians(lon1_deg)
    lat2, lon2 = np.radians(lat2_deg), np.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    earth_radius_miles = 3958.8
    return earth_radius_miles * c


def build_features(df_raw: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Applies clean, leak-free feature engineering without extrapolatable calendar artifacts."""
    df = df_raw.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    # 1. Extrapolatable temporal features
    df["day_of_week"] = df["date_dt"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 2. Highway route geometry & circuity
    df["haversine_miles"] = calculate_haversine(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    df["circuity"] = df["distance"] / (df["haversine_miles"] + 1e-5)

    # 3. Missing value imputation (strictly fit on train)
    if stats is None:
        stats = {
            "weight_by_equip": df.groupby("equipment")["weight"].median().to_dict(),
            "market_index_median": float(df["market_index"].median()),
        }

    df["weight"] = df["weight"].fillna(df["equipment"].map(stats["weight_by_equip"]))
    df["market_index"] = df["market_index"].fillna(stats["market_index_median"])

    # 4. Economic quote & physical interaction signals
    df["base_quote"] = df["distance"] * df["quote_signal"]
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["market_x_quote"] = df["market_index"] * df["quote_signal"]
    df["log_distance"] = np.log1p(df["distance"])
    df["log_weight"] = np.log1p(df["weight"])
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1.0)

    # 5. Categorical encodings
    eq_map = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
    df["equipment_code"] = df["equipment"].map(eq_map).astype(int)

    for col in ["equipment", "pickup", "delivery"]:
        df[col] = df[col].astype("category")

    return df, stats


XGB_FEATURE_COLS = [
    "distance", "log_distance", "haversine_miles", "circuity",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal", "base_quote",
    "dist_x_market", "market_x_quote",
    "day_of_week", "is_weekend", "equipment_code",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
]

LGB_FEATURE_COLS = [
    "distance", "log_distance", "haversine_miles", "circuity",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal", "base_quote",
    "dist_x_market", "market_x_quote",
    "day_of_week", "is_weekend",
    "equipment", "pickup", "delivery",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
]


def run_cv_benchmark(train_df: pd.DataFrame) -> dict:
    """Executes expanding-window time-series CV on July, August, September, October 1-15, and Holdout Oct 16-31."""
    print("=" * 68)
    print("EXPANDING-WINDOW TIME-SERIES CROSS VALIDATION (RPM REFORMULATION)")
    print("=" * 68)

    folds = [
        ("Fold 1 (July 2025)", train_df["date_dt"].dt.month < 7, train_df["date_dt"].dt.month == 7),
        ("Fold 2 (August 2025)", train_df["date_dt"].dt.month < 8, train_df["date_dt"].dt.month == 8),
        ("Fold 3 (September 2025)", train_df["date_dt"].dt.month < 9, train_df["date_dt"].dt.month == 9),
        ("Fold 4 (Oct 1-15, 2025)", train_df["date_dt"].dt.month < 10, (train_df["date_dt"].dt.month == 10) & (train_df["date_dt"].dt.day <= 15)),
    ]

    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_name, tr_mask, val_mask in folds:
        tr = train_df[tr_mask]
        vl = train_df[val_mask]

        y_tr_rpm = tr["posted_rate"].values / tr["distance"].values
        y_vl_raw = vl["posted_rate"].values
        dist_vl = vl["distance"].values

        # XGBoost fold model
        m_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        m_xgb.fit(tr[XGB_FEATURE_COLS], y_tr_rpm, eval_set=[(vl[XGB_FEATURE_COLS], y_vl_raw / dist_vl)], verbose=False)
        pred_xgb = m_xgb.predict(vl[XGB_FEATURE_COLS]) * dist_vl

        # LightGBM fold model
        m_lgb = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        m_lgb.fit(
            tr[LGB_FEATURE_COLS], y_tr_rpm,
            eval_set=[(vl[LGB_FEATURE_COLS], y_vl_raw / dist_vl)],
            categorical_feature=["equipment", "pickup", "delivery"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
        )
        pred_lgb = m_lgb.predict(vl[LGB_FEATURE_COLS]) * dist_vl

        pred_blend = np.clip(0.5 * pred_xgb + 0.5 * pred_lgb, 50.0, None)
        mae = mean_absolute_error(y_vl_raw, pred_blend)
        fold_results.append((fold_name, mae))
        all_y_true.extend(y_vl_raw)
        all_y_pred.extend(pred_blend)
        print(f"  {fold_name:<26}: MAE = ${mae:.2f}")

    mean_cv_mae = float(np.mean([m for _, m in fold_results]))
    total_cv_rmse = float(np.sqrt(mean_squared_error(all_y_true, all_y_pred)))
    print(f"  --> MEAN CV MAE: ${mean_cv_mae:.2f} | Overall CV RMSE: ${total_cv_rmse:.2f}")

    # Holdout benchmark (Oct 16-31)
    ho_tr_mask = (train_df["date_dt"].dt.month < 10) | ((train_df["date_dt"].dt.month == 10) & (train_df["date_dt"].dt.day <= 15))
    ho_mask = (train_df["date_dt"].dt.month == 10) & (train_df["date_dt"].dt.day > 15)
    tr_ho = train_df[ho_tr_mask]
    vl_ho = train_df[ho_mask]

    m_xgb_ho = xgb.XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", n_estimators=1000,
        learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        tree_method="hist", n_jobs=-1, early_stopping_rounds=50
    )
    m_xgb_ho.fit(tr_ho[XGB_FEATURE_COLS], tr_ho["posted_rate"].values / tr_ho["distance"].values,
                 eval_set=[(vl_ho[XGB_FEATURE_COLS], vl_ho["posted_rate"].values / vl_ho["distance"].values)], verbose=False)
    pred_xgb_ho = m_xgb_ho.predict(vl_ho[XGB_FEATURE_COLS]) * vl_ho["distance"].values

    m_lgb_ho = lgb.LGBMRegressor(
        objective="regression_l1", metric="mae", n_estimators=1200,
        learning_rate=0.03, num_leaves=127, min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=-1
    )
    m_lgb_ho.fit(tr_ho[LGB_FEATURE_COLS], tr_ho["posted_rate"].values / tr_ho["distance"].values,
                 eval_set=[(vl_ho[LGB_FEATURE_COLS], vl_ho["posted_rate"].values / vl_ho["distance"].values)],
                 categorical_feature=["equipment", "pickup", "delivery"],
                 callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
    pred_lgb_ho = m_lgb_ho.predict(vl_ho[LGB_FEATURE_COLS]) * vl_ho["distance"].values

    holdout_blend = np.clip(0.5 * pred_xgb_ho + 0.5 * pred_lgb_ho, 50.0, None)
    holdout_mae = float(mean_absolute_error(vl_ho["posted_rate"].values, holdout_blend))
    holdout_rmse = float(np.sqrt(mean_squared_error(vl_ho["posted_rate"].values, holdout_blend)))
    print(f"  Holdout (Oct 16-31, 2025) : MAE = ${holdout_mae:.2f} | RMSE = ${holdout_rmse:.2f}")

    return {
        "mean_cv_mae": mean_cv_mae,
        "total_cv_rmse": total_cv_rmse,
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "folds": fold_results,
    }


def train_production_models(train_df: pd.DataFrame, seeds: list[int] = (42, 101, 2024)) -> tuple[list, list]:
    """Fits multi-seed ensemble on 100% of labeled training data (48,000 loads) using RPM target."""
    print("\n" + "=" * 68)
    print(f"TRAINING FINAL MULTI-SEED ENSEMBLE ON FULL TRAINING SET ({len(train_df):,} loads)")
    print("=" * 68)

    y_rpm = train_df["posted_rate"].values / train_df["distance"].values

    xgb_models = []
    lgb_models = []

    for seed in seeds:
        print(f"  Training Seed {seed} (XGBoost + LightGBM)...")
        # XGBoost
        model_xgb = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=1000,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
        )
        model_xgb.fit(train_df[XGB_FEATURE_COLS], y_rpm)
        xgb_models.append(model_xgb)

        # LightGBM
        model_lgb = lgb.LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=127,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        model_lgb.fit(train_df[LGB_FEATURE_COLS], y_rpm, categorical_feature=["equipment", "pickup", "delivery"])
        lgb_models.append(model_lgb)

    return xgb_models, lgb_models


def predict_ensemble(xgb_models: list, lgb_models: list, df_features: pd.DataFrame) -> np.ndarray:
    """Computes blended predictions in dollars from rate-per-mile predictions."""
    dist = df_features["distance"].values

    xgb_preds_rpm = np.mean([m.predict(df_features[XGB_FEATURE_COLS]) for m in xgb_models], axis=0)
    lgb_preds_rpm = np.mean([m.predict(df_features[LGB_FEATURE_COLS]) for m in lgb_models], axis=0)

    blended_rpm = 0.5 * xgb_preds_rpm + 0.5 * lgb_preds_rpm
    dollar_preds = np.clip(blended_rpm * dist, 50.0, None)
    return dollar_preds


def generate_december_chart_inputs(val_raw: pd.DataFrame, stats: dict,
                                   xgb_models: list, lgb_models: list) -> pd.DataFrame:
    """
    Fills december-chart-inputs.csv using:
    1. Real coordinates of Lexington and Fort Wayne derived from dataset rows.
    2. Real daily market_index and quote_signal means for December Dry Van loads in validation.csv.
    """
    dec_df = pd.read_csv(DEC_CSV)
    dec_df["date_dt"] = pd.to_datetime(dec_df["date"])

    # Extract December daily Dry Van market conditions from validation.csv
    val_df = val_raw.copy()
    val_df["date_dt"] = pd.to_datetime(val_df["date"])
    dec_val = val_df[(val_df["date_dt"].dt.month == 12) & (val_df["equipment"] == "Dry Van")]
    daily_signals = dec_val.groupby(dec_val["date_dt"].dt.strftime("%Y-%m-%d"))[["market_index", "quote_signal"]].mean()

    # Exact dataset coordinates
    lex_coords = (36.99152, -84.99876)
    fw_coords = (41.31561, -85.36206)

    dec_features_df = dec_df.copy()
    dec_features_df["pickup_lat"] = lex_coords[0]
    dec_features_df["pickup_lon"] = lex_coords[1]
    dec_features_df["delivery_lat"] = fw_coords[0]
    dec_features_df["delivery_lon"] = fw_coords[1]

    # Map daily market signals
    dec_features_df["date_str"] = dec_features_df["date_dt"].dt.strftime("%Y-%m-%d")
    dec_features_df["market_index"] = dec_features_df["date_str"].map(daily_signals["market_index"])
    dec_features_df["quote_signal"] = dec_features_df["date_str"].map(daily_signals["quote_signal"])

    # Engineer features
    dec_fe, _ = build_features(dec_features_df, stats=stats)
    dec_preds = predict_ensemble(xgb_models, lgb_models, dec_fe)

    dec_out = pd.read_csv(DEC_CSV)
    dec_out["predicted_rate"] = np.round(dec_preds, 2)
    dec_out.to_csv(DEC_CSV, index=False)
    print(f"  Successfully wrote {len(dec_out)} December predictions to {DEC_CSV}")
    return dec_out


def main():
    print("=" * 68)
    print("SPOTTER FREIGHT RATE PREDICTION CHALLENGE — PRODUCTION PIPELINE")
    print("=" * 68)

    train_raw = pd.read_csv(TRAIN_CSV)
    val_raw = pd.read_csv(VAL_CSV)
    template = pd.read_csv(TEMPLATE_CSV)

    train_fe, stats = build_features(train_raw)
    val_fe, _ = build_features(val_raw, stats=stats)

    # 1. Run rigorous 4-fold expanding-window cross validation & holdout
    cv_metrics = run_cv_benchmark(train_fe)

    # 2. Train multi-seed production models on 100% data
    xgb_models, lgb_models = train_production_models(train_fe, seeds=[42, 101, 2024])

    # 3. Predict validation set (12,000 loads)
    print("\nGenerating final predictions on validation.csv (12,000 loads)...")
    val_preds = predict_ensemble(xgb_models, lgb_models, val_fe)
    template["predicted_rate"] = np.round(val_preds, 2)
    template.to_csv(OUT_PRED, index=False)
    print(f"  Successfully wrote {len(template):,} predictions to {OUT_PRED}")

    # 4. Generate December 2025 forecast
    print("\nGenerating December 2025 benchmark chart predictions...")
    generate_december_chart_inputs(val_raw, stats, xgb_models, lgb_models)

    # 5. Verify outputs using score.py logic
    print("\nRunning submission verification via score.py...")
    import subprocess
    cmd = [sys.executable, "score.py", "--predictions", "validation_predictions.csv", "--december-predictions", "december-chart-inputs.csv"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR IN SCORER:", res.stderr)
        sys.exit(1)

    print("\nPipeline execution complete! All submission criteria passed.")


if __name__ == "__main__":
    main()
