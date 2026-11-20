"""
Phase 3 Feature Engineering & Validation Script
Tests leak-free OOF encodings, rolling market signals, route geometry, and calendar cleanup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

def build_features(df_input: pd.DataFrame, is_train: bool = True, historical_df: pd.DataFrame | None = None):
    df = df_input.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    
    # 1. Date features: strictly extrapolatable
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 2. Route Geometry & Circuity
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["haversine_miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    df["circuity"] = df["distance"] / (df["haversine_miles"] + 1e-5)
    
    # Geographic region bins (roughly dividing US into quadrants / 3x3 grid)
    df["pickup_region_lat"] = pd.qcut(df["pickup_lat"], q=3, labels=[0, 1, 2]).astype(int)
    df["pickup_region_lon"] = pd.qcut(df["pickup_lon"], q=3, labels=[0, 1, 2]).astype(int)
    df["delivery_region_lat"] = pd.qcut(df["delivery_lat"], q=3, labels=[0, 1, 2]).astype(int)
    df["delivery_region_lon"] = pd.qcut(df["delivery_lon"], q=3, labels=[0, 1, 2]).astype(int)
    df["region_corridor"] = (
        df["pickup_region_lat"].astype(str) + df["pickup_region_lon"].astype(str) + "_" +
        df["delivery_region_lat"].astype(str) + df["delivery_region_lon"].astype(str)
    )

    # 3. Missing Value Imputation (strictly independent of target)
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    # 4. Fundamental signals
    df["base_quote"] = df["distance"] * df["quote_signal"]
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["market_x_quote"] = df["market_index"] * df["quote_signal"]
    df["log_distance"] = np.log1p(df["distance"])
    df["log_weight"] = np.log1p(df["weight"])
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1)
    df["log_base_quote"] = np.log1p(df["base_quote"])

    # 5. Composite lane
    df["lane"] = df["pickup"] + " → " + df["delivery"]
    df["lane_equipment"] = df["lane"] + " | " + df["equipment"]

    return df

def compute_rolling_market_features(full_features_df: pd.DataFrame):
    """Computes daily rolling 7-day market signals per equipment (strictly feature-based, zero target)."""
    df = full_features_df.copy()
    
    # Group by date and equipment to get daily average signals
    daily = df.groupby(["date_dt", "equipment"])[["market_index", "quote_signal"]].mean().reset_index()
    daily = daily.sort_values(["equipment", "date_dt"]).reset_index(drop=True)
    
    daily["market_roll7"] = daily.groupby("equipment")["market_index"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    daily["quote_roll7"] = daily.groupby("equipment")["quote_signal"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    
    # Merge back
    df = df.merge(
        daily[["date_dt", "equipment", "market_roll7", "quote_roll7"]],
        on=["date_dt", "equipment"],
        how="left"
    )
    df["market_dev_roll7"] = df["market_index"] - df["market_roll7"]
    df["quote_dev_roll7"] = df["quote_signal"] - df["quote_roll7"]
    return df

def apply_leak_free_rpm_target_encoding(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: list[str], smoothing: float = 25.0):
    """
    Computes smoothed out-of-fold target encoding for RPM (posted_rate / distance)
    using ONLY training data, with zero future data leakage.
    """
    tr = train_df.copy()
    vl = val_df.copy()
    
    tr["target_rpm"] = tr["posted_rate"] / tr["distance"]
    global_mean_rpm = tr["target_rpm"].mean()

    # Apply to val from full train
    for col in cols:
        stats = tr.groupby(col)["target_rpm"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean_rpm) / (stats["count"] + smoothing)
        mapping = smooth.to_dict()
        vl[f"{col}_te_rpm"] = vl[col].map(mapping).fillna(global_mean_rpm).astype(float)
    
    # For train, compute 5-fold OOF target encoding
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in cols:
        tr[f"{col}_te_rpm"] = np.nan
        for tr_idx, oof_idx in kf.split(tr):
            sub_tr = tr.iloc[tr_idx]
            sub_oof = tr.iloc[oof_idx]
            sub_stats = sub_tr.groupby(col)["target_rpm"].agg(["count", "mean"])
            sub_smooth = (sub_stats["count"] * sub_stats["mean"] + smoothing * global_mean_rpm) / (sub_stats["count"] + smoothing)
            mapping = sub_smooth.to_dict()
            tr.iloc[oof_idx, tr.columns.get_loc(f"{col}_te_rpm")] = sub_oof[col].map(mapping).fillna(global_mean_rpm)
        tr[f"{col}_te_rpm"] = tr[f"{col}_te_rpm"].astype(float)
        
    return tr, vl

def run_phase3_cv():
    train_raw = pd.read_csv("train-test.csv")
    df = build_features(train_raw)
    df = compute_rolling_market_features(df)
    
    # Encode categories
    from sklearn.preprocessing import LabelEncoder
    cat_cols = ["equipment", "pickup", "delivery", "lane", "region_corridor"]
    for c in cat_cols:
        le = LabelEncoder()
        df[f"{c}_code"] = le.fit_transform(df[c].astype(str))

    base_feature_cols = [
        "distance", "log_distance", "haversine_miles", "circuity",
        "weight", "log_weight", "weight_per_mile",
        "market_index", "quote_signal", "base_quote", "log_base_quote",
        "dist_x_market", "market_x_quote",
        "market_roll7", "quote_roll7", "market_dev_roll7", "quote_dev_roll7",
        "day_of_week", "is_weekend",
        "equipment_code", "pickup_code", "delivery_code", "region_corridor_code",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
    ]

    te_cols = ["lane", "pickup", "delivery", "equipment", "lane_equipment"]
    
    folds = [
        ("Fold 1 (July)", df["date_dt"].dt.month < 7, df["date_dt"].dt.month == 7),
        ("Fold 2 (August)", df["date_dt"].dt.month < 8, df["date_dt"].dt.month == 8),
        ("Fold 3 (September)", df["date_dt"].dt.month < 9, df["date_dt"].dt.month == 9),
        ("Fold 4 (Oct 1-15)", df["date_dt"].dt.month < 10, (df["date_dt"].dt.month == 10) & (df["date_dt"].dt.day <= 15)),
    ]

    for model_type in ["xgb_rpm", "lgb_rpm", "xgb_ratio", "lgb_ratio"]:
        fold_maes = []
        all_true = []
        all_pred = []
        
        for fold_name, tr_mask, val_mask in folds:
            tr_df = df[tr_mask].copy()
            vl_df = df[val_mask].copy()
            
            # Apply leak-free RPM target encoding
            tr_df, vl_df = apply_leak_free_rpm_target_encoding(tr_df, vl_df, te_cols, smoothing=25.0)
            
            feature_cols = base_feature_cols + [f"{c}_te_rpm" for c in te_cols]
            
            X_tr = tr_df[feature_cols]
            X_vl = vl_df[feature_cols]
            
            y_tr_raw = tr_df["posted_rate"].values
            y_vl_raw = vl_df["posted_rate"].values
            dist_vl = vl_df["distance"].values
            base_vl = vl_df["base_quote"].values
            dist_tr = tr_df["distance"].values
            base_tr = tr_df["base_quote"].values

            if "rpm" in model_type:
                y_tr_target = y_tr_raw / dist_tr
                y_vl_target = y_vl_raw / dist_vl
            else: # ratio
                y_tr_target = y_tr_raw / base_tr
                y_vl_target = y_vl_raw / base_vl

            if "xgb" in model_type:
                model = xgb.XGBRegressor(
                    objective="reg:absoluteerror",
                    eval_metric="mae",
                    n_estimators=1200,
                    learning_rate=0.03,
                    max_depth=7,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    gamma=2.0,
                    random_state=42,
                    tree_method="hist",
                    n_jobs=-1,
                    early_stopping_rounds=60,
                )
                model.fit(X_tr, y_tr_target, eval_set=[(X_vl, y_vl_target)], verbose=False)
                preds = model.predict(X_vl)
            else: # lgb
                model = lgb.LGBMRegressor(
                    objective="regression_l1",
                    metric="mae",
                    n_estimators=1500,
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
                model.fit(
                    X_tr, y_tr_target,
                    eval_set=[(X_vl, y_vl_target)],
                    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(period=0)]
                )
                preds = model.predict(X_vl)

            if "rpm" in model_type:
                dollar_preds = np.clip(preds * dist_vl, 50, None)
            else:
                dollar_preds = np.clip(preds * base_vl, 50, None)
                
            fold_mae = mean_absolute_error(y_vl_raw, dollar_preds)
            fold_maes.append(fold_mae)
            all_true.extend(y_vl_raw)
            all_pred.extend(dollar_preds)
            
        mean_mae = np.mean(fold_maes)
        overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        print(f"\n{model_type.upper()}:")
        for (fn, _, _), fm in zip(folds, fold_maes):
            print(f"  {fn}: MAE = ${fm:.2f}")
        print(f"  ==> MEAN CV MAE: ${mean_mae:.2f} | Overall RMSE: ${overall_rmse:.2f}")

if __name__ == "__main__":
    run_phase3_cv()
