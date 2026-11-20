"""
Phase 0 & Phase 2: Expanding-Window Time-Series CV and Target Reformulation Comparison
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def load_preprocessed_data():
    df = pd.read_csv("train-test.csv")
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)
    
    # Base features
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Coordinates
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df["haversine_miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    df["circuity"] = df["distance"] / (df["haversine_miles"] + 1e-5)
    
    # Missing value handling
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())
    
    # Simple interactions
    df["base_quote"] = df["distance"] * df["quote_signal"]
    df["dist_x_market"] = df["distance"] * df["market_index"]
    df["market_x_quote"] = df["market_index"] * df["quote_signal"]
    df["log_distance"] = np.log1p(df["distance"])
    df["log_weight"] = np.log1p(df["weight"])
    df["weight_per_mile"] = df["weight"] / (df["distance"] + 1)

    # Equipment categorical
    eq_map = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
    df["equipment_code"] = df["equipment"].map(eq_map).astype(int)

    feature_cols = [
        "distance", "log_distance", "haversine_miles", "circuity",
        "weight", "log_weight", "weight_per_mile",
        "market_index", "quote_signal", "base_quote",
        "dist_x_market", "market_x_quote",
        "day_of_week", "is_weekend", "equipment_code",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
    ]
    return df, feature_cols

def run_experiment(name, target_type, use_sample_weight=False):
    df, feature_cols = load_preprocessed_data()

    # Define Folds
    folds = [
        ("Fold 1 (July)", df["date_dt"].dt.month < 7, df["date_dt"].dt.month == 7),
        ("Fold 2 (August)", df["date_dt"].dt.month < 8, df["date_dt"].dt.month == 8),
        ("Fold 3 (September)", df["date_dt"].dt.month < 9, df["date_dt"].dt.month == 9),
        ("Fold 4 (Oct 1-15)", df["date_dt"].dt.month < 10, (df["date_dt"].dt.month == 10) & (df["date_dt"].dt.day <= 15)),
    ]

    fold_maes = []
    all_y_true = []
    all_y_pred = []

    for fold_name, tr_mask, val_mask in folds:
        train_df = df[tr_mask]
        val_df = df[val_mask]

        X_tr = train_df[feature_cols]
        X_val = val_df[feature_cols]
        y_tr_raw = train_df["posted_rate"].values
        y_val_raw = val_df["posted_rate"].values

        dist_tr = train_df["distance"].values
        dist_val = val_df["distance"].values
        base_tr = train_df["base_quote"].values
        base_val = val_df["base_quote"].values

        sample_weight = None
        if target_type == "raw":
            y_tr = y_tr_raw
        elif target_type == "rpm":
            y_tr = y_tr_raw / dist_tr
            if use_sample_weight:
                sample_weight = dist_tr
        elif target_type == "quote_ratio":
            y_tr = y_tr_raw / base_tr
            if use_sample_weight:
                sample_weight = base_tr
        elif target_type == "log_quote_ratio":
            y_tr = np.log(np.clip(y_tr_raw / base_tr, 1e-4, None))
            if use_sample_weight:
                sample_weight = base_tr

        model = xgb.XGBRegressor(
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

        # Early stopping eval set
        if target_type == "raw":
            y_val_target = y_val_raw
        elif target_type == "rpm":
            y_val_target = y_val_raw / dist_val
        elif target_type == "quote_ratio":
            y_val_target = y_val_raw / base_val
        elif target_type == "log_quote_ratio":
            y_val_target = np.log(np.clip(y_val_raw / base_val, 1e-4, None))

        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_target)],
            verbose=False,
        )

        preds = model.predict(X_val)

        # Convert back to DOLLARS
        if target_type == "raw":
            dollar_preds = np.clip(preds, 50, None)
        elif target_type == "rpm":
            dollar_preds = np.clip(preds * dist_val, 50, None)
        elif target_type == "quote_ratio":
            dollar_preds = np.clip(preds * base_val, 50, None)
        elif target_type == "log_quote_ratio":
            dollar_preds = np.clip(np.exp(preds) * base_val, 50, None)

        fold_mae = mean_absolute_error(y_val_raw, dollar_preds)
        fold_maes.append(fold_mae)
        all_y_true.extend(y_val_raw)
        all_y_pred.extend(dollar_preds)

    mean_mae = np.mean(fold_maes)
    total_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    print(f"\n{name}:")
    for (fname, _, _), f_mae in zip(folds, fold_maes):
        print(f"  {fname}: MAE = ${f_mae:.2f}")
    print(f"  --> MEAN CV MAE: ${mean_mae:.2f} | Overall RMSE: ${total_rmse:.2f}")
    return {"name": name, "folds": fold_maes, "mean_mae": mean_mae, "rmse": total_rmse}

if __name__ == "__main__":
    results = []
    # Baseline: raw posted_rate
    results.append(run_experiment("1. Baseline (Raw posted_rate)", "raw"))
    # (a) RPM = posted_rate / distance
    results.append(run_experiment("2. Target (a) RPM", "rpm", use_sample_weight=False))
    results.append(run_experiment("3. Target (a) RPM with sample_weight=distance", "rpm", use_sample_weight=True))
    # (b) Quote Ratio = posted_rate / (distance * quote_signal)
    results.append(run_experiment("4. Target (b) Quote Ratio", "quote_ratio", use_sample_weight=False))
    results.append(run_experiment("5. Target (b) Quote Ratio with sample_weight=base_quote", "quote_ratio", use_sample_weight=True))
    # (c) Log Quote Ratio
    results.append(run_experiment("6. Target (c) Log Quote Ratio", "log_quote_ratio", use_sample_weight=False))
    results.append(run_experiment("7. Target (c) Log Quote Ratio with sample_weight=base_quote", "log_quote_ratio", use_sample_weight=True))
