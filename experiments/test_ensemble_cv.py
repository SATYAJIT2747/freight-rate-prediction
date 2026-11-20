"""
Ensemble test: XGBoost + LightGBM on Target (a) RPM
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train-test.csv")
train["date_dt"] = pd.to_datetime(train["date"])
train = train.sort_values("date_dt").reset_index(drop=True)

# Base features
train["day_of_week"] = train["date_dt"].dt.dayofweek
train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)

lat1, lon1 = np.radians(train["pickup_lat"]), np.radians(train["pickup_lon"])
lat2, lon2 = np.radians(train["delivery_lat"]), np.radians(train["delivery_lon"])
dlat, dlon = lat2 - lat1, lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
train["haversine_miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
train["circuity"] = train["distance"] / (train["haversine_miles"] + 1e-5)

train["weight"] = train["weight"].fillna(train.groupby("equipment")["weight"].transform("median"))
train["market_index"] = train["market_index"].fillna(train["market_index"].median())

train["base_quote"] = train["distance"] * train["quote_signal"]
train["dist_x_market"] = train["distance"] * train["market_index"]
train["market_x_quote"] = train["market_index"] * train["quote_signal"]
train["log_distance"] = np.log1p(train["distance"])
train["log_weight"] = np.log1p(train["weight"])
train["weight_per_mile"] = train["weight"] / (train["distance"] + 1)

eq_map = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
train["equipment_code"] = train["equipment"].map(eq_map).astype(int)

# For XGBoost
xgb_features = [
    "distance", "log_distance", "haversine_miles", "circuity",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal", "base_quote",
    "dist_x_market", "market_x_quote",
    "day_of_week", "is_weekend", "equipment_code",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
]

# For LightGBM
cat_cols = ["equipment", "pickup", "delivery"]
for c in cat_cols:
    train[c] = train[c].astype("category")

lgb_features = [
    "distance", "log_distance", "haversine_miles", "circuity",
    "weight", "log_weight", "weight_per_mile",
    "market_index", "quote_signal", "base_quote",
    "dist_x_market", "market_x_quote",
    "day_of_week", "is_weekend",
    "equipment", "pickup", "delivery",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"
]

folds = [
    ("Fold 1 (July)", train["date_dt"].dt.month < 7, train["date_dt"].dt.month == 7),
    ("Fold 2 (August)", train["date_dt"].dt.month < 8, train["date_dt"].dt.month == 8),
    ("Fold 3 (September)", train["date_dt"].dt.month < 9, train["date_dt"].dt.month == 9),
    ("Fold 4 (Oct 1-15)", train["date_dt"].dt.month < 10, (train["date_dt"].dt.month == 10) & (train["date_dt"].dt.day <= 15)),
]

for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_lgb = 1.0 - w_xgb
    fold_maes = []
    all_y_true = []
    all_y_pred = []
    
    for fold_name, tr_mask, val_mask in folds:
        tr = train[tr_mask]
        vl = train[val_mask]
        
        y_tr = tr["posted_rate"].values / tr["distance"].values
        y_vl_raw = vl["posted_rate"].values
        dist_vl = vl["distance"].values
        
        # Train XGB
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
        m_xgb.fit(tr[xgb_features], y_tr, eval_set=[(vl[xgb_features], y_vl_raw / dist_vl)], verbose=False)
        pred_xgb = m_xgb.predict(vl[xgb_features])
        
        # Train LGB
        m_lgb = lgb.LGBMRegressor(
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
        m_lgb.fit(
            tr[lgb_features], y_tr,
            eval_set=[(vl[lgb_features], y_vl_raw / dist_vl)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
        )
        pred_lgb = m_lgb.predict(vl[lgb_features])
        
        # Blend
        pred_blend = w_xgb * pred_xgb + w_lgb * pred_lgb
        dollar_preds = np.clip(pred_blend * dist_vl, 50, None)
        
        f_mae = mean_absolute_error(y_vl_raw, dollar_preds)
        fold_maes.append(f_mae)
        all_y_true.extend(y_vl_raw)
        all_y_pred.extend(dollar_preds)
        
    mean_mae = np.mean(fold_maes)
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    print(f"Weight XGB={w_xgb:.1f}, LGB={w_lgb:.1f} ==> MEAN CV MAE: ${mean_mae:.2f} | RMSE: ${rmse:.2f}")
