"""Model definitions. All models predict rate-per-mile (RPM) with an L1 (MAE)
objective; dollar predictions are RPM * distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

CAT_COLS = ["equipment", "pickup", "delivery"]
MIN_RATE = 50.0


def _cats(features: list[str]) -> list[str]:
    return [c for c in features if c in CAT_COLS]


def fit_lgb(X: pd.DataFrame, y: np.ndarray, X_es=None, y_es=None, n_estimators=3000, seed=42):
    model = lgb.LGBMRegressor(
        objective="l1", n_estimators=n_estimators, learning_rate=0.03,
        num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, cat_smooth=20,
        random_state=seed, verbose=-1, n_jobs=-1,
    )
    kw = {"categorical_feature": _cats(list(X.columns))}
    if X_es is not None:
        kw.update(eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(100, verbose=False)])
    model.fit(X, y, **kw)
    return model


def fit_xgb(X: pd.DataFrame, y: np.ndarray, X_es=None, y_es=None, n_estimators=3000, seed=42):
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror", n_estimators=n_estimators, learning_rate=0.03,
        max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
        random_state=seed, n_jobs=-1,
        early_stopping_rounds=100 if X_es is not None else None,
    )
    kw = {"eval_set": [(X_es, y_es)]} if X_es is not None else {}
    model.fit(X, y, verbose=False, **kw)
    return model


def _cb_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in _cats(list(X.columns)):
        X[c] = X[c].astype(object).where(X[c].notna(), "__unseen__").astype(str)
    return X


def fit_cat(X: pd.DataFrame, y: np.ndarray, X_es=None, y_es=None, n_estimators=3000, seed=42):
    model = CatBoostRegressor(
        loss_function="MAE", iterations=n_estimators, learning_rate=0.08, depth=6,
        random_seed=seed, verbose=0, allow_writing_files=False, thread_count=-1,
        cat_features=_cats(list(X.columns)),
    )
    kw = {}
    if X_es is not None:
        kw = {"eval_set": (_cb_frame(X_es), y_es), "early_stopping_rounds": 100}
    model.fit(_cb_frame(X), y, **kw)
    return model


def best_iteration(model) -> int:
    if isinstance(model, lgb.LGBMRegressor):
        return int(model.best_iteration_ or model.n_estimators)
    if isinstance(model, xgb.XGBRegressor):
        return int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
    return int(model.get_best_iteration() or model.tree_count_) + 1


def predict_rpm(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, CatBoostRegressor):
        return model.predict(_cb_frame(X))
    return model.predict(X)


FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}
