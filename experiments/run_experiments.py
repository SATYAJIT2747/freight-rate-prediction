"""Model / feature comparison under expanding-window time-series CV.

Folds (train on everything strictly before the test month):
    Jun, Jul, Aug, Sep, Oct 2025
Early stopping uses the last 10% of the *training* window (by date), never the
test month. August matters most because it is the only labelled month
whose quote_signal regime matches the Nov-Dec validation set (signal = noise).

Usage:  python experiments/run_experiments.py
Writes: experiments/results/cv_results.csv, experiments/results/summary.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import BASE_FEATURES, MARKET_FEATURES, MARKET_REL_FEATURES, FeatureBuilder  # noqa: E402
from src.models import FITTERS, MIN_RATE, predict_rpm  # noqa: E402

OUT = ROOT / "experiments" / "results"
TEST_MONTHS = [6, 7, 8, 9, 10]

CORE = BASE_FEATURES  # no quote_signal, no market_index
WITH_MARKET = CORE + MARKET_FEATURES
WITH_REL_MARKET = CORE + MARKET_REL_FEATURES
WITH_QUOTE = WITH_MARKET + ["quote_signal"]
NO_PRIORS = [f for f in CORE if not f.endswith("_prior") and f != "lane_count"]

XGB_CORE = "XGBoost, core features"
# (name, model, feature list)
EXPERIMENTS = [
    ("LightGBM + quote_signal + market_index (previous approach)", "lgb", WITH_QUOTE),
    ("XGBoost + quote_signal + market_index", "xgb", WITH_QUOTE),
    ("XGBoost + market_index (no quote_signal)", "xgb", WITH_MARKET),
    ("XGBoost + month-relative market_index", "xgb", WITH_REL_MARKET),
    ("XGBoost, core features, no lane priors", "xgb", NO_PRIORS),
    (XGB_CORE, "xgb", CORE),
    ("LightGBM, core features", "lgb", CORE),
    ("CatBoost, core features", "cat", CORE),
]
BLEND = ("Blend: XGB + LGB + CatBoost, core features (FINAL)",
         [XGB_CORE, "LightGBM, core features", "CatBoost, core features"])


def split_es(tr: pd.DataFrame, frac: float = 0.10):
    cutoff = tr["date_dt"].quantile(1 - frac)
    return tr[tr["date_dt"] < cutoff], tr[tr["date_dt"] >= cutoff]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "train_test.csv")
    raw["month"] = pd.to_datetime(raw["date"]).dt.month

    rows = []
    preds = {}  # (exp, month) -> dollar predictions, for blends
    for month in TEST_MONTHS:
        tr_raw = raw[raw["month"] < month].reset_index(drop=True)
        te_raw = raw[raw["month"] == month].reset_index(drop=True)
        fb = FeatureBuilder()
        tr = fb.fit_transform(tr_raw)
        te = fb.transform(te_raw)
        fit_part, es_part = split_es(tr)
        y = lambda d: (d["posted_rate"] / d["distance"]).to_numpy()  # noqa: E731
        truth = te["posted_rate"].to_numpy()

        # Baseline: smoothed lane x equipment median RPM * distance
        base = np.clip(te["lane_rpm_prior"] * te["distance"], MIN_RATE, None).to_numpy()
        preds[("Baseline: lane median RPM x distance", month)] = base

        for name, kind, feats in EXPERIMENTS:
            t0 = time.time()
            model = FITTERS[kind](fit_part[feats], y(fit_part), es_part[feats], y(es_part))
            p = np.clip(predict_rpm(model, te[feats]) * te["distance"].to_numpy(), MIN_RATE, None)
            preds[(name, month)] = p
            print(f"month {month:>2} | {name:<62} MAE ${np.abs(p - truth).mean():7.2f} ({time.time() - t0:.0f}s)")

        preds[(BLEND[0], month)] = np.mean([preds[(n, month)] for n in BLEND[1]], axis=0)
        for (name, m), p in preds.items():
            if m == month:
                err = p - truth
                rows.append({"experiment": name, "month": month, "n": len(truth),
                             "mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()),
                             "median_ae": np.median(np.abs(err)), "bias": err.mean()})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_results.csv", index=False)
    table = res.pivot_table(index="experiment", columns="month", values="mae")
    table.columns = [pd.Timestamp(2025, m, 1).strftime("%b") for m in table.columns]
    table["Mean MAE"] = table.mean(axis=1)
    table = table.sort_values("Mean MAE").round(2)
    bias = res.groupby("experiment")["bias"].mean().round(1)
    table["Mean bias"] = bias.reindex(table.index)
    md = table.to_markdown()
    (OUT / "summary.md").write_text(
        "# Expanding-window CV — MAE ($) per test month\n\n" + md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
