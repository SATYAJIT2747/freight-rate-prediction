"""Final training + inference pipeline.

Steps
1. Clean + engineer features (src/features.py), fitted on training data only.
2. Holdout check: train on Jan-Sep, evaluate on October (never touched during
   fitting; early stopping uses the last 10% of the Jan-Sep window). This also
   fixes the number of boosting rounds for the final refit.
3. Refit every model on all labelled data (Jan-Oct) with 3 seeds each.
4. Predict the 12,000 validation loads -> validation_predictions.csv
5. Fill data/december_chart_inputs.csv and run the provided score.py, which
   validates both files and draws scorer_results/candidate_december.png.

Usage:  python train.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import BASE_FEATURES, FeatureBuilder
from src.models import FITTERS, MIN_RATE, best_iteration, predict_rpm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_CSV = DATA / "train_test.csv"
VALID_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"
OUT_PRED = ROOT / "validation_predictions.csv"
OUT_METRICS = ROOT / "experiments" / "results" / "final_model_metrics.json"

MODELS = ["xgb", "lgb", "cat"]  # equal-weight blend
CV_NAME = "Blend: XGB + LGB + CatBoost, core features (FINAL)"  # row in experiments/results/summary.md
SEEDS = [42, 101, 2024]
FEATURES = BASE_FEATURES


def rpm(df: pd.DataFrame) -> np.ndarray:
    return (df["posted_rate"] / df["distance"]).to_numpy()


def to_dollars(rpm_pred: np.ndarray, distance) -> np.ndarray:
    return np.clip(rpm_pred * np.asarray(distance, dtype=float), MIN_RATE, None)


def holdout_check(raw: pd.DataFrame) -> dict:
    """Train Jan-Sep, test October. Returns per-model best iterations + metrics."""
    month = pd.to_datetime(raw["date"]).dt.month
    tr_raw, te_raw = raw[month < 10].reset_index(drop=True), raw[month == 10].reset_index(drop=True)
    fb = FeatureBuilder()
    tr, te = fb.fit_transform(tr_raw), fb.transform(te_raw)
    cutoff = tr["date_dt"].quantile(0.90)
    fit, es = tr[tr["date_dt"] < cutoff], tr[tr["date_dt"] >= cutoff]

    truth, preds, iters, metrics = te["posted_rate"].to_numpy(), {}, {}, {}
    for kind in MODELS:
        m = FITTERS[kind](fit[FEATURES], rpm(fit), es[FEATURES], rpm(es))
        iters[kind] = best_iteration(m)
        preds[kind] = to_dollars(predict_rpm(m, te[FEATURES]), te["distance"])
        metrics[kind] = float(np.abs(preds[kind] - truth).mean())
        print(f"  {kind}: best_iter={iters[kind]:>5}  Oct MAE=${metrics[kind]:.2f}")
    blend = np.mean(list(preds.values()), axis=0)
    metrics["blend"] = float(np.abs(blend - truth).mean())
    metrics["blend_rmse"] = float(np.sqrt(((blend - truth) ** 2).mean()))
    print(f"  blend: Oct MAE=${metrics['blend']:.2f}  RMSE=${metrics['blend_rmse']:.2f}")
    return {"best_iterations": iters, "october_holdout": metrics}


def december_frame(valid_raw: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs for the fixed Lexington -> Fort Wayne chart rows.

    The chart file only fixes route, distance, equipment, weight and date. The
    remaining inputs come from the data itself: city coordinates from the
    (perfectly consistent) city table, and market_index = that day's mean
    market_index across all December validation loads (features only)."""
    dec = pd.read_csv(DECEMBER_CSV)
    cities = pd.concat([
        train_raw[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        train_raw[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")
    daily_mi = valid_raw.groupby("date")["market_index"].mean()

    x = dec.drop(columns="predicted_rate").copy()
    x["load_id"] = [f"DEC-{i:02d}" for i in range(1, len(x) + 1)]
    x["pickup_lat"], x["pickup_lon"] = cities.loc[x["pickup"], "lat"].values, cities.loc[x["pickup"], "lon"].values
    x["delivery_lat"], x["delivery_lon"] = cities.loc[x["delivery"], "lat"].values, cities.loc[x["delivery"], "lon"].values
    x["market_index"] = x["date"].map(daily_mi)
    x["quote_signal"] = np.nan  # not used by the model
    return x


def main() -> None:
    t0 = time.time()
    train_raw = pd.read_csv(TRAIN_CSV)
    valid_raw = pd.read_csv(VALID_CSV)
    template = pd.read_csv(TEMPLATE_CSV)
    print(f"train {len(train_raw):,} loads | validation {len(valid_raw):,} loads")

    print("\n[1/4] October holdout check (train Jan-Sep)")
    report = holdout_check(train_raw)

    print("\n[2/4] Refit on all labelled data (Jan-Oct), seeds", SEEDS)
    fb = FeatureBuilder()
    train = fb.fit_transform(train_raw)
    valid = fb.transform(valid_raw)
    dec_x = fb.transform(december_frame(valid_raw, train_raw))
    y = rpm(train)

    valid_preds, dec_preds = [], []
    for kind in MODELS:
        # +10%: the refit sees ~10% more data than the early-stopping run did
        n_rounds = int(report["best_iterations"][kind] * 1.1)
        for seed in SEEDS:
            m = FITTERS[kind](train[FEATURES], y, n_estimators=n_rounds, seed=seed)
            valid_preds.append(predict_rpm(m, valid[FEATURES]))
            dec_preds.append(predict_rpm(m, dec_x[FEATURES]))
        print(f"  {kind}: {len(SEEDS)} seeds x {n_rounds} rounds")

    print("\n[3/4] Writing predictions")
    pred = pd.Series(to_dollars(np.mean(valid_preds, axis=0), valid["distance"]), index=valid["load_id"])
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(pred).round(2)
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv(OUT_PRED, index=False)
    print(f"  {OUT_PRED.name}: {len(out):,} rows, mean ${out.predicted_rate.mean():,.2f}")

    dec = pd.read_csv(DECEMBER_CSV)
    dec["predicted_rate"] = np.round(to_dollars(np.mean(dec_preds, axis=0), dec["distance"]), 2)
    dec.to_csv(DECEMBER_CSV, index=False)
    print(f"  {DECEMBER_CSV.name}: ${dec.predicted_rate.min():.2f} .. ${dec.predicted_rate.max():.2f}")

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    report.update(features=FEATURES, models=MODELS, seeds=SEEDS, cv_experiment_name=CV_NAME)
    OUT_METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[4/4] Running provided score.py")
    res = subprocess.run([sys.executable, str(ROOT / "score.py"), "--predictions", str(OUT_PRED),
                          "--december-predictions", str(DECEMBER_CSV),
                          "--output-dir", str(ROOT / "scorer_results")],
                         capture_output=True, text=True)
    print(res.stdout or res.stderr)
    if res.returncode != 0:
        sys.exit(res.returncode)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
