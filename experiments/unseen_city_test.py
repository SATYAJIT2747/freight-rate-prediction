"""Validation has 8 cities that never occur in training (~12% of loads).
This simulates that situation: hide 8 random cities from the Jan-Sep training
window, then score October loads that touch them, comparing
  * nearest-city priors (unseen city borrows priors of nearest known city)
  * equipment-median fallback

Usage:  python experiments/unseen_city_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import BASE_FEATURES, FeatureBuilder  # noqa: E402
from src.models import fit_xgb, predict_rpm  # noqa: E402


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "train_test.csv")
    month = pd.to_datetime(raw["date"]).dt.month
    rng = np.random.default_rng(1)
    res = {"nearest-city priors": [], "equipment fallback": []}
    for rep in range(3):
        hide = rng.choice(sorted(raw["pickup"].unique()), 8, replace=False)
        touch = raw["pickup"].isin(hide) | raw["delivery"].isin(hide)
        tr_raw = raw[(month < 10) & ~touch].reset_index(drop=True)
        te_raw = raw[(month == 10) & touch].reset_index(drop=True)
        for name, nearest in (("nearest-city priors", True), ("equipment fallback", False)):
            fb = FeatureBuilder(nearest_city=nearest)
            tr, te = fb.fit_transform(tr_raw), fb.transform(te_raw)
            model = fit_xgb(tr[BASE_FEATURES], (tr.posted_rate / tr.distance).to_numpy(), n_estimators=450)
            pred = predict_rpm(model, te[BASE_FEATURES]) * te["distance"]
            res[name].append(float(np.abs(pred - te["posted_rate"]).mean()))
        print(f"rep {rep}: {len(te_raw):,} Oct loads | " + " | ".join(f"{k} ${v[-1]:.1f}" for k, v in res.items()))
    lines = [f"{k}: mean MAE ${np.mean(v):.2f} over 3 random 8-city holdouts" for k, v in res.items()]
    print("\n".join(lines))
    (ROOT / "experiments" / "results" / "unseen_city.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
