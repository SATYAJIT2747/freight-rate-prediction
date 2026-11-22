"""Data cleaning and feature engineering shared by training, experiments and EDA.

Every statistic that depends on the target (lane priors) or on training-set
distributions (imputation medians) is fitted on the training fold only and
then applied to the evaluation fold / validation set via ``FeatureBuilder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8
CATEGORICALS = ["equipment", "pickup", "delivery", "lane"]

# Features used by the final model.
#  * quote_signal is excluded: its relationship to the target flips between
#    months and it is pure noise in the Nov-Dec validation period.
#  * market_index is excluded: its *level* drifts relative to rates (Sep has the
#    lowest index of the year but normal rates), so models that use it are
#    biased out-of-time. See experiments/results/summary.md for the ablation.
BASE_FEATURES = [
    "distance", "log_distance", "haversine_miles", "circuity",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "weight", "weight_missing", "weight_negative",
    "day_of_week",
    "lane_rpm_prior", "lane_count", "pickup_rpm_prior", "delivery_rpm_prior",
    "equipment", "pickup", "delivery",
]
MARKET_FEATURES = ["market_index", "daily_market_index", "market_index_missing"]
MARKET_REL_FEATURES = ["market_index_rel_month"]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that needs no fitted statistics."""
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"])
    # ~0.6% of weights are negative but otherwise plausible in magnitude and
    # their rates match same-magnitude positive weights -> sign error.
    out["weight_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["lane"] = out["pickup"] + " > " + out["delivery"]
    return out


def city_table(df: pd.DataFrame) -> pd.DataFrame:
    """city -> (lat, lon). Each city has exactly one coordinate pair in the data."""
    a = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    b = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([a, b]).drop_duplicates("city").set_index("city").sort_index()


def nearest_known_city(lat: np.ndarray, lon: np.ndarray, known: pd.DataFrame) -> np.ndarray:
    d = haversine_miles(lat[:, None], lon[:, None], known["lat"].to_numpy()[None, :],
                        known["lon"].to_numpy()[None, :])
    return known.index.to_numpy()[d.argmin(axis=1)]


@dataclass
class FeatureBuilder:
    """Fit on a training frame, then transform any frame (train/test/validation)."""

    smoothing: float = 20.0  # shrinkage strength for lane/city target priors
    nearest_city: bool = True  # unseen city -> borrow priors of nearest known city
    stats: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        tr = clean(train)
        self.stats = {
            "weight_by_equipment": tr.groupby("equipment")["weight"].median().to_dict(),
            "weight_global": float(tr["weight"].median()),
            "market_index_global": float(tr["market_index"].median()),
            "categories": {c: sorted(tr[c].unique()) for c in CATEGORICALS},
            "city_coords": city_table(tr),
            "priors": self._fit_priors(tr),
        }
        return self

    def fit_transform(self, train: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
        """Fit, then encode training rows with OUT-OF-FOLD lane priors so the
        model never sees a prior that was computed from the row's own label."""
        self.fit(train)
        tr = clean(train).reset_index(drop=True)
        fold = np.random.default_rng(seed).integers(0, n_folds, len(tr))
        full = train.reset_index(drop=True)
        parts = []
        for k in range(n_folds):
            # transform the full frame so same-day market aggregates use every load
            encoded = self.transform(full, priors=self._fit_priors(tr[fold != k]))
            parts.append(encoded[fold == k])
        return pd.concat(parts).sort_index()

    def _fit_priors(self, tr: pd.DataFrame) -> dict:
        tr = tr.assign(rpm=tr["posted_rate"] / tr["distance"])
        global_rpm = float(tr["rpm"].median())
        return {
            "global_rpm": global_rpm,
            "equipment": tr.groupby("equipment")["rpm"].median(),
            "lane": self._smoothed_prior(tr, ["lane", "equipment"], global_rpm),
            "pickup": self._smoothed_prior(tr, ["pickup", "equipment"], global_rpm),
            "delivery": self._smoothed_prior(tr, ["delivery", "equipment"], global_rpm),
        }

    def _smoothed_prior(self, tr: pd.DataFrame, keys: list[str], global_rpm: float) -> pd.DataFrame:
        # Median is robust to the heavy-tailed RPM outliers; shrink toward the
        # equipment-level median when a group has few loads.
        eq_median = tr.groupby("equipment")["rpm"].median()
        g = tr.groupby(keys)["rpm"].agg(["median", "size"]).reset_index()
        base = g["equipment"].map(eq_median).fillna(global_rpm)
        w = g["size"] / (g["size"] + self.smoothing)
        g["prior"] = w * g["median"] + (1 - w) * base
        return g[keys + ["prior", "size"]]

    def transform(self, df: pd.DataFrame, priors: dict | None = None) -> pd.DataFrame:
        s = self.stats
        pr = priors or s["priors"]
        out = clean(df)
        index = out.index

        # --- imputation (training-fold statistics only) ---
        out["weight"] = out["weight"].fillna(out["equipment"].map(s["weight_by_equipment"]))
        out["weight"] = out["weight"].fillna(s["weight_global"])
        # Per-load market_index is noisy; the same-day mean is the real market
        # signal (daily corr with rate deviation ~0.69). It uses only features
        # of loads on the same date, never labels.
        daily = out.groupby("date_dt")["market_index"].transform("mean")
        out["daily_market_index"] = daily.fillna(s["market_index_global"])
        out["market_index"] = out["market_index"].fillna(out["daily_market_index"])
        # daily index relative to its own calendar-month mean (removes level drift)
        month_mean = out.groupby(out["date_dt"].dt.to_period("M"))["daily_market_index"].transform("mean")
        out["market_index_rel_month"] = out["daily_market_index"] - month_mean

        # --- geometry ---
        out["haversine_miles"] = haversine_miles(out["pickup_lat"], out["pickup_lon"],
                                                 out["delivery_lat"], out["delivery_lon"])
        out["circuity"] = out["distance"] / (out["haversine_miles"] + 1e-6)
        out["log_distance"] = np.log1p(out["distance"])
        out["day_of_week"] = out["date_dt"].dt.day_of_week

        # --- lane / city priors (target statistics from training fold) ---
        # Validation contains 8 cities never seen in training. Their priors are
        # borrowed from the geographically nearest known city.
        known = s["city_coords"]
        for side in ("pickup", "delivery"):
            ref = out[side].copy()
            unseen = ~ref.isin(known.index)
            out[f"{side}_unseen"] = unseen.astype(int)
            if self.nearest_city and unseen.any():
                ref[unseen] = nearest_known_city(out.loc[unseen, f"{side}_lat"].to_numpy(),
                                                 out.loc[unseen, f"{side}_lon"].to_numpy(), known)
            out[f"{side}_ref"] = ref
        out["lane_ref"] = out["pickup_ref"] + " > " + out["delivery_ref"]

        lane = pr["lane"].rename(columns={"lane": "lane_ref", "prior": "lane_rpm_prior", "size": "lane_count"})
        out = out.merge(lane, on=["lane_ref", "equipment"], how="left")
        for side in ("pickup", "delivery"):
            p = pr[side].rename(columns={side: f"{side}_ref", "prior": f"{side}_rpm_prior"}).drop(columns="size")
            out = out.merge(p, on=[f"{side}_ref", "equipment"], how="left")
        out.index = index  # merge resets the index; restore it
        eq_fallback = out["equipment"].map(pr["equipment"]).astype(float)
        for col in ("lane_rpm_prior", "pickup_rpm_prior", "delivery_rpm_prior"):
            out[col] = out[col].fillna(eq_fallback).fillna(pr["global_rpm"])
        out["lane_count"] = out["lane_count"].fillna(0)

        for c in CATEGORICALS:
            out[c] = pd.Categorical(out[c], categories=s["categories"][c])
        return out
