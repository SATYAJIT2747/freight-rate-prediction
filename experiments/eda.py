"""Exploratory data analysis: prints the key data-quality statistics and saves
the figures used in the README and the report.

Usage:  python experiments/eda.py
Writes: reports/figures/*.png, experiments/results/eda_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RES = ROOT / "experiments" / "results"

BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9a9a94", "#1f1f1d", "#5f5e58", "#e4e3de"
RELIABLE_MONTHS = [1, 2, 3, 6, 9]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def load():
    t = pd.read_csv(ROOT / "data" / "train_test.csv")
    v = pd.read_csv(ROOT / "data" / "validation.csv")
    for df in (t, v):
        df["date_dt"] = pd.to_datetime(df["date"])
        df["month"] = df["date_dt"].dt.month
        df["lane"] = df["pickup"] + " > " + df["delivery"]
    t["rpm"] = t["posted_rate"] / t["distance"]
    return t, v


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    t, v = load()
    lines = []
    log = lambda s="": (print(s), lines.append(s))  # noqa: E731

    log(f"train rows {len(t):,}  {t.date_dt.min():%Y-%m-%d} .. {t.date_dt.max():%Y-%m-%d}")
    log(f"valid rows {len(v):,}  {v.date_dt.min():%Y-%m-%d} .. {v.date_dt.max():%Y-%m-%d}")
    for name, df in (("train", t), ("valid", v)):
        log(f"[{name}] missing weight {df.weight.isna().sum()}, missing market_index "
            f"{df.market_index.isna().sum()}, negative weight {(df.weight < 0).sum()}, "
            f"weight==47,500 cap {(df.weight.abs() == 47500).sum()}, duplicate ids {df.load_id.duplicated().sum()}")
    q = t.rpm.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
    log(f"RPM quantiles 0.1%/1%/50%/99%/99.9%: {', '.join(f'{x:.2f}' for x in q)}")
    log(f"posted_rate range ${t.posted_rate.min():,.0f} .. ${t.posted_rate.max():,.0f}")
    unseen = (~v.lane.isin(t.lane)).mean()
    log(f"validation loads on lanes never seen in training: {unseen:.1%}")

    # ---- 1. quote_signal regime per month ---------------------------------
    rel = t[t.month.isin(RELIABLE_MONTHS)]
    lane_rpm = rel.groupby(["lane", "equipment"]).rpm.median().rename("lane_rpm")
    rows = []
    for name, df in (("train", t), ("valid", v)):
        d = df.join(lane_rpm, on=["lane", "equipment"]).dropna(subset=["lane_rpm"])
        for m, g in d.groupby("month"):
            rows.append((m, np.corrcoef(g.quote_signal, g.lane_rpm)[0, 1], name))
    reg = pd.DataFrame(rows, columns=["month", "corr", "split"])
    t["ratio"] = t.rpm / t.quote_signal
    share = t.groupby("month").apply(lambda g: ((g.ratio - 1).abs() < 0.02).mean(), include_groups=False)
    log("\nquote_signal regime by month (corr with lane median RPM; share of loads with posted_rate within 2% of distance*quote_signal)")
    for _, r in reg.iterrows():
        s = f"{share[r.month]:.0%}" if r.split == "train" else "n/a (no labels)"
        log(f"  {pd.Timestamp(2025, int(r.month), 1):%b} [{r.split}] corr={r['corr']:+.2f}  exact-share={s}")

    def regime(c):
        return "reliable" if c > 0.3 else ("inverted" if c < -0.3 else "noise")

    colors = {"reliable": BLUE, "inverted": ORANGE, "noise": GRAY}
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=170)
    x = np.arange(len(reg))
    ax.bar(x, reg["corr"], width=0.72, color=[colors[regime(c)] for c in reg["corr"]],
           edgecolor="white", linewidth=2)
    ax.axhline(0, color="#8d8c85", linewidth=0.8)
    ax.set_xticks(x, [pd.Timestamp(2025, m, 1).strftime("%b") for m in reg.month])
    ax.axvspan(9.5, 11.5, color="#f3f2ee", zorder=0)
    ax.text(10.5, 0.83, "validation\n(to predict)", ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_ylim(-0.75, 0.95)
    ax.set_ylabel("corr(quote_signal, lane RPM)")
    ax.set_title("quote_signal switches regime month to month; Nov–Dec look like August (noise)")
    for lab, c in colors.items():
        ax.bar([np.nan], [np.nan], color=c, label=lab)
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "quote_signal_regimes.png")
    plt.close(fig)

    # ---- 2. daily market index vs daily rate deviation --------------------
    t["lane_med"] = t.groupby(["lane", "equipment"]).rpm.transform("median")
    t["dev"] = t.rpm / t.lane_med
    daily = t.groupby("date_dt").agg(mi=("market_index", "mean"), dev=("dev", "median"))
    per_load = t[["market_index", "dev"]].dropna().corr().iloc[0, 1]
    r_daily = daily.corr().iloc[0, 1]
    log(f"\nmarket_index vs lane-relative RPM: per-load corr {per_load:+.2f}, daily-mean corr {r_daily:+.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 3.9), dpi=170)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(daily.mi, daily.dev, s=14, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("daily mean market_index")
    ax.set_ylabel("daily median RPM / lane median")
    ax.set_title(f"Daily market index moves rates (r = {r_daily:.2f})")
    ax.text(0.02, 0.95, f"per-load r = {per_load:.2f} -> average per day", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "daily_market_index.png")
    plt.close(fig)

    # ---- 3. data quality: weight + RPM tails -------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=170)
    a1.hist(t.weight.dropna(), bins=80, color=BLUE, edgecolor="white", linewidth=0.3)
    a1.set_title("weight: negative values + 47,500 lb cap")
    a1.set_xlabel("weight (lb)")
    a1.set_ylabel("loads")
    a1.annotate(f"{(t.weight < 0).sum()} negative", xy=(-30000, 15), xytext=(-40000, 900),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.annotate(f"{(t.weight == 47500).sum():,} at cap", xy=(47500, 1100), xytext=(2000, 2300),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    a2.hist(t.rpm, bins=np.logspace(np.log10(0.3), np.log10(15), 80), color=BLUE,
            edgecolor="white", linewidth=0.3)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_title("rate per mile: heavy tails both ways")
    a2.set_xlabel("posted_rate / distance ($/mile, log)")
    a2.set_ylabel("loads (log)")
    fig.tight_layout()
    fig.savefig(FIG / "data_quality.png")
    plt.close(fig)

    (RES / "eda_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
