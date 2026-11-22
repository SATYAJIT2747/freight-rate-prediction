"""Builds reports/Freight_Rate_Report.pdf from the experiment outputs.

Run after `python experiments/eda.py`, `python experiments/run_experiments.py`
and `python train.py`.  Requires: pip install reportlab
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RES = ROOT / "experiments" / "results"
OUT = ROOT / "reports" / "Freight_Rate_Report.pdf"
TEAL, INK, MUTED, RULE = colors.HexColor("#064A56"), colors.HexColor("#1f1f1d"), colors.HexColor("#5f5e58"), colors.HexColor("#d9d8d2")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=20, leading=24, textColor=TEAL, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10, textColor=MUTED, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13.5, leading=17, textColor=TEAL, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.8, leading=13.6, textColor=INK, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.3, leading=11, textColor=MUTED)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.4, leading=10.5, spaceAfter=0)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=10) for i in items],
                        bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def table(rows, widths, bold_row=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f4")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#fdf3e7")))
    t.setStyle(TableStyle(style))
    return t


def fig(path: Path, width_cm: float):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def main() -> None:
    cv = pd.read_csv(RES / "cv_results.csv")
    final = json.loads((RES / "final_model_metrics.json").read_text())
    final_name = final["cv_experiment_name"]
    piv = cv.pivot_table(index="experiment", columns="month", values="mae")
    piv["mean"] = piv.mean(axis=1)
    piv["bias"] = cv.groupby("experiment")["bias"].mean()
    piv = piv.sort_values("mean")
    prev = "LightGBM + quote_signal + market_index (previous approach)"
    fm, pm = piv.loc[final_name], piv.loc[prev]

    s = []
    s += [Paragraph("Freight Rate Prediction — Validation Approach &amp; Results", H1),
          Paragraph("Machine Learning Engineer assessment · US full-truckload posted-rate model", SUB)]

    s += [Paragraph("1. Summary", H2), bullets([
        "Development data: 48,000 labelled loads, 1 Jan – 31 Oct 2025. Prediction set: 12,000 loads, "
        "1 Nov – 31 Dec 2025. Because the task is to predict the future, every split is a "
        "<b>time-based</b> split; a random K-fold would leak future market conditions into training.",
        "The key exploratory finding: <b>quote_signal changes meaning month to month</b>. In Jan, Feb, Mar, Jun "
        "and Sep it equals the realised rate-per-mile for ~93% of loads. In Apr, May, Jul and Oct it is "
        "inversely related to it. In August, and in the entire Nov–Dec prediction set, it is noise. A model "
        "trained on it learns a shortcut that does not exist in the period we must predict.",
        "Final model: equal-weight blend of XGBoost, LightGBM and CatBoost, each predicting rate-per-mile "
        "with an L1 (MAE) objective and averaged over 3 seeds. Features: lane, geography, equipment, weight "
        "and day of week. No quote_signal and no market_index.",
        f"On the three most recent folds it beats the previous quote_signal-based approach every month: "
        f"Aug <b>${fm[8]:.0f}</b> vs ${pm[8]:.0f}, Sep <b>${fm[9]:.0f}</b> vs ${pm[9]:.0f}, "
        f"Oct <b>${fm[10]:.0f}</b> vs ${pm[10]:.0f}. Its weak spot is June (see section 4).",
    ])]

    s += [Paragraph("2. How the data was split and validated", H2),
          Paragraph("<b>Expanding-window, month-ahead cross-validation.</b> For each test month M in "
                    "Jun – Oct 2025, the model is trained on every load dated before M and scored on all loads in M. "
                    "This mirrors production exactly: train on the past, predict the next month. It also gives five "
                    "independent out-of-time estimates instead of one.", BODY)]
    split_rows = [["Fold", "Train window", "Test window", "Test loads"]]
    for m in sorted(cv.month.unique()):
        n = int(cv[(cv.month == m)].n.iloc[0])
        split_rows.append([f"{m - 5}", f"Jan 1 – {pd.Timestamp(2025, m, 1) - pd.Timedelta(days=1):%b %d}",
                           f"{pd.Timestamp(2025, m, 1):%b} 2025", f"{n:,}"])
    split_rows.append(["Final", "Jan 1 – Oct 31 (all 48,000)", "Nov – Dec validation.csv", "12,000"])
    s += [table(split_rows, [1.6 * cm, 5.2 * cm, 5.2 * cm, 2.6 * cm], bold_row=len(split_rows) - 1),
          Spacer(1, 6),
          Paragraph("<b>Design choices and leakage controls.</b>", BODY),
          bullets([
              "Why per-month folds rather than one split: quote_signal changes regime by calendar month, so "
              "a single window mixes regimes and hides the failure. Scoring each month separately exposes it.",
              "Early stopping uses the last 10% of the <i>training</i> window by date, never the test month. "
              "The number of boosting rounds for the final model is fixed this way on a Jan–Sep fit and "
              "checked on October.",
              "Imputation medians and lane / city target encodings are fitted on the training window only. "
              "Training rows receive <i>out-of-fold</i> lane encodings, so no row sees a prior built from its own label.",
              "Same-day aggregates (daily mean market_index, used only in ablations) are computed from features, never labels.",
              "Unseen cities: validation contains 8 cities absent from training (~1,450 loads). Their priors are borrowed "
              "from the nearest known city. This was validated by hiding 8 random cities from a Jan–Sep fit and "
              "scoring October: MAE $161 vs $168 for a plain equipment-level fallback.",
              "Train/validation consistency checks: identical schema, every city has exactly one "
              "coordinate pair (64 cities in training, 72 incl. validation), no duplicate load IDs, and no overlap in dates. 12.2% of validation loads are on lanes "
              "never seen in training, so city-level priors and coordinates are included.",
              "Regime check on validation without labels: correlation of quote_signal with each lane's historical "
              "rate-per-mile is +0.7 in reliable months, −0.6 in inverted months, and ≈0 in August, November and December.",
          ])]

    s += [KeepTogether([Paragraph("3. Data quality issues and fixes", H2),
          fig(FIG / "quote_signal_regimes.png", 16.5), Spacer(1, 4)])]
    s += [table([
        ["Issue", "Evidence", "Treatment"],
        ["quote_signal regime switching", "Corr with lane RPM: +0.7 / −0.6 / 0 by month; Nov–Dec ≈ 0",
         "Dropped from the model (ablation below)"],
        ["market_index level drift", "Strong within-month signal, but Sep has the lowest index of the year "
         "with normal rates; models using it are biased out-of-time", "Dropped (ablation below)"],
        ["Negative weights", "292 train / 145 validation; rates match same-magnitude positive weights",
         "Sign fixed (abs) + indicator flag"],
        ["Weight capped at 47,500 lb", "1,191 train loads sit exactly at the cap", "Kept as-is (legal max)"],
        ["Missing weight / market_index", "300 / 374 train, 165 / 249 validation",
         "Equipment median imputation + missing flags"],
        ["8 cities only in validation", "Chicago, Charlotte, San Diego, Laredo, …; ~1,450 loads",
         "Coordinates + nearest-known-city priors"],
        ["Corrupted labels", "~1.3% of loads at 0.2–0.6× or 2–5× their lane's typical rate, all months",
         "L1 (MAE) objective and median-based lane priors (robust to them)"],
    ], [4.0 * cm, 6.6 * cm, 6.0 * cm])]
    s += [Spacer(1, 6), fig(FIG / "data_quality.png", 16.5)]

    s += [PageBreak(), Paragraph("4. Model selection (time-series CV, MAE in $)", H2)]
    rows = [["Experiment", "Jun", "Jul", "Aug", "Sep", "Oct", "Mean", "Bias"]]
    for name, r in piv.iterrows():
        rows.append([name] + [f"{r[m]:.1f}" for m in (6, 7, 8, 9, 10)] + [f"<b>{r['mean']:.1f}</b>", f"{r['bias']:+.0f}"])
    s += [table(rows, [6.6 * cm] + [1.35 * cm] * 7, bold_row=list(piv.index).index(final_name) + 1),
          Paragraph("Bias = mean(prediction − actual). Lower MAE is better. Highlighted row = final model.", SMALL),
          Paragraph("<b>Why this model.</b>", BODY),
          bullets([
              "Target = rate-per-mile, not raw dollars. Rates scale with distance, so modelling $/mile makes "
              "lanes of different length comparable. Dollars are recovered as RPM × distance.",
              "L1 objective because the metric is MAE and ~1.3% of labels are corrupted. The median is robust to "
              "them; squared loss is not.",
              "Gradient-boosted trees handle the non-linear lane × equipment × weight structure and the "
              "categorical cities natively. The three boosters score within ~$3 of each other, and their equal "
              "blend is the best of them on every fold from August on, so the blend is used.",
              "<b>Selection by the folds that resemble the prediction period.</b> Mean CV MAE is within ~$3 across "
              "the top models, so the mean alone cannot decide. Aug–Oct are the most recent folds (the most training "
              "history), and August is the only month with the same noise-regime quote_signal as Nov–Dec. It is the best "
              "row in each of those months and beats the quote_signal models by $13–79 per month.",
              "<b>Known trade-off: June.</b> Rates jumped ~4% in June and only market_index anticipated it. Models "
              "without it score ~$220 MAE in June vs ~$140 with it. The same market_index then misled models in September "
              "(lowest index of the year, normal rates), so it was judged too unstable to rely on for Nov–Dec.",
              "Negative bias is expected under L1: the MAE-optimal prediction is the median, and most corrupted labels "
              "lie far above it.",
          ])]

    ho = final["october_holdout"]
    s += [Paragraph("5. Final training", H2), Paragraph(
        f"Boosting rounds were fixed by early stopping on a Jan–Sep fit (October holdout MAE: blend "
        f"${ho['blend']:.2f}; XGBoost ${ho['xgb']:.2f}, LightGBM ${ho['lgb']:.2f}, CatBoost ${ho['cat']:.2f}), "
        f"plus 10% for the larger refit. The "
        f"model was then refit on all 48,000 labelled loads with {len(final['seeds'])} seeds and the predictions "
        f"averaged. The 12,000 validation predictions are written to <font name='Courier'>validation_predictions.csv</font> "
        f"(columns load_id, predicted_rate), and the provided score.py validates the file.", BODY)]

    s += [KeepTogether([Paragraph("6. December prediction chart (produced by score.py)", H2),
          fig(ROOT / "scorer_results" / "candidate_december.png", 17.2),
          Paragraph("Fixed inputs: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes. "
                    "City coordinates come from the dataset's city table. Because the final model deliberately "
                    "ignores the drifting market level, day-to-day movement comes from the learned day-of-week "
                    "pattern: mid-week loads price slightly higher than weekend loads, as seen in the training data. "
                    "Level: ~$820 for 360 miles = $2.28/mile, in line with this lane's history.", SMALL)])]

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                      topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                      title="Freight Rate Prediction Report").build(s)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
