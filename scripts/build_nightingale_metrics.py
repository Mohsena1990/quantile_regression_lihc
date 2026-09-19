"""Nightingale (stacked polar area) charts: one wedge per category, radiating
outward from the centre as separated, segmented stacked blocks -- not a
continuous radar line. Three figures:

  1. Regression evaluation metrics -- one wedge per R-code (R1-R3), each wedge a
     stack of its continuous cost-model metrics (MAE, MAPE, RMSE, R^2,
     Adjusted R^2, pinball loss, calibration).
  2. Classification evaluation metrics -- one wedge per (R-code, classifier)
     combo, each wedge a stack of its four-class screening metrics (Accuracy,
     Precision, Recall/Sensitivity, F1, ROC-AUC, 1/(1+Log Loss)).
  3. SHAP feature effectiveness -- one wedge per top-N feature (A0 config,
     mean over families), each wedge a stack of its per-country SHAP share, so
     a feature's total wedge height is its overall importance and the stack
     shows which countries it matters most in.

Every metric is min-max normalised across wedges and, for loss-type metrics,
inverted (1 - x) so every segment reads "bigger = better" and stacks are
comparable in the same units. Source values stay in the underlying CSV;
normalisation is for the chart only.

Outputs:
  outputs/figures/nightingale_regression_metrics.png/.pdf
  outputs/figures/nightingale_classification_metrics.png/.pdf
  outputs/figures/nightingale_shap_features.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import RETAINED_COUNTRY_NAMES
SURFACE, INK, INK_MUTED = "#ffffff", "#1a1a1a", "#7a7570"
# Pastel orange / pink / grey family, interpolated so any segment count gets a
# distinct, harmonious tone from the same three-hue corporate palette.
PASTEL_STOPS = ["#f6b28c", "#f2a0b8", "#cfc9c0", "#f8c9a8", "#e9aecb", "#b7b2a9", "#fadcc0", "#dc8fae"]


def pastel(n: int) -> list[str]:
    return [PASTEL_STOPS[i % len(PASTEL_STOPS)] for i in range(n)]


def normalise(frame: pd.DataFrame, metrics: list[str], loss_metrics: set[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for m in metrics:
        lo, hi = frame[m].min(), frame[m].max()
        scaled = (frame[m] - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=frame.index)
        out[m] = (1 - scaled) if m in loss_metrics else scaled
    return out


def nightingale(wedge_labels: list[str], stacks: pd.DataFrame, segment_labels: list[str], *,
                 title: str, path: Path, wedge_gap: float = 0.12) -> None:
    """`stacks` has one row per wedge (aligned with wedge_labels) and one column per
    segment (aligned with segment_labels), values already normalised to a common scale."""
    n = len(wedge_labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = (2 * np.pi / n) - wedge_gap
    colors = pastel(len(segment_labels))

    fig = plt.figure(figsize=(13, 12))
    fig.patch.set_facecolor(SURFACE)
    # A fixed, generously-margined axes position, not tight_layout/subplots_adjust: this
    # chart is saved with bbox_inches=None (see below) because matplotlib's tight-bbox text
    # measurement is unreliable for polar tick labels at the 0-degree position (verified:
    # it clips the last glyph or two regardless of pad_inches, since pad_inches only widens
    # an already-undersized computed bbox rather than fixing the measurement itself).
    ax = fig.add_axes((0.16, 0.1, 0.68, 0.78), projection="polar")
    ax.tick_params(axis="x", pad=14)
    ax.set_facecolor(SURFACE)
    bottom = np.zeros(n)
    for seg, color in zip(segment_labels, colors):
        vals = stacks[seg].to_numpy()
        ax.bar(theta, vals, width=width, bottom=bottom, color=color, edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label=seg)
        bottom += vals
    ax.set_xticks(theta)
    ax.set_xticklabels(wedge_labels, fontsize=10, color=INK, fontweight="medium")
    for label in ax.get_xticklabels():
        label.set_clip_on(False)  # polar axes otherwise clip tick-label glyphs at the plot boundary
    ax.set_yticklabels([])
    ax.set_ylim(0, bottom.max() * 1.08 if bottom.max() > 0 else 1)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#e7e3dd", linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=13, color=INK, pad=28, fontweight="semibold")
    fig.legend(loc="lower center", ncol=min(len(segment_labels), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.015), labelcolor=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_regression() -> None:
    d = pd.read_csv(ROOT / "outputs/tables/table_evaluation_summary.csv")
    d = d[d.family.ne("M0")].drop_duplicates("regression_code").sort_values("regression_code")
    if d.empty:
        print("SKIP regression Nightingale: no non-M0 rows in table_evaluation_summary.csv")
        return
    metrics = {"mae_eur": "MAE", "mape": "MAPE", "rmse_eur": "RMSE", "r2": "R²",
               "adjusted_r2": "Adj. R²", "pinball_loss": "Pinball", "empirical_coverage_error": "Calibration"}
    loss = {"mae_eur", "mape", "rmse_eur", "pinball_loss", "empirical_coverage_error"}
    d = d.assign(empirical_coverage_error=d.empirical_coverage_error.abs())
    norm = normalise(d.reset_index(drop=True), list(metrics), loss)
    norm.columns = [metrics[c] for c in norm.columns]
    wedge_labels = [f"{r} ({f})" for r, f in zip(d.regression_code, d.family)]
    nightingale(wedge_labels, norm.reset_index(drop=True), list(metrics.values()),
                title="Regression evaluation metrics by method",
                path=ROOT / "outputs/figures/nightingale_regression_metrics")


def build_classification() -> None:
    d = pd.read_csv(ROOT / "outputs/tables/table_evaluation_summary.csv").sort_values(["regression_code", "classifier_code"])
    if d.empty:
        print("SKIP classification Nightingale: table_evaluation_summary.csv is empty")
        return
    d = d.assign(inv_log_loss=1 / (1 + d.log_loss))
    metrics = {"accuracy": "Accuracy", "precision_macro": "Precision", "recall_sensitivity_macro": "Recall",
               "macro_f1": "F1", "roc_auc_macro_ovr": "ROC-AUC", "inv_log_loss": "1/(1+LogLoss)"}
    norm = normalise(d.reset_index(drop=True), list(metrics), set())
    norm.columns = [metrics[c] for c in norm.columns]
    wedge_labels = [f"{r}{c}" for r, c in zip(d.regression_code, d.classifier_code)]
    nightingale(wedge_labels, norm.reset_index(drop=True), list(metrics.values()),
                title="Classification evaluation metrics by regression x classifier combo",
                path=ROOT / "outputs/figures/nightingale_classification_metrics")


def build_shap(top_n: int = 10) -> None:
    d = pd.read_csv(ROOT / "outputs/tables/table_shap_feature_effectiveness.csv")
    countries = list(RETAINED_COUNTRY_NAMES)
    d = d[d.config.eq("A0") & d.scope.isin(countries)]
    if d.empty:
        print("SKIP SHAP Nightingale: no A0/per-country rows in table_shap_feature_effectiveness.csv")
        return
    agg = d.groupby(["feature", "scope"]).importance_share.mean().unstack("scope").fillna(0.0)
    agg = agg.reindex(columns=countries, fill_value=0.0)
    top = agg.assign(_total=agg.sum(axis=1)).sort_values("_total", ascending=False).head(top_n).drop(columns="_total")
    nightingale(list(top.index), top.reset_index(drop=True), countries,
                title=f"SHAP feature effectiveness -- top {top_n} predictors, by country",
                path=ROOT / "outputs/figures/nightingale_shap_features")


def main() -> None:
    build_regression()
    build_classification()
    build_shap()
    print("Wrote outputs/figures/nightingale_{regression_metrics,classification_metrics,shap_features}.png/.pdf")


if __name__ == "__main__":
    main()
