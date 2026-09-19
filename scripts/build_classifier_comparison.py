"""Table + figure comparing the two screening classifiers' evaluation metrics.

Aggregates every AVAILABLE development-stage screening run (all M0/A0/A1
measurements -- see scripts/run_screening_matrix.py) per classifier, so the
comparison reflects the full breadth of measurement families rather than one
cherry-picked config.

Outputs:
  outputs/tables/table_classifier_comparison.csv            (per classifier, mean/std/n)
  outputs/tables/table_classifier_comparison_per_class.csv  (per classifier x risk class, long form,
                                                               F1 and recall for all four risk classes)
  outputs/figures/classifier_comparison.png/pdf             (small multiples, one box per classifier;
                                                               the last two panels break F1 and recall
                                                               out by risk class instead of collapsing
                                                               to a single Double-Risk-only number)
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, recall_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import classifier_code

# Fixed categorical assignment (never re-ordered by rank/score) -- validated pair from the
# project's dataviz reference palette (slots 1/2: blue/orange). Labels are the short
# C0-C1 codes (see lib/reporting/codes.py) so figure axes/titles stay compact; the
# code->full-name mapping is restated in the figure's own caption and in
# outputs/tables/table_symbol_legend.csv.
MODEL_ORDER = ["catboost", "hierarchical_catboost"]
MODEL_LABEL = {name: classifier_code(name) for name in MODEL_ORDER}
MODEL_FULL_NAME = {"catboost": "CatBoost", "hierarchical_catboost": "Hierarchical CatBoost"}
MODEL_COLOR = {"catboost": "#2a78d6", "hierarchical_catboost": "#eb6834"}
LEGEND_CAPTION = "  |  ".join(f"{classifier_code(name)}={MODEL_FULL_NAME[name]}" for name in MODEL_ORDER)
METRICS = ["macro_f1", "balanced_accuracy", "log_loss"]
METRIC_LABEL = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced accuracy", "log_loss": "Log loss (lower is better)"}
RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
RISK_SHORT = ["No", "Income", "Expend.", "Double"]
SURFACE, GRID, INK_MUTED = "#fcfcfb", "#e1e0d9", "#898781"


def load_per_class_metrics(d: pd.DataFrame) -> pd.DataFrame:
    """Per-class F1 and recall for every AVAILABLE development config, from the raw
    prediction parquets (columns UID/truth/prediction/prob_0..3/country) -- the
    classifier-comparison metrics table only carries macro_f1 and a Double-Risk-only
    F1/recall, so a genuine per-class breakdown across all four risk classes has to be
    recomputed from the raw rows.
    """
    pred_dir = ROOT / "artifacts/predictions"
    rows = []
    for row in d.itertuples():
        p = pred_dir / f"{row.measurement}_{row.model}_{row.evaluation_stage}.parquet"
        if not p.exists():
            continue
        pdf = pd.read_parquet(p, columns=["truth", "prediction"])
        per_class_f1 = f1_score(pdf.truth, pdf.prediction, labels=[0, 1, 2, 3], average=None, zero_division=0)
        per_class_recall = recall_score(pdf.truth, pdf.prediction, labels=[0, 1, 2, 3], average=None, zero_division=0)
        for cls_idx, cls_name in enumerate(RISK_ORDER):
            rows.append({"model": row.model, "class": cls_name,
                         "f1": per_class_f1[cls_idx], "recall": per_class_recall[cls_idx]})
    return pd.DataFrame(rows)


def load_development_runs() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "outputs/tables/screening_development_all.csv")
    return d[d.status.eq("AVAILABLE")]


def build_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in d.groupby("model"):
        row = {"classifier": model, "n_configs": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
        rows.append(row)
    table = pd.DataFrame(rows)
    table["classifier"] = pd.Categorical(table["classifier"], categories=MODEL_ORDER, ordered=True)
    return table.sort_values("classifier")


def _style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)


def _plot_per_class_panel(ax, per_class: pd.DataFrame, value_col: str, title: str) -> None:
    """One group of 2 (one per classifier) boxes per risk class, so all four classes
    are visible side by side rather than only the Double-Risk class.
    """
    offsets = {m: off for m, off in zip(MODEL_ORDER, (-0.16, 0.16))}
    for m in MODEL_ORDER:
        data = [per_class.loc[per_class.model.eq(m) & per_class["class"].eq(cls), value_col].dropna().to_numpy()
                for cls in RISK_ORDER]
        positions = [i + 1 + offsets[m] for i in range(len(RISK_ORDER))]
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.2,
                         medianprops={"color": "#0b0b0b", "linewidth": 1.4},
                         flierprops={"markersize": 2.5, "markeredgecolor": INK_MUTED})
        for patch in bp["boxes"]:
            patch.set_facecolor(MODEL_COLOR[m]); patch.set_edgecolor(SURFACE); patch.set_linewidth(1.0); patch.set_alpha(0.85)
    ax.set_xlim(0.4, len(RISK_ORDER) + 0.6)
    ax.set_xticks(range(1, len(RISK_ORDER) + 1))
    ax.set_xticklabels(RISK_SHORT, fontsize=8, rotation=20, ha="right")
    ax.set_title(title, fontsize=9.5)
    _style_axis(ax)


def plot_comparison(d: pd.DataFrame, per_class: pd.DataFrame) -> None:
    n_panels = len(METRICS) + 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6))
    for ax, metric in zip(axes, METRICS):
        data = [d.loc[d.model.eq(m), metric].dropna().to_numpy() for m in MODEL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "#0b0b0b", "linewidth": 1.6},
                         flierprops={"markersize": 3, "markeredgecolor": INK_MUTED})
        for patch, m in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLOR[m]); patch.set_edgecolor(SURFACE); patch.set_linewidth(1.2); patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=8, rotation=20, ha="right")
        ax.set_title(METRIC_LABEL[metric], fontsize=9.5)
        _style_axis(ax)

    _plot_per_class_panel(axes[-2], per_class, "f1", "Per-class F1 (all four classes)")
    _plot_per_class_panel(axes[-1], per_class, "recall", "Per-class recall (all four classes)")

    n_measurements = d.measurement.nunique()
    fig.suptitle(f"Screening classifier comparison across all {len(d)} development-stage configurations\n"
                  f"({n_measurements} measurements x 2 classifiers)\n"
                  f"{LEGEND_CAPTION}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, ROOT / "outputs/figures/classifier_comparison")
    plt.close(fig)


def main() -> None:
    d = load_development_runs()
    if d.empty:
        print("No AVAILABLE development screening runs found -- run scripts/run_screening_matrix.py first.")
        return
    table = build_table(d)
    table.to_csv(ROOT / "outputs/tables/table_classifier_comparison.csv", index=False)
    per_class = load_per_class_metrics(d)
    per_class.to_csv(ROOT / "outputs/tables/table_classifier_comparison_per_class.csv", index=False)
    plot_comparison(d, per_class)
    print(f"Wrote classifier comparison for {d.model.nunique()} classifiers across {len(d)} configs -> "
          f"outputs/tables/table_classifier_comparison.csv, "
          f"outputs/tables/table_classifier_comparison_per_class.csv, "
          f"outputs/figures/classifier_comparison.png/pdf")


if __name__ == "__main__":
    main()
