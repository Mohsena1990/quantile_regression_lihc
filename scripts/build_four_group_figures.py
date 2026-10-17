"""The four groups and how well the models separate them -- figures rebuilt for the current design (five countries, study lines).

(1) four_groups_space   Every labelled household on the two axes that define its group: income bracket (jittered) against
                        annual energy cost relative to its expenditure line (log), per country, for the P80 baseline and the
                        need line. Lines: the income cut-off and cost = line.
(2) four_groups_prediction
                        Out-of-fold Ensemble+ (need line), averaged over the three repeats. Income is observed, so the
                        predicted P(low income) is ~0 or 1 and every error lies on the expenditure axis; the figure shows
                        P(high expenditure) = P(Expenditure Risk) + P(Double Risk) by TRUE group, the two confused pairs
                        side by side (No vs Expenditure Risk; Income vs Double Risk), per country.
(3) four_groups_confusion
                        Row-normalised confusion matrices of Ensemble+ for the four lines (all five countries; counts summed
                        over the three repeats; argmax of the class probabilities).
Out-of-fold probabilities come from artifacts/identification/main_results.pkl (scripts/build_feature_stages.py), so the
figures match Table 2 and the per-group scores.

Outputs: outputs/figures/four_groups_space, four_groups_prediction, four_groups_confusion;
         outputs/tables/table_four_groups_confusion.csv
"""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib.identification.data import load_development_frame
from lib.identification.lines import CLASS_NAMES, study_labels
from lib.reporting.figures import save_figure
from build_feature_stages import assemble

TABLES, FIGS = ROOT / "outputs/tables", ROOT / "outputs/figures"
COUNTRIES = ["Bulgaria", "Hungary", "Italy", "Poland", "Spain"]
LINE_NAMES = {"p80": "P80 line (baseline)", "needqr": "Need line", "needqr_noclim": "Need line, no climate",
              "burden": "Energy-burden line"}
GROUP_COLOURS = {0: "#b5b3ad", 1: "#eda100", 2: "#2a78d6", 3: "#d1342f"}   # as in the hidden under-heating figure
INK, MUTED, GRID = "#2b2a27", "#6b6a66", "#e9e8e4"
ENSEMBLE_PLUS = ("variant", "ensemble_plus", "all")
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def legend_handles() -> list:
    return [Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=GROUP_COLOURS[k], markeredgecolor="white",
                   label=CLASS_NAMES[k]) for k in range(4)]


def plot_space(labs: dict) -> None:
    rng = np.random.default_rng(3)
    rows = [("p80", "P80 line (baseline)"), ("needqr", "Need line")]
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.9), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.28})
    for r, (line, rname) in enumerate(rows):
        frame, spec = labs[line]
        for c_i, c in enumerate(COUNTRIES):
            ax = axes[r, c_i]
            g = frame[frame.country == c]
            rel = (g.cost / g.thr.clip(lower=1)).clip(0.03, 30)
            x = g.income_bracket + rng.uniform(-0.3, 0.3, len(g))
            for k in (0, 1, 2, 3):
                m = g.y == k
                ax.scatter(x[m], rel[m], s=4, color=GROUP_COLOURS[k], alpha=0.75, lw=0, rasterized=True)
            ax.set_yscale("log"); ax.set_ylim(0.03, 30)
            ax.axhline(1, color=INK, lw=0.7); ax.axvline(spec["cutoff"] + 0.5, color=INK, lw=0.7)
            ax.set_xticks([1, 3, 5, 7, 9]); ax.set_xlim(0.4, 10.6)
            ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(c, fontsize=8, color=INK, loc="left")
            if c_i == 0:
                ax.set_ylabel(f"{rname}\ncost / line (log)")
            else:
                ax.tick_params(labelleft=False)
            if r == 1 and c_i == 2:
                ax.set_xlabel("Income bracket (1 = lowest; vertical line: low-income cut-off)")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    save_figure(fig, FIGS / "four_groups_space"); plt.close(fig)


def plot_prediction(labs: dict, proba: dict) -> None:
    """Income is observed, so P(low income) is ~0 or 1 and every error lies on the expenditure axis: show how far
    P(high expenditure) separates the two pairs that are confused (No vs Expenditure Risk; Income vs Double Risk)."""
    frame, _ = labs["needqr"]
    p = proba["needqr"][ENSEMBLE_PLUS].mean(axis=0)
    p_high = p[:, 2] + p[:, 3]
    y = frame.y.to_numpy()
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.3), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, c in zip(axes, COUNTRIES):
        m_c = (frame.country == c).to_numpy()
        for pos, k in enumerate((0, 2, 1, 3)):
            m = m_c & (y == k)
            v = p_high[m]
            ax.scatter(pos + rng.uniform(-0.28, 0.28, m.sum()), v, s=3.5, color=GROUP_COLOURS[k], alpha=0.75, lw=0,
                       rasterized=True)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([pos - 0.32, pos + 0.32], [med, med], color=INK, lw=1.4, zorder=4)
            ax.plot([pos, pos], [q1, q3], color=INK, lw=0.8, zorder=4)
        ax.axhline(0.5, color=MUTED, lw=0.6, ls="--"); ax.axvline(1.5, color=GRID, lw=0.8)
        ax.set_xticks([0.5, 2.5], ["not low\nincome", "low\nincome"]); ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ba = balanced_accuracy_score(frame.y[m_c], p[m_c].argmax(1))
        ax.set_title(f"{c}  (BA {ba:.2f})", fontsize=7.5, color=INK, loc="left")
    axes[0].set_ylabel("Predicted P(high expenditure)")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.24),
               title="True group (bar: median, line: interquartile range)", title_fontsize=7.5)
    save_figure(fig, FIGS / "four_groups_prediction"); plt.close(fig)


def confusion(labs: dict, proba: dict) -> pd.DataFrame:
    rows = []
    for line in LINE_NAMES:
        frame, _ = labs[line]
        y = frame.y.to_numpy()
        for rep, p in enumerate(proba[line][ENSEMBLE_PLUS]):
            pred = p.argmax(1)
            for c in ["All five"] + COUNTRIES:
                m = np.ones(len(y), bool) if c == "All five" else (frame.country == c).to_numpy()
                cm = pd.crosstab(pd.Categorical(y[m], categories=range(4)), pd.Categorical(pred[m], categories=range(4)),
                                 dropna=False)
                for i in range(4):
                    for j in range(4):
                        rows.append({"line": line, "scope": c, "repeat": rep, "true": CLASS_NAMES[i],
                                     "predicted": CLASS_NAMES[j], "households": int(cm.iloc[i, j])})
    return pd.DataFrame(rows)


def plot_confusion(cm: pd.DataFrame, labs: dict, proba: dict) -> None:
    order = [CLASS_NAMES[k] for k in range(4)]
    short = ["No", "Income", "Expend.", "Double"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15), gridspec_kw={"wspace": 0.12})
    for ax, (line, name) in zip(axes, LINE_NAMES.items()):
        t = (cm[(cm.line == line) & (cm.scope == "All five")].pivot_table(index="true", columns="predicted",
                                                                            values="households", aggfunc="sum")
             .reindex(index=order, columns=order))
        share = t.div(t.sum(axis=1), axis=0).to_numpy()
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{share[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 else INK, fontweight="bold" if i == j else "normal")
        frame, _ = labs[line]
        ba = np.mean([balanced_accuracy_score(frame.y, p.argmax(1)) for p in proba[line][ENSEMBLE_PLUS]])
        ax.set_title(f"{name}\nbalanced accuracy {ba:.3f}", fontsize=7.5, color=INK, loc="left")
        ax.set_xticks(range(4), short, rotation=35, ha="right", rotation_mode="anchor"); ax.set_yticks(range(4), short if line == "p80" else [""] * 4)
        ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    axes[0].set_ylabel("True group")
    fig.text(0.5, -0.02, "Predicted group (Ensemble+, out of fold); cells: share of the true group", ha="center",
             color=INK, fontsize=7.5)
    save_figure(fig, FIGS / "four_groups_confusion"); plt.close(fig)


def main() -> None:
    dev = load_development_frame()
    labs = study_labels(dev)
    with open(ROOT / "artifacts/identification/main_results.pkl", "rb") as fh:
        res = pickle.load(fh)
    proba = {line: assemble(labs[line][0], [r for l, r in res if l == line])["proba"] for line in LINE_NAMES}
    for line in LINE_NAMES:
        frame, _ = labs[line]
        ba = np.mean([balanced_accuracy_score(frame.y, p.argmax(1)) for p in proba[line][ENSEMBLE_PLUS]])
        print(f"{line:14s} Ensemble+ balanced accuracy {ba:.3f}", flush=True)
    plot_space(labs)
    plot_prediction(labs, proba)
    cm = confusion(labs, proba)
    cm.to_csv(TABLES / "table_four_groups_confusion.csv", index=False)
    plot_confusion(cm, labs, proba)


if __name__ == "__main__":
    main()
