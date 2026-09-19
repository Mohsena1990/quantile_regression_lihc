"""The flow, one scatter plot per step (Van Hove-style quadrant scatters, per country).

  step 1  cost threshold      observed cost vs the need-based threshold (regression), with Van Hove's country Q80
  step 2  income x cost       income bracket vs cost relative to its threshold, four coloured quadrants:
                              (a) Van Hove design, raw cost vs Q80   (b) this project, cost vs need-based threshold
  step 2b income sensitivity  static (bracket <= 3) vs dynamic (household-size-adjusted) income: who flips
  step 4  prediction space    predicted income-axis vs cost-axis probability, coloured by true class,
                              income excluded and included

Only countries passing the four-class support gate are drawn. Development partitions only.
Outputs: outputs/figures/step1_cost_threshold, step2_income_cost_quadrants, step2b_income_dynamic_vs_static,
step4_prediction_space (PNG + PDF).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame, tier_features
from lib.identification.labels import CLASS_NAMES, add_labels, support_table, viability
from lib.identification.protocol import oof_proba
from lib.labels.income_harmonization import equivalence_scale, income_proxy_frame
from lib.reporting.figures import save_figure

FIG = ROOT / "outputs/figures"
# Van Hove Fig. 1 / Fig. 4 class colours, so the figures read like the paper this project extends
CLASS_COLOR = {0: "#2ca02c", 1: "#e6d800", 2: "#ff7f0e", 3: "#d62728"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FLIP = {"agree low": "#d62728", "agree not low": "#c3c2b7", "static low only": "#2a78d6", "dynamic low only": "#eb6834"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def class_legend(fig, ncol=4):
    h = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLOR[k], ms=7, label=CLASS_NAMES[k]) for k in range(4)]
    fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, fontsize=8.5, labelcolor=INK2)


def jitter(a, scale, seed=0):
    return np.asarray(a, float) + np.random.default_rng(seed).normal(0, scale, len(a))


def step1(need, raw, countries):
    fig, axes = plt.subplots(1, len(countries), figsize=(3.9 * len(countries), 4.3)); fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(np.atleast_1d(axes), countries):
        style(ax)
        g = need[need.country == c]; q80 = raw[raw.country == c].threshold.iloc[0]
        pos = g.cost > 0
        for hc, col, lab in ((0, "#9ec5f4", "at or below need-based threshold"), (1, "#eb6834", "above it (high cost)")):
            s = g[pos & (g.hc == hc)]
            ax.scatter(s.threshold, s.cost, s=11, color=col, alpha=0.6, linewidths=0, label=lab)
        xlim = [g.threshold.min() * 0.9, g.threshold.max() * 1.1]
        ylim = [max(g[pos].cost.quantile(0.005) * 0.9, 20), g.cost.quantile(0.995) * 1.15]
        ax.plot([1, 1e5], [1, 1e5], color=INK2, lw=1.1, label="need-based threshold (y = x)")
        ax.axhline(q80, color=MUTED, lw=1.1, ls="--", label="Van Hove: country Q80 (per panel)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(c, fontsize=10, color=INK, loc="left"); ax.set_xlabel("Need-based cost threshold (EUR/yr, out-of-fold regression)", fontsize=8, color=INK2)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("Observed annual energy cost (EUR)", fontsize=8, color=INK2)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Step 1 -- cost threshold: households above the diagonal spend more than comparable homes need; the dashed line is Van Hove's single country-wide Q80",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9)); save_figure(fig, FIG / "step1_cost_threshold"); plt.close(fig)


def step2(need, raw, countries):
    fig, axes = plt.subplots(2, len(countries), figsize=(3.9 * len(countries), 7.8), squeeze=False); fig.patch.set_facecolor(SURFACE)
    for j, c in enumerate(countries):
        for i, (lab, ylabel) in enumerate(((raw, "cost / country Q80"), (need, "cost / need-based threshold"))):
            ax = axes[i, j]; style(ax)
            g = lab[(lab.country == c) & (lab.cost > 0)]
            ratio = g.cost / g.threshold
            ax.scatter(jitter(g.income_bracket, 0.13, j), ratio, s=10, c=g.y.map(CLASS_COLOR), alpha=0.7, linewidths=0)
            ax.axvline(3.5, color=INK2, lw=1.1); ax.axhline(1, color=INK2, lw=1.1)
            ax.set_yscale("log"); ax.set_xticks(range(1, 11))
            if i == 0:
                ax.set_title(c, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("Income bracket (line: <= 3 = low income)", fontsize=8, color=INK2)
            ax.set_ylabel(("(a) Van Hove design\n" if i == 0 else "(b) this project\n") + ylabel + " (log)", fontsize=8, color=INK2) if j == 0 else None
    class_legend(fig)
    fig.suptitle("Step 2 -- the four classes as quadrants (Van Hove Fig. 4 style): income bracket against cost relative to its threshold",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); save_figure(fig, FIG / "step2_income_cost_quadrants"); plt.close(fig)


def step2b(dev, countries):
    scale = equivalence_scale(dev.working_age_count + dev.elderly_count, dev.children_count)
    prox = income_proxy_frame(dev.income_bracket, dev.country, scale, cost=dev.cost)
    d = dev.join(prox).dropna(subset=["proxy_equiv_eur", "income_bracket"])
    d["static"] = d.income_bracket <= 3; d["dyn"] = d.low_income_equiv60.astype(bool)
    d["cat"] = np.select([d.static & d.dyn, ~d.static & ~d.dyn, d.static & ~d.dyn], ["agree low", "agree not low", "static low only"], "dynamic low only")
    fig, axes = plt.subplots(1, len(countries), figsize=(3.9 * len(countries), 4.4)); fig.patch.set_facecolor(SURFACE)
    from lib.labels.income_harmonization import _country_decile_table
    dec = _country_decile_table()
    for ax, c in zip(np.atleast_1d(axes), countries):
        style(ax); g = d[d.country == c]
        for k, col in FLIP.items():
            s = g[g.cat == k]
            ax.scatter(jitter(s.income_bracket, 0.13, 3), s.proxy_equiv_eur, s=10, color=col, alpha=0.65, linewidths=0, label=f"{k} (n={len(s)})")
        ax.axhline(0.6 * dec[c][4], color=INK2, lw=1.1, ls="--"); ax.axvline(3.5, color=INK2, lw=1.1)
        ax.set_yscale("log"); ax.set_xticks(range(1, 11)); ax.set_title(f"{c}: agreement {(g.static == g.dyn).mean() * 100:.0f}%", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Income bracket", fontsize=8, color=INK2)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("Equivalised income proxy (EUR/yr, log)", fontsize=8, color=INK2)
    h, l = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Step 2b -- static vs dynamic income rule: vertical line = bracket 3.5 (headline), dashed = 60% of the Eurostat national median after household-size adjustment",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9)); save_figure(fig, FIG / "step2b_income_dynamic_vs_static"); plt.close(fig)


def step4(need, countries):
    fig, axes = plt.subplots(2, len(countries), figsize=(3.9 * len(countries), 7.8), squeeze=False); fig.patch.set_facecolor(SURFACE)
    for i, inc in enumerate((False, True)):
        feats = tier_features(include_income=inc)["T2"]
        for j, c in enumerate(countries):
            ax = axes[i, j]; style(ax)
            g = need[need.country == c].reset_index(drop=True)
            p = oof_proba(g, feats, "lr4").mean(axis=0)
            ax.scatter(p[:, 1] + p[:, 3], p[:, 2] + p[:, 3], s=10, c=g.y.map(CLASS_COLOR), alpha=0.7, linewidths=0)
            ax.axvline(0.5, color=INK2, lw=1.1); ax.axhline(0.5, color=INK2, lw=1.1); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(c, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("Predicted P(low income)", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel(("Income EXCLUDED\n" if not inc else "Income INCLUDED (Van Hove)\n") + "Predicted P(high cost)", fontsize=8, color=INK2)
    class_legend(fig)
    fig.suptitle("Step 4 -- prediction space (out-of-fold, tier T2): each household by predicted income-axis and cost-axis probability, coloured by its TRUE class",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); save_figure(fig, FIG / "step4_prediction_space"); plt.close(fig)


def main():
    dev = load_development_frame()
    need = add_labels(dev, cost_kind="need_adjusted"); raw = add_labels(dev, cost_kind="raw_q80")
    v = viability(support_table(need)); countries = v[v.four_class_ok].country.tolist()
    need, raw, dev = need[need.country.isin(countries)], raw[raw.country.isin(countries)], dev[dev.country.isin(countries)]
    step1(need, raw, countries); step2(need, raw, countries); step2b(dev, countries); step4(need, countries)
    print("wrote step figures for", countries)


if __name__ == "__main__":
    main()
