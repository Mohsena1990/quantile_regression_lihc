"""Managerial implications: what missed and wrongly-included households cost, by information tier and country.

A scheme (energy-bill support, retrofit priority, social tariff) must decide which households to assist. The
target group is the LIHC-type "Double Risk" class (low income AND high cost). Two error types follow:
  * MISSED (exclusion error, false negative): an eligible household the scheme does not reach
  * WRONGLY INCLUDED (inclusion error / leakage, false positive): a household assisted although not Double Risk
The information a scheme can obtain sets how good the targeting is:
  T1        structural: dwelling, household composition, heating fuel, climate (register / census / EPC-type data)
  T2        + socio-economic survey items (education, employment, vehicles, appliances, tenure-type items)
  T3        + welfare screen (S7 aid received, S8 income adequacy)
  T3+Inc    + income bracket (from a tax / benefits register) -- Van Hove's design
Rules evaluated from out-of-fold predictions (development partitions only):
  argmax    assist households whose most probable class is Double Risk
  coverage  assist the top-q share of households by P(Double Risk), q swept 5%..50% (a budget-constrained scheme)
Reported per 1,000 households: assisted-correctly (TP), missed (FN), wrongly included (FP), plus precision, recall,
leakage (= 1 - precision), under-coverage (= 1 - recall), accuracy, and the households screened per correct hit.
No euro amounts are assumed. Total error cost is expressed as an index FP + rho x FN (rho = how many times worse a
missed household is than a wrongly included one), with rho in {1, 2, 5} so a manager can read off the tier that
minimises it for their own priorities.

Outputs: outputs/tables/table_managerial_{targeting,coverage_sweep,information_value,cost_ratio}.csv;
figures managerial_targeting_frontier, managerial_errors_per_1000.
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
from lib.identification.labels import add_labels, support_table, viability
from lib.identification.protocol import oof_proba
from lib.reporting.figures import save_figure

TIERS = [("T1", False), ("T2", False), ("T3", False), ("T3", True)]      # (tier, income included)
TIER_LABEL = {("T1", False): "T1 structural", ("T2", False): "T2 + socio-economic", ("T3", False): "T3 + welfare screen",
              ("T3", True): "T3 + income (register)"}
RHOS = (1, 2, 5)
SWEEP = np.round(np.arange(0.05, 0.501, 0.05), 2)
N_BOOT = 300
MODEL = "lr4"                       # fixed, transparent model (not chosen per cell); cb4 is reported as a robustness column set
DOUBLE = 3
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TIER_STYLE = {("T1", False): ("#9ec5f4", "o"), ("T2", False): ("#5598e7", "s"), ("T3", False): ("#256abf", "^"), ("T3", True): ("#0d366b", "D")}
FN_COLOR, FP_COLOR = "#eb6834", "#2a78d6"


def confusion_rates(is_pos: np.ndarray, pred_pos: np.ndarray) -> dict:
    """`pred_pos` is (R, n) booleans (one row per CV repeat); counts are averaged over repeats."""
    pred_pos = np.atleast_2d(pred_pos)
    n = len(is_pos)
    tp = float(np.mean([(is_pos & p).sum() for p in pred_pos]))
    fp = float(np.mean([(~is_pos & p).sum() for p in pred_pos]))
    fn = float(np.mean([(is_pos & ~p).sum() for p in pred_pos]))
    tn = n - tp - fp - fn
    assisted = tp + fp
    return {"n": n, "prevalence": is_pos.mean(), "assisted_share": assisted / n,
            "tp_per_1000": 1000 * tp / n, "fp_per_1000": 1000 * fp / n, "fn_per_1000": 1000 * fn / n, "tn_per_1000": 1000 * tn / n,
            "precision": tp / assisted if assisted else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
            "leakage": fp / assisted if assisted else np.nan, "undercoverage": fn / (tp + fn) if tp + fn else np.nan,
            "accuracy": (tp + tn) / n, "screened_per_hit": assisted / tp if tp else np.nan}


def with_ci(is_pos, pred_pos, keys=("precision", "recall", "leakage", "undercoverage", "fp_per_1000", "fn_per_1000"), seed=5):
    point = confusion_rates(is_pos, pred_pos)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(is_pos), len(is_pos))
        m = confusion_rates(is_pos[idx], np.atleast_2d(pred_pos)[:, idx])
        for k in keys:
            draws[k].append(m[k])
    for k in keys:
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5]); point[f"{k}_lo"], point[f"{k}_hi"] = float(lo), float(hi)
    return point


def topq_decision(score: np.ndarray, q: float) -> np.ndarray:
    k = int(np.ceil(q * len(score)))
    pos = np.zeros(len(score), bool); pos[np.argsort(-score, kind="stable")[:k]] = True
    return pos[None]


def evaluate(lab, countries, model=MODEL, cost_kind="need_adjusted"):
    tgt_rows, sweep_rows = [], []
    for c in countries:
        g = lab[lab.country == c].reset_index(drop=True)
        is_pos = (g.y.to_numpy() == DOUBLE)
        for tier, inc in TIERS:
            feats = tier_features(include_income=inc)[tier]
            proba = oof_proba(g, feats, model)                       # (R, n, 4)
            base = {"cost_kind": cost_kind, "country": c, "model": model, "tier": TIER_LABEL[(tier, inc)], "income_included": inc}
            tgt_rows.append({**base, "rule": "argmax", **with_ci(is_pos, proba.argmax(2) == DOUBLE)})
            score = proba[:, :, DOUBLE].mean(axis=0)
            for q in SWEEP:
                sweep_rows.append({**base, "coverage": q, **confusion_rates(is_pos, topq_decision(score, q))})
    return pd.DataFrame(tgt_rows), pd.DataFrame(sweep_rows)


def information_value(tgt):
    rows = []
    order = [TIER_LABEL[t] for t in TIERS]
    for c, d in tgt.groupby("country"):
        d = d.set_index("tier").loc[order]
        prev = None
        for tier, r in d.iterrows():
            row = {"country": c, "tier": tier, "fp_per_1000": r.fp_per_1000, "fn_per_1000": r.fn_per_1000}
            for rho in RHOS:
                row[f"cost_index_rho{rho}"] = r.fp_per_1000 + rho * r.fn_per_1000
            if prev is not None:
                row["d_fp_vs_previous_tier"] = r.fp_per_1000 - prev.fp_per_1000
                row["d_fn_vs_previous_tier"] = r.fn_per_1000 - prev.fn_per_1000
                for rho in RHOS:
                    row[f"d_cost_rho{rho}"] = row[f"cost_index_rho{rho}"] - (prev.fp_per_1000 + rho * prev.fn_per_1000)
            rows.append(row); prev = r
    iv = pd.DataFrame(rows)
    best = []
    for c, d in iv.groupby("country"):
        for rho in RHOS:
            b = d.loc[d[f"cost_index_rho{rho}"].idxmin()]
            best.append({"country": c, "rho": rho, "cheapest_tier": b.tier, "cost_index": b[f"cost_index_rho{rho}"]})
    return iv, pd.DataFrame(best)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, linewidth=0.7); ax.set_axisbelow(True)


def fig_frontier(tgt, sweep, countries):
    fig, axes = plt.subplots(1, len(countries), figsize=(3.9 * len(countries), 4.4), sharey=True); fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, countries):
        style(ax)
        d = tgt[tgt.country == c]; prev = d.prevalence.iloc[0]
        qs = np.linspace(0.02, 0.98, 50)
        ax.plot(np.full_like(qs, 1 - prev), 1 - qs, color=MUTED, lw=1, ls=":", label="random targeting")
        for key in (("T1", False), ("T3", True)):
            s = sweep[(sweep.country == c) & (sweep.tier == TIER_LABEL[key])].sort_values("coverage")
            ax.plot(s.leakage, s.undercoverage, color=TIER_STYLE[key][0], lw=1.2, alpha=0.9, zorder=1)
        for key in TIERS:
            r = d[d.tier == TIER_LABEL[key]].iloc[0]; col, mk = TIER_STYLE[key]
            ax.errorbar(r.leakage, r.undercoverage, xerr=[[r.leakage - r.leakage_lo], [r.leakage_hi - r.leakage]],
                        yerr=[[r.undercoverage - r.undercoverage_lo], [r.undercoverage_hi - r.undercoverage]],
                        fmt=mk, color=col, ecolor=col, elinewidth=0.9, ms=8, mec=INK, mew=0.6, zorder=3, label=TIER_LABEL[key])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{c}  (Double Risk = {prev * 100:.0f}% of households)", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Wrongly included: share of assisted households not Double Risk", fontsize=8, color=INK2)
        if ax is axes[0]:
            ax.set_ylabel("Missed: share of Double-Risk households not reached", fontsize=8, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Targeting frontier by information tier: lower-left is better. Points = 'assist predicted Double Risk'; thin lines = coverage sweep (T1 light, T3+income dark)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9)); save_figure(fig, ROOT / "outputs/figures/managerial_targeting_frontier"); plt.close(fig)


def fig_errors(tgt, countries):
    fig, axes = plt.subplots(1, len(countries), figsize=(3.9 * len(countries), 4.4), sharey=True); fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, countries):
        style(ax); ax.grid(axis="x", visible=False)
        d = tgt[tgt.country == c].set_index("tier").loc[[TIER_LABEL[t] for t in TIERS]]
        x = np.arange(len(d))
        ax.bar(x, d.fn_per_1000, color=FN_COLOR, width=0.62, label="Missed (eligible, not reached)")
        ax.bar(x, d.fp_per_1000, bottom=d.fn_per_1000, color=FP_COLOR, width=0.62, label="Wrongly included (assisted, not eligible)", edgecolor=SURFACE, linewidth=1.5)
        for xi, (fn, fp) in enumerate(zip(d.fn_per_1000, d.fp_per_1000)):
            ax.text(xi, fn + fp + 4, f"{fn + fp:.0f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, ["T1", "T2", "T3", "T3+Inc"], fontsize=8, color=INK2)
        ax.set_title(c, fontsize=9.5, color=INK, loc="left")
        if ax is axes[0]:
            ax.set_ylabel("Households per 1,000 screened", fontsize=8, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Targeting errors per 1,000 households, by information tier (rule: assist predicted Double Risk)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92)); save_figure(fig, ROOT / "outputs/figures/managerial_errors_per_1000"); plt.close(fig)


def main():
    tab = ROOT / "outputs/tables"; tab.mkdir(parents=True, exist_ok=True)
    dev = load_development_frame(); lab = add_labels(dev, cost_kind="need_adjusted")
    v = viability(support_table(lab)); countries = v[v.four_class_ok].country.tolist()
    lab = lab[lab.country.isin(countries)]
    tgt, sweep = evaluate(lab, countries)
    tgt.to_csv(tab / "table_managerial_targeting.csv", index=False)
    sweep.to_csv(tab / "table_managerial_coverage_sweep.csv", index=False)
    iv, best = information_value(tgt)
    iv.to_csv(tab / "table_managerial_information_value.csv", index=False); best.to_csv(tab / "table_managerial_cost_ratio.csv", index=False)
    fig_frontier(tgt, sweep, countries); fig_errors(tgt, countries)
    pd.set_option("display.width", 220)
    print(tgt[["country", "tier", "prevalence", "tp_per_1000", "fn_per_1000", "fp_per_1000", "precision", "recall", "leakage", "undercoverage", "accuracy"]].round(3).to_string(index=False))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
