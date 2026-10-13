"""Class-imbalance strategies for the four-class task -- compared on DEVELOPMENT data only, with the rules fixed in advance.

The test partition is already spent (config/final_selection.json) and is not used here. Trying many strategies on the same
development households inflates the best one's score (winner's curse), so the protocol is declared BEFORE any run:

Round 1 strategies (fixed, 7): none | balanced (current default) | sqrt | ros | rus | smote (logistic only) | posthoc
Round 2 strategies (declared after round 1 found nothing, before running it): balanced_posthoc | alpha_tuned | bagging
                       -- see lib/identification/imbalance.py. Round 2 uses a NEW confirmation seed (202); the same decision rule.
Multiplicity: 9 non-baseline strategies in total; the bar is not relaxed to compensate for the number of tries.
Models: logistic (lr4) and CatBoost (cb4). Labels: need-based cost axis, income axis bracket <= 3.

PHASE 1 -- selection.  Tier T2, income excluded and included, four countries (8 cells per model), CV seed 1 (the benchmark's
seed, so `balanced` reproduces the benchmark). For each model the challenger is the non-baseline strategy with the highest
mean macro-F1 over the 8 cells (and only if it beats `balanced`).

PHASE 2 -- confirmation on FRESH folds (CV seed 101, never used to choose anything).  Tiers T1-T3 x income excluded/included x
four countries x two cost definitions (need_adjusted, raw_q80) = 48 cells per model.  Each challenger is compared with
`balanced` on identical folds; a paired household bootstrap gives the difference and its 95% CI per cell.

DECISION RULE (a strategy counts as a genuine improvement only if ALL hold, per model):
  (a) mean macro-F1 gain over balanced >= +0.02 across the 48 cells,
  (b) it improves macro-F1 in at least 75% of the cells,
  (c) mean MCC is not lower than balanced.
Log-loss, accuracy and Double-Risk recall are reported alongside; resampling and post-hoc scaling distort probabilities, so a
gain in F1 may come with worse log-loss. Nothing here changes config/final_selection.json or the pre-registered test result.

Outputs: outputs/tables/table_imbalance_phase1[_round2].csv, table_imbalance_phase2_paired[_round2].csv, table_imbalance_verdict[_round2].csv,
         outputs/figures/imbalance_strategy_tradeoff (written by the round-2 run, which needs the round-1 table)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame, tier_features
from lib.identification.labels import add_labels
from lib.identification.metrics import average_over_repeats
from lib.identification.protocol import oof_proba, summarise

OUT = ROOT / "outputs/tables"
MODELS = ("lr4", "cb4")
BASELINE = "balanced"
SEED_SELECT = 1
ROUNDS = {1: {"strategies": ("none", "balanced", "sqrt", "ros", "rus", "smote", "posthoc"), "seed_confirm": 101, "suffix": ""},
          2: {"strategies": ("balanced", "balanced_posthoc", "alpha_tuned", "bagging"), "seed_confirm": 202, "suffix": "_round2"}}
N_JOBS = 12
PAIRED_KEYS = ("f1_macro", "mcc", "accuracy", "balanced_accuracy", "recall_double_risk", "auc_macro_ovr", "log_loss")
MIN_GAIN, MIN_SHARE = 0.02, 0.75


def _feats(tier: str, state: str) -> list[str]:
    return tier_features(include_income=(state == "included"))[tier]


def phase1_cell(g, country, state, model, strategy):
    t0 = time.time()
    p = oof_proba(g, _feats("T2", state), model, seed=SEED_SELECT, imbalance=strategy)
    return {"country": country, "income_state": state, "model": model, "strategy": strategy, **summarise(g.y.values, p, n_boot=100),
            "seconds": time.time() - t0}


def run_phase1(lab: pd.DataFrame, strategies) -> pd.DataFrame:
    jobs = [(lab[lab.country == c].reset_index(drop=True), c, st, m, s) for c in COUNTRIES for st in ("excluded", "included")
            for m in MODELS for s in strategies if not (m == "cb4" and s == "smote")]
    rows = Parallel(n_jobs=N_JOBS)(delayed(phase1_cell)(*j) for j in jobs)
    return pd.DataFrame(rows)


def paired_boot(y, base, chal, n_boot=300, seed=11):
    rng = np.random.default_rng(seed)
    b0, c0 = average_over_repeats(y, base), average_over_repeats(y, chal)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        b, c = average_over_repeats(y[idx], base[:, idx]), average_over_repeats(y[idx], chal[:, idx])
        draws.append([c[k] - b[k] for k in PAIRED_KEYS])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    out = {}
    for i, k in enumerate(PAIRED_KEYS):
        out[f"d_{k}"], out[f"d_{k}_lo"], out[f"d_{k}_hi"] = c0[k] - b0[k], float(lo[i]), float(hi[i])
    out.update({f"base_{k}": b0[k] for k in ("f1_macro", "mcc", "accuracy", "log_loss")})
    out.update({f"chal_{k}": c0[k] for k in ("f1_macro", "mcc", "accuracy", "log_loss")})
    return out


def phase2_cell(g, kind, country, state, tier, model, challenger, seed_confirm):
    feats = _feats(tier, state)
    base = oof_proba(g, feats, model, seed=seed_confirm, imbalance=BASELINE)
    chal = oof_proba(g, feats, model, seed=seed_confirm, imbalance=challenger)
    return {"cost_kind": kind, "country": country, "income_state": state, "tier": tier, "model": model, "challenger": challenger,
            **paired_boot(g.y.values, base, chal)}


def fig_tradeoff() -> None:
    """Scatter: mean macro-F1 against mean Double-Risk recall per strategy (phase-1 cells, both rounds). Needs both round tables."""
    import matplotlib.pyplot as plt
    from lib.reporting.figures import save_figure
    a, b = OUT / "table_imbalance_phase1.csv", OUT / "table_imbalance_phase1_round2.csv"
    if not (a.exists() and b.exists()):
        return
    p = pd.concat([pd.read_csv(a), pd.read_csv(b).query("strategy != 'balanced'")])
    t = p.groupby(["model", "strategy"])[["f1_macro", "recall_double_risk", "accuracy"]].mean().reset_index()
    ink, ink2, muted, grid, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    style = {"lr4": ("#2a78d6", "o", "logistic"), "cb4": ("#eb6834", "s", "CatBoost")}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8)); fig.patch.set_facecolor(surface)
    for ax, (xcol, xlab) in zip(axes, (("recall_double_risk", "Double-Risk recall (households reached)"), ("accuracy", "Accuracy"))):
        ax.set_facecolor(surface)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(muted)
        ax.grid(color=grid, linewidth=0.7); ax.set_axisbelow(True); ax.tick_params(colors=muted, labelsize=8)
        for m, (col, mk, name) in style.items():
            d = t[t.model == m]
            ax.scatter(d[xcol], d.f1_macro, s=70, color=col, marker=mk, edgecolor=surface, linewidth=1.2, zorder=3, label=name)
            for _, r in d.iterrows():
                ax.annotate(r.strategy, (r[xcol], r.f1_macro), xytext=(5, 4), textcoords="offset points", fontsize=7, color=ink2)
        ax.set_xlabel(xlab, fontsize=8.5, color=ink2); ax.set_ylabel("Macro-F1 (mean of 8 cells)", fontsize=8.5, color=ink2)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=ink2)
    fig.suptitle("Class-imbalance strategies trade minority recall for accuracy; no strategy raises macro-F1 by the pre-declared +0.02\n"
                 "(need-based labels, tier T2, four countries x income excluded/included, development CV)", fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, ROOT / "outputs/figures/imbalance_strategy_tradeoff"); plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2), default=1)
    cfg = ROUNDS[ap.parse_args().round]
    sfx = cfg["suffix"]
    dev = load_development_frame()
    lab = add_labels(dev, cost_kind="need_adjusted")
    t = time.time()
    p1 = run_phase1(lab, cfg["strategies"])
    p1.to_csv(OUT / f"table_imbalance_phase1{sfx}.csv", index=False)
    print(f"phase 1 done in {time.time() - t:.0f}s", flush=True)
    rank = p1.groupby(["model", "strategy"]).f1_macro.mean().unstack(0)
    pd.set_option("display.width", 200)
    print("phase 1: mean macro-F1 over 8 cells (rows = strategy)\n", rank.round(4).to_string(), flush=True)
    challengers = {}
    for m in MODELS:
        s = rank[m].dropna().drop(BASELINE).sort_values(ascending=False)
        if s.iloc[0] > rank.loc[BASELINE, m]:
            challengers[m] = s.index[0]
    print("challengers carried to phase 2:", challengers, flush=True)
    labs = {k: add_labels(dev, cost_kind=k) for k in ("need_adjusted", "raw_q80")}
    jobs = [(labs[k][labs[k].country == c].reset_index(drop=True), k, c, st, tier, m, ch)
            for m, ch in challengers.items() for k in labs for c in COUNTRIES for st in ("excluded", "included") for tier in ("T1", "T2", "T3")]
    t = time.time()
    p2 = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(phase2_cell)(*j, cfg["seed_confirm"]) for j in jobs))
    p2.to_csv(OUT / f"table_imbalance_phase2_paired{sfx}.csv", index=False)
    print(f"phase 2 done in {time.time() - t:.0f}s ({len(p2)} paired cells)", flush=True)
    verdict = []
    for (m, ch), d in p2.groupby(["model", "challenger"]):
        row = {"model": m, "challenger": ch, "cells": len(d), "mean_d_f1": d.d_f1_macro.mean(), "share_cells_f1_up": (d.d_f1_macro > 0).mean(),
               "share_ci_excludes_0_up": (d.d_f1_macro_lo > 0).mean(), "share_ci_excludes_0_down": (d.d_f1_macro_hi < 0).mean(),
               "mean_d_mcc": d.d_mcc.mean(), "mean_d_accuracy": d.d_accuracy.mean(), "mean_d_double_risk_recall": d.d_recall_double_risk.mean(),
               "mean_d_log_loss": d.d_log_loss.mean()}
        row["rule_a_gain_ge_0.02"] = bool(row["mean_d_f1"] >= MIN_GAIN)
        row["rule_b_share_ge_75pct"] = bool(row["share_cells_f1_up"] >= MIN_SHARE)
        row["rule_c_mcc_not_lower"] = bool(row["mean_d_mcc"] >= 0)
        row["genuine_improvement"] = row["rule_a_gain_ge_0.02"] and row["rule_b_share_ge_75pct"] and row["rule_c_mcc_not_lower"]
        verdict.append(row)
        for st, dd in d.groupby("income_state"):
            print(f"  {m} {ch} | income {st}: mean dF1 {dd.d_f1_macro.mean():+.4f}, dMCC {dd.d_mcc.mean():+.4f}, up in {(dd.d_f1_macro > 0).mean():.0%}", flush=True)
    v = pd.DataFrame(verdict)
    v.to_csv(OUT / f"table_imbalance_verdict{sfx}.csv", index=False)
    print(v.round(4).to_string(index=False))
    fig_tradeoff()


if __name__ == "__main__":
    main()
