"""B0 threshold sensitivity and multi-objective choice of the income cut-off and the expenditure level for every line family.

Grid: income bracket cut-off c in {2, 3, 4} x expenditure level tau in {0.60, 0.65, ..., 0.90}, for the families pct, needqr and
burden (lib/identification/lines_v4.py; the climate-free need line reuses needqr's choice). Objectives, all to MAXIMISE, on the
development households of the four countries:
  f1  identifiability  balanced accuracy of the B0 model (Van Hove's seven predictors + country; CatBoost, one pooled model,
                       5-fold CV stratified by country x class; selection seed 101, different from the evaluation seeds)
  f2  validity         hardship gap on the expenditure axis: among low-income households, the share finding it (very) difficult
                       to live on income (S8 >= 3) in Double Risk minus in Income Risk -- averaged over the four countries
  f3  stability        minimum sample size: the smallest class of the pooled four-country data the model is trained on
                       (Dalla Longa et al. 2021)
Constraint: every class has >= 15 households in every country, so country-specific results still have support. (A per-country
minimum of 30, used for per-country models, is infeasible for almost every option: Poland's sample is 83% low-income, so few
households are high-income and high-spending, and Italy has few households below bracket 3.) Method: the grid is small, so every option is evaluated (exact
multi-objective optimisation), the Pareto front is kept, and one option is chosen by TOPSIS (vector-normalised objectives, equal
weights). f2 is what stops the optimisation from simply choosing lines that are easy to predict because they measure home size.

Outputs: outputs/tables/table_threshold_moo.csv; artifacts/identification/moo_thresholds.json; outputs/figures/threshold_moo.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame
from lib.identification.lines_v4 import MOO_PATH, eligible, expenditure_flag, make_labels
from lib.identification.models3 import fit as fit_model
from lib.identification.stages import B0_FEATURES
from lib.reporting.figures import save_figure

CUTOFFS = (2, 3, 4)
TAUS = tuple(np.round(np.arange(0.60, 0.901, 0.05), 2))
FAMILIES = ("pct", "needqr", "burden")
MIN_COUNTRY_CLASS = 15
SEED = 101


def f1_b0(lab: pd.DataFrame) -> float:
    feats = B0_FEATURES + ["country"]
    strata = lab.country + "_" + lab.y.astype(str)
    pred = np.zeros(len(lab), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(lab, strata):
        pred[te] = fit_model("catboost", lab.iloc[tr], feats, SEED).predict_proba(lab.iloc[te]).argmax(1)
    return float(balanced_accuracy_score(lab.y, pred))


def f2_gap(lab: pd.DataFrame) -> float:
    gaps = []
    for _, g in lab.groupby("country"):
        h = (g.S8 >= 3).where(g.S8.notna())
        dr, ir = h[g.y == 3].mean(), h[g.y == 1].mean()
        gaps.append(dr - ir)
    return float(np.mean(gaps))


def pareto(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    v = df[cols].to_numpy(float)
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        dominated = ((v >= v[i]).all(axis=1) & (v > v[i]).any(axis=1)).any()
        keep[i] = not dominated
    return keep


def topsis(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    v = df[cols].to_numpy(float)
    v = v / np.sqrt((v ** 2).sum(axis=0))
    best, worst = v.max(axis=0), v.min(axis=0)
    d_best, d_worst = np.sqrt(((v - best) ** 2).sum(axis=1)), np.sqrt(((v - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)


def main() -> None:
    dev = load_development_frame()
    rows = []
    for fam in FAMILIES:
        e = eligible(dev, fam)
        for tau in TAUS:
            hc = expenditure_flag(e, fam, tau)
            for c in CUTOFFS:
                lab = make_labels(dev, fam, c, tau, hc=hc)
                counts = lab.groupby(["country", "y"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
                country_min = int(counts.to_numpy().min())
                rec = {"family": fam, "cutoff": c, "tau": float(tau), "n": len(lab), "f3_min_class": int(counts.sum().min()),
                       "min_country_class": country_min, "feasible": country_min >= MIN_COUNTRY_CLASS, "f2_hardship_gap": f2_gap(lab)}
                rec["f1_b0_balanced_accuracy"] = f1_b0(lab) if rec["feasible"] else np.nan
                rows.append(rec)
                print(f"{fam} c={c} tau={tau:.2f}: f1={rec['f1_b0_balanced_accuracy']:.3f} f2={rec['f2_hardship_gap']:.3f} "
                      f"f3={rec['f3_min_class']} country-min={country_min}", flush=True)
    t = pd.DataFrame(rows)
    objs = ["f1_b0_balanced_accuracy", "f2_hardship_gap", "f3_min_class"]
    t["pareto"], t["topsis"], t["chosen"] = False, np.nan, False
    chosen = {}
    for fam, g in t[t.feasible].groupby("family"):
        idx = g.index[pareto(g, objs)]
        t.loc[idx, "pareto"] = True
        front = t.loc[idx]
        t.loc[idx, "topsis"] = topsis(front, objs)
        best = t.loc[idx, "topsis"].idxmax()
        t.loc[best, "chosen"] = True
        chosen[fam] = {k: (int(t.loc[best, k]) if k in ("cutoff", "f3_min_class") else float(t.loc[best, k]))
                       for k in ["cutoff", "tau"] + objs}
    vh = t[(t.family == "pct") & (t.cutoff == 3) & (t.tau == 0.80)].iloc[0]
    chosen["van_hove_benchmark"] = {"cutoff": 3, "tau": 0.80, **{k: float(vh[k]) for k in objs}}
    MOO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MOO_PATH.write_text(json.dumps(chosen, indent=2) + "\n")
    t.to_csv(ROOT / "outputs/tables/table_threshold_moo.csv", index=False)

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(16, 4.6))
    names = {"pct": "Country percentile", "needqr": "Quantile Regression on need", "burden": "Energy burden"}
    for ax, fam in zip(axes, FAMILIES):
        g = t[(t.family == fam) & t.feasible]
        sc = ax.scatter(g.f1_b0_balanced_accuracy, g.f2_hardship_gap, s=g.f3_min_class / 2, c=g.cutoff, cmap="viridis",
                        alpha=0.55, edgecolor="none")
        p = g[g.pareto]
        ax.scatter(p.f1_b0_balanced_accuracy, p.f2_hardship_gap, s=p.f3_min_class / 2, facecolor="none", edgecolor="#d1342f", lw=1.2)
        c = g[g.chosen]
        ax.scatter(c.f1_b0_balanced_accuracy, c.f2_hardship_gap, marker="*", s=260, color="#d1342f", label="chosen (TOPSIS)")
        if fam == "pct":
            ax.scatter(vh.f1_b0_balanced_accuracy, vh.f2_hardship_gap, marker="X", s=120, color="black", label="Van Hove (3, 0.80)")
        for _, r in c.iterrows():
            ax.annotate(f"bracket <= {int(r.cutoff)}, tau {r.tau:.2f}", (r.f1_b0_balanced_accuracy, r.f2_hardship_gap),
                        xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.set_title(names[fam], fontsize=10, loc="left")
        ax.set_xlabel("f1: B0 balanced accuracy", fontsize=8.5); ax.set_ylabel("f2: hardship gap (Double vs Income Risk)", fontsize=8.5)
        ax.grid(color="#e1e0d9", lw=0.6); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.colorbar(sc, ax=axes, label="income cut-off (bracket)", shrink=0.8)
    fig.suptitle("Choosing the thresholds: every option (size = smallest class, f3); red rings = Pareto front", fontsize=10.5, x=0.01, ha="left")
    save_figure(fig, ROOT / "outputs/figures/threshold_moo")
    plt.close(fig)
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
