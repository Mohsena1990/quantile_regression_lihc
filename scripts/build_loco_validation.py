"""An independent test without the locked test partition: leave one country out.

The frozen 20% test partition was evaluated once (2026-09-20) and stays locked, so every later result is cross-validated on the
development households. The strictest check still available is a NEW COUNTRY: for each study country c, the best recipe of the
study is trained on the other four study countries + the six extra ENABLE countries and scored on c's development households,
which it has never seen -- not their country, not their regions. This is also the policy question "can a model built elsewhere
be used in a country that has no survey of its own?".

  models    income only (CatBoost); B0 core profile + country (CatBoost); B1 (B0 + region + wealth + vulnerability +
            income in EUR) with CatBoost, XGBoost, FT-Transformer and the multilevel GPBoost bill model; ensemble_mixed (their
            average) -- the study's best recipe
  the line  the held-out country's own expenditure line (its thresholds are a policy setting computed from its own data, as the
            labels are); the bill model compares its predicted bills with that line
  compare   the same country's score when it IS in the training data (cross-validation, table_pool_metrics.csv, arm "pool")

Outputs: outputs/tables/table_loco.csv; outputs/figures/loco.
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.bill_mixed import MixedBillModel
from lib.identification.bill_model import four_class
from lib.identification.data import COUNTRIES, load_pool_frame
from lib.reporting.names import LINES as LABEL_NAMES
from lib.identification.lines import STUDY_LINES as LABELS, cost_only, study_labels
from lib.identification.models import fit as fit_model
from lib.identification.metrics import summarise_all as summarise
from lib.reporting.figures import save_figure
from lib.identification.pool import B0, B1, extra_labels

TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
SEEDS = (1, 2, 3)


def job(label, focus, extra, conly, country):
    t0 = time.time()
    tr, te = focus[focus.country != country], focus[focus.country == country]
    train = pd.concat([tr, extra], ignore_index=True)
    prior = np.bincount(tr.y, minlength=4) / len(tr)
    out = {}
    for s in SEEDS:                                                     # three seeds stand in for the CV repeats
        p = {"income only": fit_model("catboost", train, ["income_bracket"], s).predict_proba(te),
             "B0 core profile (CatBoost)": fit_model("catboost", train, B0, s).predict_proba(te)}
        for clf, name in (("catboost", "B1 CatBoost"), ("xgboost", "B1 XGBoost"), ("ft_transformer", "B1 FT-Transformer")):
            p[name] = fit_model(clf, train, B1, s).predict_proba(te)
        bill_train = pd.concat([train, conly[conly.country != country]], ignore_index=True)
        p["B1 multilevel bill model"] = four_class(te.li.to_numpy(), MixedBillModel(bill_train, B1, s).p_above(te, te.thr.to_numpy()), prior)
        p["ensemble_mixed"] = np.mean([p[k] for k in ("B1 CatBoost", "B1 XGBoost", "B1 FT-Transformer", "B1 multilevel bill model")], axis=0)
        for k, v in p.items():
            out.setdefault(k, []).append(v)
    print(f"{label} without {country}: {time.time() - t0:.0f}s", flush=True)
    return label, country, te.y.to_numpy(), {k: np.stack(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from table_loco.csv")
    a = ap.parse_args()
    if a.figure_only:
        return figure(pd.read_csv(TAB / "table_loco.csv"))
    pool = load_pool_frame()
    dev, ext = pool[pool.pool == "focus"].reset_index(drop=True), pool[pool.pool == "extra"].reset_index(drop=True)
    labs = study_labels(dev)
    conly = cost_only(pool)
    res = Parallel(n_jobs=a.jobs)(delayed(job)(k, f, extra_labels(ext, spec), conly, c)
                                  for k, (f, spec) in labs.items() for c in COUNTRIES)
    rows = [{"label": k, "country": c, "model": m, "n": len(y), **summarise(y, p, n_boot=200)} for k, c, y, out in res for m, p in out.items()]
    t = pd.DataFrame(rows)
    cv = pd.read_csv(TAB / "table_pool_metrics.csv")
    cv = cv[(cv.arm == "pool") & (cv.stage == "B1") & (cv.model == "ensemble_mixed")][["label", "scope", "balanced_accuracy"]]
    t = t.merge(cv.rename(columns={"scope": "country", "balanced_accuracy": "cv_ensemble_mixed_in_training"}), on=["label", "country"], how="left")
    t.to_csv(TAB / "table_loco.csv", index=False)
    figure(t)
    pd.set_option("display.width", 250)
    print(t.pivot_table(index=["label", "model"], columns="country", values="balanced_accuracy").round(3).to_string())
    e = t[t.model == "ensemble_mixed"]
    print(e[["label", "country", "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi", "cv_ensemble_mixed_in_training",
             "recall_double_risk", "precision_double_risk"]].round(3).to_string())


def figure(t):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4.4), sharey=True)
    models = ["income only", "B0 core profile (CatBoost)", "B1 XGBoost", "B1 multilevel bill model", "ensemble_mixed"]
    for ax, lab in zip(axes, LABELS):
        g = t[t.label == lab]
        x = np.arange(len(COUNTRIES))
        for j, m in enumerate(models):
            v = g[g.model == m].set_index("country").reindex(COUNTRIES).balanced_accuracy
            ax.bar(x + (j - 2) * 0.16, v, width=0.15, label=m)
        cvv = g[g.model == "ensemble_mixed"].set_index("country").reindex(COUNTRIES).cv_ensemble_mixed_in_training
        ax.scatter(x + 0.32, cvv, marker="_", s=300, color="black", zorder=3, label="ensemble_mixed, country in training (CV)")
        ax.set_xticks(x, [c[:3] for c in COUNTRIES]); ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left")
        ax.axhline(0.25, color="#999", lw=0.7, ls=":"); ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    axes[0].set_ylabel("balanced accuracy on the held-out country"); axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Leave one country out: trained on the other four study countries + six extra countries, scored on an unseen country",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "loco"); plt.close(fig)


if __name__ == "__main__":
    main()
