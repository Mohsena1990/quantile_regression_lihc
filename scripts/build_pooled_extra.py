"""More training households from the other ENABLE countries, and a multilevel bill model, on the four-group lines.

Scored households: the five study countries' development households, in exactly the main study's outer folds (repeated 5-fold
x3, stratified by country x group). Two training sets per fold:
  five   the fold's training households of the five study countries
  pool   the same + every labelled household of France, Germany, Norway, Serbia, Ukraine and the United Kingdom
         (lib/identification/pool.py)
Models (fixed feature sets, no selection, so the two training sets differ ONLY in the households):
  B0     core profile + country
  B1     B0 + region + wealth index + vulnerability index + income per adult (EUR) -- the B1 variables chosen most often
  classifiers CatBoost, XGBoost, FT-Transformer; bill model (XGBoost, bill_model.py); multilevel bill model (GPBoost with a
  region random effect, bill_mixed.py); ensemble_plus = CatBoost + XGBoost + FT-Transformer + bill model; ensemble_mixed = the
  same with the multilevel bill model. Households with a bill but no income help both bill models.

Outputs: outputs/tables/table_pool_sample.csv, table_pool_metrics.csv, table_pool_deltas.csv; figure pool;
artifacts/identification/pool_results.pkl (--report-only rebuilds from it; --refit-bill-mixed refits only the GPBoost models).
"""
from __future__ import annotations
import argparse, pickle, sys, time, warnings
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
from lib.identification.bill_model import BillModel, four_class
from lib.identification.data import COUNTRIES, load_pool_frame
from lib.reporting.names import LINES as LABEL_NAMES
from lib.identification.lines import STUDY_LINES, cost_only, fold_threshold, study_labels
from lib.identification.pool import B0, B1, N_REPEATS, extra_labels
from lib.identification.metrics import CLASS_KEYS, summarise_all as summarise
from lib.identification.models import MODELS3, fit as fit_model
from lib.identification.selection import paired_delta
from lib.identification.stages import outer_splits
from lib.reporting.figures import save_figure

TAB, FIG, ART = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "artifacts/identification"
LABELS = STUDY_LINES
N_BOOT = 200


def run_fold(label, focus, extra, conly, spec, fold, tr, te):
    t0 = time.time()
    seed = 20261015 + fold
    test = focus.iloc[te]
    prior = np.bincount(focus.y.iloc[tr], minlength=4) / len(tr)
    thr_te = fold_threshold(focus.iloc[tr], test, spec["family"], spec["tau"])
    out = {}
    for arm in ("five", "pool"):
        train = focus.iloc[tr] if arm == "five" else pd.concat([focus.iloc[tr], extra], ignore_index=True)
        for st, feats in (("B0", B0), ("B1", B1)):
            for clf in MODELS3:
                out[(arm, st, clf)] = fit_model(clf, train, feats, seed).predict_proba(test)
        bill_train = pd.concat([train, conly if arm == "pool" else conly[conly.pool == "focus"]], ignore_index=True)
        out[(arm, "B1", "bill")] = four_class(test.li.to_numpy(), BillModel(bill_train, B1, seed + 1).p_above(test, thr_te), prior)
        out[(arm, "B1", "bill_mixed")] = four_class(test.li.to_numpy(), MixedBillModel(bill_train, B1, seed + 2).p_above(test, thr_te), prior)
    print(f"{label} fold {fold}: {time.time() - t0:.0f}s", flush=True)
    return label, fold, te, out


def refit_mixed(label, focus, extra, conly, spec, fold, tr, te):
    seed = 20261015 + fold
    test = focus.iloc[te]
    prior = np.bincount(focus.y.iloc[tr], minlength=4) / len(tr)
    thr_te = fold_threshold(focus.iloc[tr], test, spec["family"], spec["tau"])
    out = {}
    for arm in ("five", "pool"):
        train = focus.iloc[tr] if arm == "five" else pd.concat([focus.iloc[tr], extra], ignore_index=True)
        bill_train = pd.concat([train, conly if arm == "pool" else conly[conly.pool == "focus"]], ignore_index=True)
        out[(arm, "B1", "bill_mixed")] = four_class(test.li.to_numpy(), MixedBillModel(bill_train, B1, seed + 2).p_above(test, thr_te), prior)
    return label, fold, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--refit-bill-mixed", action="store_true", help="refit only the multilevel bill model in the saved results")
    a = ap.parse_args()
    pool = load_pool_frame()
    dev, ext = pool[pool.pool == "focus"].reset_index(drop=True), pool[pool.pool == "extra"].reset_index(drop=True)
    labs = study_labels(dev)
    extras = {k: extra_labels(ext, spec) for k, (_, spec) in labs.items()}
    conly = cost_only(pool)
    rows = []
    for k, (f, _) in labs.items():
        for name, frame in (("focus", f), ("extra", extras[k])):
            for c, g in frame.groupby("country"):
                rows.append({"label": k, "pool": name, "country": c, "households": len(g),
                             **{f"class_{CLASS_KEYS[j]}": int((g.y == j).sum()) for j in range(4)}})
    sample = pd.DataFrame(rows); sample.to_csv(TAB / "table_pool_sample.csv", index=False)
    print(sample.groupby(["label", "pool"]).households.sum().to_string(), flush=True)
    if a.report_only:
        res = pickle.load(open(ART / "pool_results.pkl", "rb"))
    elif a.refit_bill_mixed:
        old = pickle.load(open(ART / "pool_results.pkl", "rb"))
        splits = {k: outer_splits(f, n_repeats=N_REPEATS) for k, (f, _) in labs.items()}
        new = Parallel(n_jobs=a.jobs)(delayed(refit_mixed)(k, labs[k][0], extras[k], conly, labs[k][1], fold, *splits[k][fold])
                                      for k, fold, _, _ in old)
        upd = {(k, fold): out for k, fold, out in new}
        res = [(k, fold, te, {**out, **upd[(k, fold)]}) for k, fold, te, out in old]
        pickle.dump(res, open(ART / "pool_results.pkl", "wb"))
    else:
        jobs = [(k, f, extras[k], conly, spec, i, tr, te) for k, (f, spec) in labs.items()
                for i, (tr, te) in enumerate(outer_splits(f, n_repeats=N_REPEATS))]
        res = Parallel(n_jobs=a.jobs)(delayed(run_fold)(*j) for j in jobs)
        pickle.dump(res, open(ART / "pool_results.pkl", "wb"))
    report(labs, res)


def report(labs, res):
    metrics, deltas = [], []
    for k, (f, _) in labs.items():
        y = f.y.to_numpy()
        P = {}
        for lab, fold, te, out in res:
            if lab != k:
                continue
            for key, p in out.items():
                P.setdefault(key, np.zeros((N_REPEATS, len(f), 4)))[fold // 5, te] = p
        for arm in ("five", "pool"):
            P[(arm, "B1", "ensemble_plus")] = np.mean([P[(arm, "B1", c)] for c in ("catboost", "xgboost", "ft_transformer", "bill")], axis=0)
            P[(arm, "B1", "ensemble_mixed")] = np.mean([P[(arm, "B1", c)] for c in ("catboost", "xgboost", "ft_transformer", "bill_mixed")], axis=0)
        scopes = {"pooled": np.arange(len(f)), **{c: np.where(f.country == c)[0] for c in COUNTRIES}}
        parts = Parallel(n_jobs=24)(delayed(_scope)(k, scope, idx, y, P) for scope, idx in scopes.items())
        for m, d in parts:
            metrics += m; deltas += d
    m, d = pd.DataFrame(metrics), pd.DataFrame(deltas)
    m.to_csv(TAB / "table_pool_metrics.csv", index=False); d.to_csv(TAB / "table_pool_deltas.csv", index=False)
    figure(m)
    pd.set_option("display.width", 250)
    print(m[m.scope == "pooled"].pivot_table(index=["stage", "model"], columns=["label", "arm"], values="balanced_accuracy").round(3).to_string())
    d["sig"] = np.where(d.delta_balanced_accuracy_lo > 0, "+", np.where(d.delta_balanced_accuracy_hi < 0, "-", ""))
    print(d.groupby(["comparison"]).agg(n=("sig", "size"), better=("sig", lambda s: (s == "+").sum()), worse=("sig", lambda s: (s == "-").sum()),
                                        mean=("delta_balanced_accuracy", "mean")).round(3).to_string())


def _scope(k, scope, idx, y, P):
    rows, deltas = [], []
    for (arm, st, clf), p in P.items():
        rows.append({"label": k, "scope": scope, "n": len(idx), "arm": arm, "stage": st, "model": clf, **summarise(y[idx], p[:, idx], n_boot=N_BOOT)})
    models = [("B0", c) for c in MODELS3] + [("B1", c) for c in MODELS3 + ("bill", "bill_mixed", "ensemble_plus", "ensemble_mixed")]
    for st, clf in models:
        deltas.append({"label": k, "scope": scope, "comparison": "pool vs five", "stage": st, "model": clf,
                       **paired_delta(y[idx], P[("five", st, clf)][:, idx], P[("pool", st, clf)][:, idx])})
    for arm in ("five", "pool"):
        for a_, b_ in (("bill", "bill_mixed"), ("ensemble_plus", "ensemble_mixed")):
            deltas.append({"label": k, "scope": scope, "comparison": f"{b_} vs {a_} ({arm})", "stage": "B1", "model": b_,
                           **paired_delta(y[idx], P[(arm, "B1", a_)][:, idx], P[(arm, "B1", b_)][:, idx])})
    return rows, deltas


def figure(m):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4.6), sharey=True)
    order = [("B0", "catboost"), ("B0", "xgboost"), ("B1", "catboost"), ("B1", "xgboost"), ("B1", "ft_transformer"), ("B1", "bill"),
             ("B1", "bill_mixed"), ("B1", "ensemble_plus"), ("B1", "ensemble_mixed")]
    for ax, lab in zip(axes, LABELS):
        g = m[(m.label == lab) & (m.scope == "pooled")]
        for j, arm in enumerate(("five", "pool")):
            v = [g[(g.arm == arm) & (g.stage == s) & (g.model == c)].balanced_accuracy.iloc[0] for s, c in order]
            ax.plot(range(len(order)), v, marker="os"[j], label="train: five study countries" if arm == "five" else "train: + six other countries")
        ax.set_xticks(range(len(order)), [f"{s} {c}" for s, c in order], rotation=50, fontsize=7.5, ha="right")
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
    axes[0].set_ylabel("balanced accuracy on the five study countries"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Do households from other countries help? Same scored households, same folds, two training sets", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "pool"); plt.close(fig)


if __name__ == "__main__":
    main()
