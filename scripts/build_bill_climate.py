"""Version 6, step 1 -- does climate help DETECTION when it enters where it acts: the bill model?

Version 6 showed that climate drives the size of the bill (a 10% colder winter: heating bill +12%) but not the four-group
classifiers. The bill models of versions 5-6 used only household variables. Here the same bill models get the climate measures as
well, and are tested against a placebo with the climate swapped between regions of the same country:

  bill models     XGBoost bill model (lib/identification/bill_model.py) and multilevel GPBoost bill model (bill_mixed.py)
  feature sets    B1 (as build_pooled_extra.py: Van Hove's seven + country + region + wealth + vulnerability + income in EUR)
                  B1+C   + 9 climate measures (degree days of 2016/17 and the 1991-2020 normals, heating need, heating need per
                         person, cold winter x no insulation, cold winter x old home, hot summer x air conditioning)
                  B1+Cp  the same with climate swapped between regions within each country (household climate x home measures
                         recomputed); the LINE (threshold) always uses the real data
  training        the "pool" arm of build_pooled_extra.py (five study countries + six extra countries), same folds
  ensemble        CatBoost + XGBoost + FT-Transformer (from artifacts/identification/v6_pool_results.pkl) + the bill model
Outputs: outputs/tables/table_v6_bill_climate_metrics.csv, table_v6_bill_climate_deltas.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.climate.b2_indices import B2_CLIMATE
from lib.features.indices import climate_features
from lib.identification.bill_mixed import MixedBillModel
from lib.identification.bill_model import BillModel, four_class
from lib.identification.data import COUNTRIES_V5 as COUNTRIES, load_pool_frame
from lib.identification.lines_v5 import cost_only, fold_threshold, study_labels_v5
from lib.identification.protocol import summarise
from lib.identification.stages import paired_delta, placebo_climate
from lib.identification.stages_v5 import outer_splits
from build_pooled_extra import B1, LABELS, N_REPEATS, extra_labels

TAB, ART = ROOT / "outputs/tables", ROOT / "artifacts/identification"
CLIMATE = ["hdd_hs1617", "cdd_s2017", "hdd_normal", "cdd_normal", "heating_need_kwh", "heating_need_pp", "hdd_x_uninsulated",
           "hdd_x_old", "cdd_x_ac"]
SETS = {"B1": B1, "B1+C": B1 + CLIMATE, "B1+Cp": B1 + CLIMATE}


def placebo(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    keep = frame[["y", "li", "hc", "thr"]].copy() if "y" in frame else None
    out = climate_features(placebo_climate(frame, list(B2_CLIMATE), seed))
    if keep is not None:
        out[keep.columns] = keep
    return out


def run_fold(label, focus, extra, conly, spec, fold, tr, te):
    t0 = time.time()
    seed, rep = 20261015 + fold, fold // 5
    fp, ep, cp = placebo(focus, 900 + rep), placebo(extra, 950 + rep), placebo(conly, 990 + rep)
    prior = np.bincount(focus.y.iloc[tr], minlength=4) / len(tr)
    thr = fold_threshold(focus.iloc[tr], focus.iloc[te], spec["family"], spec["tau"])     # the line: real data only
    li = focus.li.iloc[te].to_numpy()
    out = {}
    for st, feats in SETS.items():
        f_, e_, c_ = (fp, ep, cp) if st == "B1+Cp" else (focus, extra, conly)
        train = pd.concat([f_.iloc[tr], e_, c_], ignore_index=True)
        test = f_.iloc[te]
        out[(st, "bill")] = four_class(li, BillModel(train, feats, seed + 1).p_above(test, thr), prior)
        out[(st, "bill_mixed")] = four_class(li, MixedBillModel(train, feats, seed + 2).p_above(test, thr), prior)
    print(f"{label} fold {fold}: {time.time() - t0:.0f}s", flush=True)
    return label, fold, te, out


def _scope(k, scope, idx, y, P):
    rows = [{"label": k, "scope": scope, "n": len(idx), "set": st, "model": m, **summarise(y[idx], p[:, idx], n_boot=200)}
            for (st, m), p in P.items()]
    deltas = []
    for m in ("bill", "bill_mixed", "ensemble"):
        for name, a, b in (("climate vs none", "B1", "B1+C"), ("climate vs placebo", "B1+Cp", "B1+C")):
            deltas.append({"label": k, "scope": scope, "model": m, "comparison": name,
                           **paired_delta(y[idx], P[(a, m)][:, idx], P[(b, m)][:, idx])})
    return rows, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    pool = load_pool_frame()
    dev, ext = pool[pool.pool == "focus"].reset_index(drop=True), pool[pool.pool == "extra"].reset_index(drop=True)
    labs = {k: v for k, v in study_labels_v5(dev).items() if k in LABELS}
    conly = cost_only(pool)
    jobs = [(k, f, extra_labels(ext, spec), conly, spec, i, tr, te) for k, (f, spec) in labs.items()
            for i, (tr, te) in enumerate(outer_splits(f, n_repeats=N_REPEATS))]
    res = Parallel(n_jobs=a.jobs)(delayed(run_fold)(*j) for j in jobs)
    pickle.dump(res, open(ART / "v6_bill_climate_results.pkl", "wb"))
    clf = pickle.load(open(ART / "v6_pool_results.pkl", "rb"))
    rows, deltas = [], []
    for k, (f, _) in labs.items():
        y = f.y.to_numpy()
        P = {}
        for src in (res, clf):
            for lab, fold, te, out in src:
                if lab != k:
                    continue
                for key, p in out.items():
                    if src is clf:
                        if key[0] != "pool" or key[1] != "B1" or key[2] not in ("catboost", "xgboost", "ft_transformer"):
                            continue
                        key = ("clf", key[2])
                    P.setdefault(key, np.zeros((N_REPEATS, len(f), 4)))[fold // 5, te] = p
        base = np.mean([P[("clf", c)] for c in ("catboost", "xgboost", "ft_transformer")], axis=0) * 0.75
        for st in SETS:
            P[(st, "ensemble")] = base + 0.25 * P[(st, "bill_mixed")]
        P = {key: p for key, p in P.items() if key[0] != "clf"}
        scopes = {"pooled": np.arange(len(f)), **{c: np.where(f.country == c)[0] for c in COUNTRIES}}
        for r, d in Parallel(n_jobs=24)(delayed(_scope)(k, s, idx, y, P) for s, idx in scopes.items()):
            rows += r; deltas += d
    m, d = pd.DataFrame(rows), pd.DataFrame(deltas)
    m.to_csv(TAB / "table_v6_bill_climate_metrics.csv", index=False); d.to_csv(TAB / "table_v6_bill_climate_deltas.csv", index=False)
    pd.set_option("display.width", 250)
    print(m[m.scope == "pooled"].pivot_table(index=["model", "set"], columns="label", values="balanced_accuracy").round(3).to_string())
    d["sig"] = np.where(d.delta_balanced_accuracy_lo > 0, "+", np.where(d.delta_balanced_accuracy_hi < 0, "-", ""))
    x = d[d.scope == "pooled"].assign(cell=lambda t: t.delta_balanced_accuracy.round(3).astype(str) + t.sig)
    print(x.pivot_table(index=["comparison", "model"], columns="label", values="cell", aggfunc="first").to_string())
    print(d[d.scope != "pooled"].groupby(["comparison", "model"]).agg(better=("sig", lambda s: (s == "+").sum()),
                                                                         worse=("sig", lambda s: (s == "-").sum()), n=("sig", "size")).to_string())


if __name__ == "__main__":
    main()
