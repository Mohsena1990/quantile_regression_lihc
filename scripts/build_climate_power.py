"""Version 6, step 2 -- power: how strong would climate have to be for our tests to see it?

Part 1 (analytic, the bill): the smallest effect a two-sided 5% test detects with 80% power is about 2.8 standard errors. From the
multilevel model (table_v6_climate_elasticity.csv, table_v6_country_slopes.csv): the minimum detectable elasticity of the bill to
heating degree days, pooled and per country.

Part 2 (simulation, the four groups): the real bills are made MORE climate-dependent by a known amount delta,
    bill' = bill x exp(delta x (log HDD_region - mean log HDD of the country)),
so that within each country a region 10% colder pays about delta x 10% more on top of the real effect (estimated elasticity about
0.5 for the total bill). The lines are recomputed from bill' and the climate test of the four-group study is rerun: CatBoost,
XGBoost and the XGBoost bill model with B1 (Van Hove's seven + country + region + wealth + vulnerability + income in EUR),
B1 + climate and B1 + placebo climate (swapped between regions of a country); repeated 5-fold CV x3 on the five study countries;
paired bootstrap of climate vs placebo. delta = 0 is the real data. Lines: Van Hove's percentile, the energy burden and the need
line without climate (the need line WITH climate adjusts for degree days by construction, so it cannot react).

Finding of part 2 (first run): even an added elasticity of 2 is not detected. Climate has one value per region, so any climate
effect is a region effect, and the region variable -- and the placebo, whose shuffled values still identify regions -- absorb it.
This test can only ask "does the exact climate add anything once the region is known?", never "does climate matter?".
Part 3 (--grouped): the test that CAN see a regional effect -- households are scored in regions the model never saw (folds by
region: GroupKFold x3 with shuffled region-to-fold assignment, stratification not possible), and the region variable is left out.
Region identity cannot help in an unseen region; a real climate effect carries over to new regions, a shuffled one does not.

Outputs: outputs/tables/table_v6_power_bill.csv, table_v6_power_detection.csv (+ table_v6_power_detection_grouped.csv);
outputs/figures/v6_power (+ v6_power_grouped).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, sys, warnings
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
from lib.climate.b2_indices import B2_CLIMATE
from lib.features.indices import climate_features
from lib.identification.bill_model import BillModel, four_class
from lib.identification.data import load_development_frame
from lib.identification.lines_v5 import chosen_thresholds_v5, fold_threshold, make_labels
from lib.identification.models3 import fit as fit_model
from lib.identification.stages import paired_delta, placebo_climate
from lib.identification.stages_v5 import outer_splits
from lib.reporting.figures import save_figure
from build_bill_climate import CLIMATE
from build_pooled_extra import B1

TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
DELTAS = (0.0, 0.25, 0.5, 1.0, 2.0)
MODELS = ("catboost", "xgboost", "bill")
SETS = {"B1": B1, "B1+C": B1 + CLIMATE, "B1+Cp": B1 + CLIMATE}


def analytic() -> pd.DataFrame:
    e = pd.read_csv(TAB / "table_v6_climate_elasticity.csv")
    e = e[e.term == "log_hdd"][["sample", "outcome", "households", "regions", "coef", "se"]].assign(country="all")
    s = pd.read_csv(TAB / "table_v6_country_slopes.csv").drop(columns=["regions"]).rename(
        columns={"coef_log_hdd": "coef", "regions_in_country": "regions"})
    s = s[["sample", "outcome", "country", "regions", "coef", "se"]]
    t = pd.concat([e, s], ignore_index=True)
    t["min_detectable_elasticity"] = 2.8 * t.se
    t["detectable_10pct_colder_pct"] = 100 * (1.1 ** t.min_detectable_elasticity - 1)
    t["observed_is_detectable"] = t.coef.abs() >= t.min_detectable_elasticity
    return t


def labels_for(dev: pd.DataFrame, family: str, cutoff: int, tau: float, delta: float) -> pd.DataFrame:
    d = dev.copy()
    lh = np.log(d.hdd_hs1617)
    d["cost"] = d.cost * np.exp(delta * (lh - lh.groupby(d.country).transform("mean")))
    return make_labels(d, family, cutoff, tau)


def grouped_splits(f: pd.DataFrame, n_repeats: int = 3, n_splits: int = 5):
    """Folds by region: every region's households are held out together; three different region-to-fold assignments."""
    out = []
    for r in range(n_repeats):
        regions = f.climate_region.astype(str).unique()
        rng = np.random.default_rng(100 + r)
        fold_of = dict(zip(regions, rng.permutation(len(regions)) % n_splits))
        g = f.climate_region.astype(str).map(fold_of).to_numpy()
        out += [(np.where(g != k)[0], np.where(g == k)[0]) for k in range(n_splits)]
    return out


def fold_job(key, f, fp, spec, fold, tr, te):
    seed = 31 + fold
    out = {}
    prior = np.bincount(f.y.iloc[tr], minlength=4) / len(tr)
    thr = fold_threshold(f.iloc[tr], f.iloc[te], spec["family"], spec["tau"])
    for st, feats in SETS.items():
        src = fp if st == "B1+Cp" else f
        for m in MODELS:
            if m == "bill":
                p = four_class(f.li.iloc[te].to_numpy(), BillModel(src.iloc[tr], feats, seed).p_above(src.iloc[te], thr), prior)
            else:
                p = fit_model(m, src.iloc[tr], feats, seed).predict_proba(src.iloc[te])
            out[(st, m)] = p
    return key, fold, te, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--grouped", action="store_true", help="part 3: unseen regions, no region variable")
    a = ap.parse_args()
    global SETS
    sfx = "_grouped" if a.grouped else ""
    if a.grouped:
        b1 = [f for f in B1 if f != "region"]
        SETS = {"B1": b1, "B1+C": b1 + CLIMATE, "B1+Cp": b1 + CLIMATE}
    an = analytic(); an.to_csv(TAB / "table_v6_power_bill.csv", index=False)
    dev = load_development_frame(extended=True)
    dev = dev[dev.hdd_hs1617.notna()].reset_index(drop=True)
    ch = chosen_thresholds_v5()
    specs = {"vh": {"family": "pct", "cutoff": 3, "tau": 0.80}, "burden": {"family": "burden", **{k: ch["burden"][k] for k in ("cutoff", "tau")}},
             "needqr_noclim": {"family": "needqr_noclim", **{k: ch["needqr"][k] for k in ("cutoff", "tau")}}}
    frames, tasks = {}, []
    for lab, spec in specs.items():
        for delta in DELTAS:
            f = labels_for(dev, spec["family"], spec["cutoff"], spec["tau"], delta)
            keep = f[["y", "li", "hc", "thr", "cost"]]
            fp = climate_features(placebo_climate(f, list(B2_CLIMATE), 5)); fp[keep.columns] = keep
            frames[(lab, delta)] = f
            splits = grouped_splits(f) if a.grouped else outer_splits(f, n_repeats=3)
            tasks += [((lab, delta), f, fp, spec, i, tr, te) for i, (tr, te) in enumerate(splits)]
    res = Parallel(n_jobs=a.jobs)(delayed(fold_job)(*t) for t in tasks)
    P = {}
    for key, fold, te, out in res:
        for k, p in out.items():
            P.setdefault((*key, *k), np.zeros((3, len(frames[key]), 4)))[fold // 5, te] = p
    rows = []
    for (lab, delta), f in frames.items():
        y = f.y.to_numpy()
        for m in MODELS:
            for name, sa, sb in (("climate vs placebo", "B1+Cp", "B1+C"), ("climate vs none", "B1", "B1+C")):
                d = paired_delta(y, P[(lab, delta, sa, m)], P[(lab, delta, sb, m)], metrics=("balanced_accuracy",))
                rows.append({"label": lab, "added_elasticity": delta, "total_elasticity_approx": 0.5 + delta, "model": m,
                             "comparison": name, **d, "detected": d["delta_balanced_accuracy_lo"] > 0})
    t = pd.DataFrame(rows); t.to_csv(TAB / f"table_v6_power_detection{sfx}.csv", index=False)
    figure(t, sfx)
    pd.set_option("display.width", 250)
    print(an.round(3).to_string())
    x = t[t.comparison == "climate vs placebo"]
    print(x.pivot_table(index=["label", "model"], columns="added_elasticity", values="delta_balanced_accuracy").round(3).to_string())
    print(x.pivot_table(index=["label", "model"], columns="added_elasticity", values="detected").to_string())


def figure(t, sfx=""):
    labs = list(dict.fromkeys(t.label))
    fig, axes = plt.subplots(1, len(labs), figsize=(15, 4.2), sharey=True)
    names = {"vh": "Van Hove (country percentile)", "burden": "Energy burden", "needqr_noclim": "Need line without climate"}
    for ax, lab in zip(axes, labs):
        g = t[(t.label == lab) & (t.comparison == "climate vs placebo")]
        for m, h in g.groupby("model"):
            ax.errorbar(h.added_elasticity + {"catboost": -0.03, "xgboost": 0, "bill": 0.03}[m], h.delta_balanced_accuracy,
                        yerr=[h.delta_balanced_accuracy - h.delta_balanced_accuracy_lo, h.delta_balanced_accuracy_hi - h.delta_balanced_accuracy],
                        marker="o", capsize=2, label={"bill": "bill model"}.get(m, m))
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(names[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_xlabel("climate effect ADDED to the bills (elasticity; real data = 0, real effect about 0.5)", fontsize=7.5)
    axes[0].set_ylabel("gain of climate over placebo\n(balanced accuracy, 95% CI)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("How strong must climate be before the four-group test sees it?" + (" Households scored in UNSEEN regions, no region variable"
                 if sfx else " (region known)"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / f"v6_power{sfx}"); plt.close(fig)


if __name__ == "__main__":
    main()
