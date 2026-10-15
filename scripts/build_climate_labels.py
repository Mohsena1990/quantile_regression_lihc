"""Version 6 -- climate-sensitive outcomes (idea G), with the extra countries as training households (idea B).

The four-class label asks about SPENDING, and spending lines remove most of what climate would explain. Climate should matter
more for COMFORT and for under-heating. ENABLE asked the usual indoor temperature (C1A winter, C1B summer: 24 C or above, 22-23,
20-21, 18-19, 17 or below) in Hungary and Spain among the study countries, and in Germany, France and Ukraine. Labels (yes / no):
  cold_home      usual winter indoor temperature 19 C or below (WHO guidance: at least 18 C, about 20-21 C in living rooms)
  hot_home       usual summer indoor temperature 24 C or above
  winter_ep      cold home AND low income (bracket <= 3)       -- winter energy poverty
  summer_ep      hot home AND low income (bracket <= 3)        -- summer energy poverty
  underheat      hidden under-heating: energy bill per unit of modelled heating need below half the median of the household's
                 country x heating-fuel group (lib/identification/required_heat.py; all five study countries). The need uses
                 heating degree days, so a colder region raises the bar: this label asks whether spending keeps pace with climate.
Scored on: Hungary + Spain (comfort labels), the five study countries (underheat). Training adds Germany, Ukraine (and France,
which has no regions, so no climate) for the "pool" arm.

Two tests:
  1 Logistic regression (statsmodels GLM), country fixed effects + the household terms of build_climate_multilevel.py + climate
    (log HDD, CDD/100, heatwave days/10), standard errors clustered by region; placebo = climate shuffled between regions of the
    same country (N_PERM runs); p = share of placebo runs whose log-likelihood gain is at least the real one.
  2 Machine learning (CatBoost, XGBoost; balanced class weights), repeated 5-fold CV x3 on the scored households, feature sets:
      H      Van Hove's seven + country + household indices (+ air conditioning)
      H+R    + region
      H+C    + 14 climate measures (regional and climate x home)
      H+Cp   + the same, shuffled between regions of a country (placebo)
      H+C+R  + both
    arms "five" (train on the scored countries only) and "pool" (+ extra countries). ROC AUC and balanced accuracy, paired
    household bootstrap for differences.

Outputs: outputs/tables/table_v6_climate_labels_sample.csv, table_v6_climate_labels_glm.csv, table_v6_climate_labels_ml.csv,
table_v6_climate_labels_deltas.csv; outputs/figures/v6_climate_labels.
Only CPU models run here; the GPU is hidden (more than four CUDA processes crash under WSL, and the FT-Transformer module opens
a CUDA context on import).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse, sys, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.climate.b2_indices import B2_CLIMATE
from lib.features.indices import climate_features
from lib.identification.data import COUNTRIES_V5, load_pool_frame
from lib.identification.models3 import fit as fit_model
from lib.identification.required_heat import hidden_underheating
from lib.identification.stages import B0_FEATURES, placebo_climate
from lib.reporting.figures import save_figure
from lib.identification.multilevel import HH, prepare as prepare_terms, shuffle_climate

TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
LABELS = ("cold_home", "hot_home", "winter_ep", "summer_ep", "underheat")
SCORED = {"cold_home": ["Hungary", "Spain"], "hot_home": ["Hungary", "Spain"], "winter_ep": ["Hungary", "Spain"],
          "summer_ep": ["Hungary", "Spain"], "underheat": list(COUNTRIES_V5)}
HOUSEHOLD = B0_FEATURES + ["country", "dwelling_efficiency_index", "inefficient_equipment_index", "heating_system_type",
                           "vulnerability_index", "wealth_index", "income_eur_equiv", "has_air_conditioning"]
CLIMATE = ["hdd_hs1617", "cdd_s2017", "hdd_normal", "cdd_normal", "tmin_hs", "heatwave_days", "tropical_nights", "rh_hs",
           "solar_hs", "wind_hs", "heating_need_kwh", "hdd_x_old", "hdd_x_uninsulated", "cdd_x_ac"]
SETS = {"H": HOUSEHOLD, "H+R": HOUSEHOLD + ["region"], "H+C": HOUSEHOLD + CLIMATE, "H+Cp": HOUSEHOLD + CLIMATE,
        "H+C+R": HOUSEHOLD + CLIMATE + ["region"]}
MODELS = ("catboost", "xgboost")
N_PERM, N_REPEATS, N_BOOT = 200, 3, 500
GLM_CLIM = "log_hdd + cdd100 + heatwave10"


def build_labels(pool: pd.DataFrame) -> pd.DataFrame:
    d = pool.copy()
    c1a = pd.to_numeric(d.C1A, errors="coerce").where(lambda s: s.between(1, 5))
    c1b = pd.to_numeric(d.C1B, errors="coerce").where(lambda s: s.between(1, 5))
    low = (d.income_bracket <= 3).astype(float).where(d.income_bracket.notna())
    d["cold_home"] = (c1a >= 4).astype(float).where(c1a.notna())
    d["hot_home"] = (c1b == 1).astype(float).where(c1b.notna())
    d["winter_ep"] = (d.cold_home * low).where(d.cold_home.notna() & low.notna())
    d["summer_ep"] = (d.hot_home * low).where(d.hot_home.notna() & low.notna())
    hu = hidden_underheating(d[d.hdd_hs1617.notna()])
    d["underheat"] = hu.hidden_underheating.reindex(d.index)
    return d


def sample_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in LABELS:
        x = d[d[lab].notna()]
        for c, g in x.groupby("country"):
            rows.append({"label": lab, "country": c, "pool": g.pool.iloc[0], "scored": c in SCORED[lab], "households": len(g),
                         "positives": int(g[lab].sum()), "share": float(g[lab].mean()), "regions": g.climate_region.nunique()})
    return pd.DataFrame(rows)


# ---------------- 1. logistic regression with clustered errors + placebo -------------------------------------------------
def glm_terms(d: pd.DataFrame) -> pd.DataFrame:
    x = prepare_terms(d.assign(_one=1.0), "_one")
    x["heatwave10"] = x.heatwave_days / 10
    for c in ("H3f", "H1f", "H2f", "heat_sys"):                 # rare levels merged (they separate perfectly in small samples)
        n = x[c].map(x[c].value_counts())
        x[c] = x[c].where(n >= 20, "other")
    return x


def _rhs(x: pd.DataFrame, rhs: str) -> str:
    """Drops terms that are constant in this sample (e.g. 'income missing' when the label needs income)."""
    terms = [t.strip() for t in rhs.split("+")]
    keep = [t for t in terms if t.startswith("C(") or t not in x or x[t].nunique() > 1]
    return " + ".join(keep)


def _glm(x: pd.DataFrame, lab: str, rhs: str):
    return smf.glm(f"{lab} ~ {_rhs(x, rhs)}", x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(x.climate_region)[0]})


def _placebo_gain(x, lab, seed):
    p = shuffle_climate(x, ["hdd_hs1617", "cdd_s2017", "heatwave_days"], seed)
    p["heatwave10"] = p.heatwave_days / 10
    try:
        return _glm(p, lab, f"{HH} + {GLM_CLIM}").llf - _glm(p, lab, HH).llf
    except Exception:
        return np.nan


def glm_tests(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in LABELS:
        for sample in ("five", "pool"):
            x = d[d[lab].notna() & d.hdd_hs1617.notna() & d.climate_region.notna()]
            x = x[x.country.isin(SCORED[lab])] if sample == "five" else x
            x = glm_terms(x)
            if x[lab].sum() < 20:
                continue
            m0, m1 = _glm(x, lab, HH), _glm(x, lab, f"{HH} + {GLM_CLIM}")
            gain = m1.llf - m0.llf
            plac = np.array(Parallel(n_jobs=24)(delayed(_placebo_gain)(x, lab, 2000 + i) for i in range(N_PERM)))
            base = {"label": lab, "sample": sample, "households": len(x), "positives": int(x[lab].sum()),
                    "regions": x.climate_region.nunique(), "countries": x.country.nunique(), "loglik_gain_climate": gain,
                    "placebo_gain_mean": float(np.nanmean(plac)), "placebo_gain_p95": float(np.nanpercentile(plac, 95)),
                    "placebo_p": float(np.nanmean(plac >= gain))}
            for t, unit in (("log_hdd", "10% more heating degree days"), ("cdd100", "+100 cooling degree days"),
                            ("heatwave10", "+10 heatwave days")):
                b, se = float(m1.params[t]), float(m1.bse[t])
                scale = np.log(1.1) if t == "log_hdd" else 1.0
                rows.append({**base, "term": t, "unit": unit, "coef": b, "se_cluster": se, "p_value": float(m1.pvalues[t]),
                             "odds_ratio_per_unit": float(np.exp(b * scale)), "or_lo": float(np.exp((b - 1.96 * se) * scale)),
                             "or_hi": float(np.exp((b + 1.96 * se) * scale))})
            print(f"GLM {lab} {sample}: n={len(x)} gain={gain:.2f} placebo p={rows[-1]['placebo_p']:.3f}", flush=True)
    return pd.DataFrame(rows)


# ---------------- 2. machine learning ----------------------------------------------------------------------------------
def ml_job(lab, arm, scored, extra, plac_scored, plac_extra, fold, tr, te, seed):
    out = {}
    for st, feats in SETS.items():
        s_src, e_src = (plac_scored, plac_extra) if st == "H+Cp" else (scored, extra)
        train = s_src.iloc[tr]
        if arm == "pool":
            train = pd.concat([train, e_src], ignore_index=True)
        train = train.assign(y=train[lab].astype(int))
        for clf in MODELS:
            out[(st, clf)] = fit_model(clf, train, feats, seed).predict_proba(s_src.iloc[te])[:, 1]
    return lab, arm, fold, te, out


def _avg(y, P, metric):
    return float(np.mean([metric(y, p) for p in P]))


def _ba(y, p):
    return balanced_accuracy_score(y, (p > 0.5).astype(int))


def ml_tests(d: pd.DataFrame, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks, frames = [], {}
    for lab in LABELS:
        x = d[d[lab].notna()]
        scored = x[x.country.isin(SCORED[lab]) & (x.pool == "focus")].reset_index(drop=True)
        extra = x[~x.country.isin(SCORED[lab]) & (x.pool == "extra")].reset_index(drop=True)
        ps = climate_features(placebo_climate(scored, list(B2_CLIMATE), 77))
        pe = climate_features(placebo_climate(extra, list(B2_CLIMATE), 78)) if len(extra) else extra
        frames[lab] = scored
        strata = scored.country + "_" + scored[lab].astype(int).astype(str)
        splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=5).split(scored, strata)
        for i, (tr, te) in enumerate(splits):
            for arm in ("five", "pool"):
                if arm == "pool" and not len(extra):
                    continue
                tasks.append((lab, arm, scored, extra, ps, pe, i, tr, te, 20261015 + i))
    t0 = time.time()
    res = Parallel(n_jobs=jobs)(delayed(ml_job)(*t) for t in tasks)
    print(f"ML fits done in {time.time() - t0:.0f}s", flush=True)
    proba = {}
    for lab, arm, fold, te, out in res:
        n = len(frames[lab])
        for k, p in out.items():
            proba.setdefault((lab, arm, *k), np.zeros((N_REPEATS, n)))[fold // 5, te] = p
    rows, deltas = [], []
    rng = np.random.default_rng(11)
    for lab, scored in frames.items():
        y = scored[lab].astype(int).to_numpy()
        scopes = {"all scored": np.arange(len(y)), **{c: np.where(scored.country == c)[0] for c in SCORED[lab]}}
        for scope, idx in scopes.items():
            if len(np.unique(y[idx])) < 2:
                continue
            for (l, arm, st, clf), P in proba.items():
                if l != lab:
                    continue
                rows.append({"label": lab, "scope": scope, "arm": arm, "set": st, "model": clf, "households": len(idx),
                             "positives": int(y[idx].sum()), "roc_auc": _avg(y[idx], P[:, idx], roc_auc_score),
                             "balanced_accuracy": _avg(y[idx], P[:, idx], _ba)})
            comps = [("H+C vs H", "five", "H", "five", "H+C"), ("H+C vs H+Cp", "five", "H+Cp", "five", "H+C"),
                     ("H+C vs H+R", "five", "H+R", "five", "H+C"), ("H+C+R vs H+R", "five", "H+R", "five", "H+C+R"),
                     ("pool vs five, H+C", "five", "H+C", "pool", "H+C"), ("pool vs five, H", "five", "H", "pool", "H"),
                     ("pool H+C vs pool H+Cp", "pool", "H+Cp", "pool", "H+C")]
            for name, aa, sa, ab, sb in comps:
                for clf in MODELS:
                    ka, kb = (lab, aa, sa, clf), (lab, ab, sb, clf)
                    if ka not in proba or kb not in proba:
                        continue
                    pa, pb = proba[ka][:, idx], proba[kb][:, idx]
                    obs = _avg(y[idx], pb, roc_auc_score) - _avg(y[idx], pa, roc_auc_score)
                    draws = []
                    for _ in range(N_BOOT):
                        b = rng.integers(0, len(idx), len(idx))
                        if len(np.unique(y[idx][b])) < 2:
                            continue
                        draws.append(_avg(y[idx][b], pb[:, b], roc_auc_score) - _avg(y[idx][b], pa[:, b], roc_auc_score))
                    deltas.append({"label": lab, "scope": scope, "comparison": name, "model": clf, "delta_roc_auc": obs,
                                   "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def figure(ml: pd.DataFrame):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(19, 4.2), sharey=False)
    names = {"cold_home": "cold home (winter <= 19 C)", "hot_home": "hot home (summer >= 24 C)", "winter_ep": "cold home + low income",
             "summer_ep": "hot home + low income", "underheat": "hidden under-heating"}
    for ax, lab in zip(axes, LABELS):
        g = ml[(ml.label == lab) & (ml.scope == "all scored")]
        for j, (arm, clf) in enumerate([(a, c) for a in ("five", "pool") for c in MODELS]):
            h = g[(g.arm == arm) & (g.model == clf)].set_index("set").reindex(list(SETS))
            ax.plot(range(len(SETS)), h.roc_auc, marker="os^D"[j], label=f"{clf}, train {'+ extra countries' if arm == 'pool' else 'scored countries'}")
        ax.set_xticks(range(len(SETS)), list(SETS), fontsize=8); ax.set_title(names[lab], fontsize=9, loc="left")
        ax.grid(color="#e1e0d9", lw=0.6)
    axes[0].set_ylabel("ROC AUC"); axes[0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Climate-sensitive outcomes: household (H), + region (R), + climate (C), + placebo climate (Cp)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, FIG / "v6_climate_labels"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="12 workers crashed (SIGSEGV) next to the GPU run; 4 are safe")
    ap.add_argument("--skip-glm", action="store_true", help="reuse table_v6_climate_labels_glm.csv")
    a = ap.parse_args()
    d = build_labels(load_pool_frame())
    sample_table(d).to_csv(TAB / "table_v6_climate_labels_sample.csv", index=False)
    print(sample_table(d).to_string(), flush=True)
    if a.skip_glm:
        glm = pd.read_csv(TAB / "table_v6_climate_labels_glm.csv")
    else:
        glm = glm_tests(d); glm.to_csv(TAB / "table_v6_climate_labels_glm.csv", index=False)
    ml, dl = ml_tests(d, a.jobs)
    ml.to_csv(TAB / "table_v6_climate_labels_ml.csv", index=False); dl.to_csv(TAB / "table_v6_climate_labels_deltas.csv", index=False)
    figure(ml)
    pd.set_option("display.width", 250)
    print(glm.round(3).to_string())
    print(ml[ml.scope == "all scored"].pivot_table(index=["label", "arm", "model"], columns="set", values="roc_auc").round(3).to_string())
    print(dl[dl.scope == "all scored"].round(3).to_string())


if __name__ == "__main__":
    main()
