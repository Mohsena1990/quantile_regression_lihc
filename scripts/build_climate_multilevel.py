"""Version 6 -- a fair test of climate on the energy BILL with a multilevel (mixed) model (ideas A, E, F).

Why the bill: a continuous outcome has far more statistical power than the four-class label, and climate should act on what a
household spends before it acts on a class.  Why multilevel: climate has ONE value per region, so it must be tested against the
spread BETWEEN regions, with households nested in regions (partial pooling: small regions borrow strength from the rest).

Model (statsmodels MixedLM, maximum likelihood), households i in regions r:
  log(bill_ir) = country fixed effect + household terms + climate_r + u_r + e_ir,   u_r ~ N(0, s2_region), e ~ N(0, s2)
  household terms: floor-area band, log household size, dwelling type, construction period, number of insulation types, heating
                   system type, income bracket (and a flag when it is missing)
  climate:         log heating degree days of winter 2016/17 (HDD) and cooling degree days of summer 2017 (/100)

E  climate elasticity: the HDD coefficient. A 10% colder winter changes the bill by 1.1^b - 1. Identified WITHIN countries
   (country fixed effects), so price levels and national fuel mix cannot drive it.
F  the region effect split up: s2_region without climate (M0) and with it (M1): share of the regional differences explained by
   climate = 1 - s2_region(M1) / s2_region(M0). For the five study countries also with regional economy (GDP per head,
   unemployment, poverty rate, population density; M2) and both (M3).
   Placebo: the regional climate values are shuffled between regions of the same country (N_PERM times); the observed share and
   coefficient are compared with that placebo distribution (p = share of placebo runs at least as large).
A  the region effects u_r are partially pooled estimates (a region with few households is pulled towards its country mean).

Samples: "five" = the five study countries' development households; "pool" = + every household of Germany, Norway, Serbia and
Ukraine (the extra countries with regions; France and the UK have none) -- more regions and a far wider climate range (idea B).
Outcomes: total energy bill, and the heating bill alone (households that pay for heating).

Outputs: outputs/tables/table_v6_climate_elasticity.csv, table_v6_region_variance.csv, table_v6_country_slopes.csv,
table_v6_region_effects.csv; outputs/figures/v6_climate_bill.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES_V5, load_pool_frame
from lib.identification.multilevel import (CLIM, ECON, HH, N_PERM, REGION_CLIMATE, REGION_ECON, fit, prepare, s2_region,
                                             shuffle_climate)
from lib.reporting.figures import save_figure

TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"


def _placebo(d, seed):
    x = shuffle_climate(d, REGION_CLIMATE, seed)
    try:
        m0s2 = s2_region(fit(x, HH))
        m1 = fit(x, f"{HH} + {CLIM}")
        return 1 - s2_region(m1) / m0s2, float(m1.params["log_hdd"])
    except Exception:
        return np.nan, np.nan


def analyse(d: pd.DataFrame, sample: str, outcome: str) -> tuple[list, list, list, pd.DataFrame]:
    m0, m1 = fit(d, HH), fit(d, f"{HH} + {CLIM}")
    b, se = float(m1.params["log_hdd"]), float(m1.bse["log_hdd"])
    lr = 2 * (m1.llf - m0.llf)
    base = {"sample": sample, "outcome": outcome, "households": len(d), "regions": d.climate_region.nunique(),
            "countries": d.country.nunique()}
    plac = Parallel(n_jobs=24)(delayed(_placebo)(d, 1000 + i) for i in range(N_PERM))
    ps, pb = np.array([p[0] for p in plac]), np.array([p[1] for p in plac])
    share = 1 - s2_region(m1) / s2_region(m0)
    el = [{**base, "term": t, "coef": float(m1.params[t]), "se": float(m1.bse[t]), "ci_lo": float(m1.params[t] - 1.96 * m1.bse[t]),
           "ci_hi": float(m1.params[t] + 1.96 * m1.bse[t]), "p_value": float(m1.pvalues[t])} for t in ("log_hdd", "cdd100")]
    el[0].update({"pct_change_bill_10pct_colder": 100 * (1.1 ** b - 1), "pct_lo": 100 * (1.1 ** (b - 1.96 * se) - 1),
                  "pct_hi": 100 * (1.1 ** (b + 1.96 * se) - 1), "lr_stat_climate": lr, "lr_p_climate": float(stats.chi2.sf(lr, 2)),
                  "placebo_coef_mean": float(np.nanmean(pb)), "placebo_coef_p": float(np.nanmean(np.abs(pb) >= abs(b)))})
    var = {**base, "s2_region_M0": s2_region(m0), "s2_household_M0": float(m0.scale), "icc_region_M0": s2_region(m0) / (s2_region(m0) + m0.scale),
           "s2_region_M1_climate": s2_region(m1), "share_explained_climate": share,
           "placebo_share_mean": float(np.nanmean(ps)), "placebo_share_p95": float(np.nanpercentile(ps, 95)),
           "placebo_share_p": float(np.nanmean(ps >= share)), "placebo_runs": int(np.isfinite(ps).sum())}
    if sample == "five" and d[REGION_ECON].notna().all().all():
        m2, m3 = fit(d, f"{HH} + {ECON}"), fit(d, f"{HH} + {ECON} + {CLIM}")
        var.update({"s2_region_M2_economy": s2_region(m2), "share_explained_economy": 1 - s2_region(m2) / s2_region(m0),
                    "s2_region_M3_both": s2_region(m3), "share_explained_both": 1 - s2_region(m3) / s2_region(m0),
                    "climate_coef_given_economy": float(m3.params["log_hdd"]), "climate_p_given_economy": float(m3.pvalues["log_hdd"])})
    slopes = []
    mc = fit(d, f"{HH} + cdd100 + log_hdd:C(country)")
    for c in sorted(d.country.unique()):
        t = f"log_hdd:C(country)[{c}]"
        if t in mc.params:
            slopes.append({**base, "country": c, "regions_in_country": d[d.country == c].climate_region.nunique(),
                           "coef_log_hdd": float(mc.params[t]), "se": float(mc.bse[t]), "p_value": float(mc.pvalues[t])})
    re = pd.DataFrame({"climate_region": list(m0.random_effects), "region_effect_M0": [float(v.iloc[0]) for v in m0.random_effects.values()]})
    re["region_effect_M1"] = re.climate_region.map({k: float(v.iloc[0]) for k, v in m1.random_effects.items()})
    reg = d.groupby("climate_region").agg(country=("country", "first"), households=("y", "size"), hdd=("hdd_hs1617", "first"))
    re = re.merge(reg, left_on="climate_region", right_index=True).assign(sample=sample, outcome=outcome)
    return el, [var], slopes, re


def main() -> None:
    pool = load_pool_frame()
    heat = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost_all.parquet")[["UID", "heating_cost_eur"]]
    pool = pool.merge(heat, on="UID", how="left")
    pool["bill"] = pool.cost
    samples = {"five": pool[pool.pool == "focus"], "pool": pool[pool.country.isin(COUNTRIES_V5 + ["Germany", "Norway", "Serbia", "Ukraine"])]}
    els, vars_, slopes, res = [], [], [], []
    for outcome in ("bill", "heating_cost_eur"):
        for sample, d in samples.items():
            x = prepare(d, outcome)
            e, v, s, r = analyse(x, sample, outcome)
            els += e; vars_ += v; slopes += s; res.append(r)
            print(f"{outcome} {sample}: n={len(x)} regions={x.climate_region.nunique()} elasticity={e[0]['coef']:.3f} "
                  f"({e[0]['ci_lo']:.3f}, {e[0]['ci_hi']:.3f}) share={v[0]['share_explained_climate']:.2f} placebo p={v[0]['placebo_share_p']:.3f}",
                  flush=True)
    pd.DataFrame(els).to_csv(TAB / "table_v6_climate_elasticity.csv", index=False)
    pd.DataFrame(vars_).to_csv(TAB / "table_v6_region_variance.csv", index=False)
    pd.DataFrame(slopes).to_csv(TAB / "table_v6_country_slopes.csv", index=False)
    res = pd.concat(res); res.to_csv(TAB / "table_v6_region_effects.csv", index=False)
    figure(res, pd.DataFrame(els))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(vars_).round(3).T.to_string())
    print(pd.DataFrame(els).round(3).to_string())
    print(pd.DataFrame(slopes).round(3).to_string())


COLOURS = dict(zip(["Bulgaria", "Hungary", "Italy", "Poland", "Spain", "Germany", "Norway", "Serbia", "Ukraine"],
                   ["#1f77b4", "#2ca02c", "#d62728", "#8c564b", "#7f7f7f", "#ff7f0e", "#9467bd", "#e377c2", "#bcbd22"]))


def figure(res: pd.DataFrame, els: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (outcome, sample) in zip(axes, [(o, s) for o in ("bill", "heating_cost_eur") for s in ("five", "pool")]):
        r = res[(res.outcome == outcome) & (res["sample"] == sample)].copy()
        r["log_hdd_within"] = np.log(r.hdd) - np.log(r.hdd).groupby(r.country).transform("mean")
        for c, g in r.groupby("country"):
            ax.scatter(g.log_hdd_within, g.region_effect_M0, s=8 + g.households / 4, alpha=0.7, label=c, color=COLOURS[c])
        b = els[(els.outcome == outcome) & (els["sample"] == sample) & (els.term == "log_hdd")].iloc[0]
        xs = np.linspace(r.log_hdd_within.min(), r.log_hdd_within.max(), 10)
        ax.plot(xs, b.coef * xs, color="black", lw=1.2)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(f"{'energy bill' if outcome == 'bill' else 'heating bill'}, {'five study countries' if sample == 'five' else '+ DE, NO, RS, UA'}\n"
                     f"elasticity {b.coef:.2f} ({b.ci_lo:.2f} to {b.ci_hi:.2f}); 10% colder: {b.pct_change_bill_10pct_colder:+.1f}%",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("region's log heating degree days minus its country's mean", fontsize=8)
        ax.grid(color="#e1e0d9", lw=0.5)
    axes[0].set_ylabel("region effect on log bill\n(multilevel model without climate)", fontsize=8.5)
    axes[-1].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.suptitle("Do colder regions pay more? Region effects (partially pooled) against the region's winter, within countries; "
                 "line = climate slope of the multilevel model", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, FIG / "v6_climate_bill"); plt.close(fig)


if __name__ == "__main__":
    if "--figure-only" in sys.argv:
        figure(pd.read_csv(TAB / "table_v6_region_effects.csv"), pd.read_csv(TAB / "table_v6_climate_elasticity.csv"))
    else:
        main()
