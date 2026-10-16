"""Shared terms of the multilevel climate tests (scripts/build_climate_multilevel.py, scripts/build_climate_labels.py).

Household terms, climate terms, the mixed model (statsmodels MixedLM, maximum likelihood, random intercept per climate region) and
the within-country climate shuffle used as placebo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

N_PERM = 200
HH = "C(country) + C(H3f) + log_size + C(H1f) + C(H2f) + insulation_count + C(heat_sys) + income_c + income_missing"
CLIM = "log_hdd + cdd100"
ECON = "log_gdp + unemployment_rate + poverty_rate + log_density"
REGION_CLIMATE = ["hdd_hs1617", "cdd_s2017"]
REGION_ECON = ["gdp_per_capita", "unemployment_rate", "poverty_rate", "pop_density"]


def prepare(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = d[d[outcome].gt(0) & d.hdd_hs1617.notna() & d.climate_region.notna()].copy()
    d["y"] = np.log(d[outcome])
    for c in ("H3", "H1", "H2"):
        d[f"{c}f"] = d[c].map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    d["heat_sys"] = d.heating_system_type.fillna("NA").astype(str)
    d["log_size"] = np.log(d.household_size.fillna(d.household_size.median()).clip(lower=1))
    d["insulation_count"] = d.insulation_count.fillna(d.insulation_count.median())
    d["income_missing"] = d.income_bracket.isna().astype(float)
    d["income_c"] = (d.income_bracket - 5.5).fillna(0.0)
    return add_region_terms(d)


def add_region_terms(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_hdd"] = np.log(d.hdd_hs1617) - np.log(d.hdd_hs1617).mean()
    d["cdd100"] = d.cdd_s2017 / 100
    if "gdp_per_capita" in d:
        d["log_gdp"], d["log_density"] = np.log(d.gdp_per_capita), np.log(d.pop_density)
    return d


def fit(d: pd.DataFrame, rhs: str):
    """Maximum likelihood. L-BFGS alone can stop at a degenerate point (region variance 0, log-likelihood inf), so Powell runs
    first; a non-finite fit raises."""
    m = smf.mixedlm(f"y ~ {rhs}", d, groups=d["climate_region"].astype(str)).fit(reml=False, method=["powell", "lbfgs"])
    if not np.isfinite(m.llf):
        raise RuntimeError("mixed model did not converge")
    return m


def s2_region(m) -> float:
    return float(m.cov_re.iloc[0, 0])


def shuffle_climate(d: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """Region-level columns moved to another region of the same country (one permutation of regions per country)."""
    rng = np.random.default_rng(seed)
    reg = d.groupby("climate_region")[cols + ["country"]].first()
    out = []
    for _, g in reg.groupby("country"):
        g = g.copy()
        g[cols] = g[cols].to_numpy()[rng.permutation(len(g))]
        out.append(g)
    reg = pd.concat(out)
    x = d.drop(columns=cols).merge(reg[cols], left_on="climate_region", right_index=True, how="left")
    return add_region_terms(x)


