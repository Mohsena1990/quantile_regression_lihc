"""Regional context for every household climate region (NUTS3; Poland NUTS2) from Eurostat -- Dalla Longa et al. (2021)
used area-level context (population density, share of rented houses, share of new houses, house value); ENABLE has none of
these, so they come from official regional statistics:

  pop_density          persons per km2, 2017                              demo_r_d3dens   NUTS3 (Poland NUTS2)
  gdp_per_capita       GDP per inhabitant, EUR, 2017                       nama_10r_3gdp   NUTS3 (Poland NUTS2)
  unemployment_rate    % of labour force 15-74, 2017                       lfst_r_lfu3rt   NUTS2
  poverty_rate         at-risk-of-poverty rate %, nearest of 2017/2016/2018/2019   ilc_li41  NUTS2 (else country)
  owner_share          % of private households owning their home, 2011     cens_11htts_r2  NUTS2 (2011 codes)
  pre1981_share        % of dwellings built before 1981, 2021 census       cens_21dwop_r3  NUTS3 (else NUTS2)
A region missing at its own level takes its parent NUTS2 value (NUTS2 2011 codes: HU10 for HU11/HU12, PL12 for PL91/PL92).
No consistent regional house-value statistic exists for the study countries; the household wealth index
(lib/features/indices.py) stands in for house value.

Output: artifacts/reference/regional_context.csv (one row per climate region, with the source code used per variable).
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
LEGACY_NUTS2 = {"HU11": "HU10", "HU12": "HU10", "PL91": "PL12", "PL92": "PL12"}
OUT = ROOT / "artifacts/reference/regional_context.csv"


def jsonstat(payload: dict) -> pd.DataFrame:
    dims = payload["id"]
    cats = [sorted(payload["dimension"][d]["category"]["index"], key=payload["dimension"][d]["category"]["index"].get) for d in dims]
    vals = payload.get("value", {})
    rows = [dict(zip(dims, c), value=float(vals[str(i)])) for i, c in enumerate(itertools.product(*cats)) if str(i) in vals]
    return pd.DataFrame(rows)


def fetch(ds: str, geos: list[str], **filters) -> pd.DataFrame:
    q = "&".join([f"geo={g}" for g in geos] + [f"{k}={v}" for k, v in filters.items()])
    return jsonstat(json.load(urlopen(f"{BASE}/{ds}?lang=EN&{q}", timeout=120)))


def main() -> None:
    regions = pd.read_csv(ROOT / "artifacts/climate/openmeteo/points.csv").climate_region.tolist()
    nuts2 = sorted({r[:4] for r in regions})
    legacy = sorted({LEGACY_NUTS2.get(r, r) for r in nuts2})
    countries = sorted({r[:2] for r in regions})
    out = pd.DataFrame({"climate_region": regions})
    out["nuts2"] = out.climate_region.str[:4]

    def by_region(series: pd.Series, name: str):
        """region value, else its NUTS2 parent (or legacy 2011 code), else the country; records the code used."""
        vals, src = [], []
        for r in out.climate_region:
            for code in (r, r[:4], LEGACY_NUTS2.get(r[:4], r[:4]), r[:2]):
                if code in series.index and pd.notna(series[code]):
                    vals.append(series[code]); src.append(code); break
            else:
                vals.append(float("nan")); src.append(None)
        out[name], out[f"{name}_source"] = vals, src

    d = fetch("demo_r_d3dens", regions + nuts2, time=2017)
    by_region(d.set_index("geo").value, "pop_density")
    g = fetch("nama_10r_3gdp", regions + nuts2, time=2017, unit="EUR_HAB")
    by_region(g.set_index("geo").value, "gdp_per_capita")
    u = fetch("lfst_r_lfu3rt", nuts2 + countries, time=2017, isced11="TOTAL", sex="T", age="Y15-74")
    by_region(u.set_index("geo").value, "unemployment_rate")
    pov = pd.Series(dtype=float)
    for y in (2017, 2016, 2018, 2019):
        p = fetch("ilc_li41", nuts2 + countries, time=y).set_index("geo").value
        pov = pov.combine_first(p)
    by_region(pov, "poverty_rate")
    t = fetch("cens_11htts_r2", legacy + countries, hhcomp="TOTAL")
    t = t.pivot_table(index="geo", columns="tenure", values="value")
    by_region(100 * t["OWN"] / (t["TOTAL"] - t.get("UNK", 0)), "owner_share")
    c = fetch("cens_21dwop_r3", regions + nuts2 + countries, housing="DW")
    c = c.pivot_table(index="geo", columns="y_const", values="value")
    old = c[["Y_LT1919", "Y1919-1945", "Y1946-1960", "Y1961-1980"]].sum(axis=1)
    by_region(100 * old / (c["TOTAL"] - c.get("UNK", 0)), "pre1981_share")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    vars_ = ["pop_density", "gdp_per_capita", "unemployment_rate", "poverty_rate", "owner_share", "pre1981_share"]
    print(out[vars_].describe().round(1).to_string())
    print("missing:", out[vars_].isna().sum().to_dict())
    print("level used:", {v: out[f"{v}_source"].str.len().value_counts().to_dict() for v in vars_})


if __name__ == "__main__":
    main()
