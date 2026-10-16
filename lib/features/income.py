"""Household income in EUR from the income bracket (an ordinal proxy, used for the energy-burden line and the income indices).

Chain: bracket k (1-10; respondents placed themselves in their own country's income deciles) -> national decile midpoint
(k - 0.5) / 10 -> the country's Eurostat equivalised income at that point (ilc_di01 2018 deciles D1-D9, linear interpolation, flat
beyond D1 / D9; artifacts/reference/income_quantiles_2018.csv from scripts/prepare_country_references.py) -> household total =
that value x the country's mean equivalence scale -> per adult equivalent with the household's own OECD-modified scale.
Countries without Eurostat deciles (Ukraine) get no EUR income.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QUANTILES_PATH = ROOT / "artifacts/reference/income_quantiles_2018.csv"
_DECILE_FRACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def country_deciles(path: Path | str = QUANTILES_PATH) -> dict[str, np.ndarray]:
    q = pd.read_csv(path)
    q["rank"] = q["quantile"].str.removeprefix("D").astype(int)
    q = q.sort_values("rank")
    out = {}
    for country, g in q.groupby("country"):
        if not np.array_equal(g["rank"].to_numpy(), np.arange(1, 10)):
            raise ValueError(f"{country}: expected deciles D1..D9, got {sorted(g['rank'])}")
        out[country] = g["cutoff_eur"].to_numpy(dtype=float)
    return out


def income_at_percentile(percentile: pd.Series, country: pd.Series, path: Path | str = QUANTILES_PATH) -> pd.Series:
    out = pd.Series(np.nan, index=percentile.index, dtype=float)
    for c, cutoffs in country_deciles(path).items():
        mask = (country == c).to_numpy()
        if mask.any():
            out.loc[mask] = np.interp(percentile.loc[mask], _DECILE_FRACTIONS, cutoffs)
    return out


def equivalence_scale(adults: pd.Series, children: pd.Series) -> pd.Series:
    """OECD-modified scale: 1 for the first adult, 0.5 per further adult, 0.3 per child (adults floored at 1)."""
    a = pd.to_numeric(adults, errors="coerce").clip(lower=1)
    c = pd.to_numeric(children, errors="coerce")
    return 1.0 + 0.5 * (a - 1.0) + 0.3 * c


def income_eur(income_bracket: pd.Series, country: pd.Series, scale: pd.Series) -> pd.DataFrame:
    """income_eur_total (household) and income_eur_equiv (per adult equivalent)."""
    b = pd.to_numeric(income_bracket, errors="coerce")
    pct = ((b - 0.5) / 10.0).where(b.between(1, 10))
    eur = income_at_percentile(pct.dropna(), country.loc[pct.notna()]).reindex(pct.index)
    scale = pd.to_numeric(scale, errors="coerce")
    ok = eur.notna() & scale.notna()
    total = eur * scale.where(ok).groupby(country).transform("mean")
    return pd.DataFrame({"income_eur_total": total, "income_eur_equiv": total / scale}, index=pct.index)
