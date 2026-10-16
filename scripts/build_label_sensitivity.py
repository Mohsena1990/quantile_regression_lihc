"""Two label checks for the paper: (1) who changes group when degree days enter the need line; (2) a country-specific income
cut-off instead of bracket <= 3 everywhere.

(1) Climate in the line. The need line (needqr) and the climate-free need line (needqr_noclim) label the same households with the
    same income cut-off and tau; they differ only in the degree days of the quantile regression. For every country: the share of
    households whose group changes, the transition table, and how the switches into / out of high expenditure line up with the
    region's winter severity relative to its country (log HDD minus the country mean).
(2) Income cut-off. The self-reported brackets are far from uniform national deciles (brackets 1-3 hold 23-83% of the labelled
    households). Robustness check: per country, the bracket cut-off whose cumulative share of labelled households is closest to
    30% ("bottom ~30%"). For each study line, with the study cut-off (3) and the country cut-off: low-income share, smallest
    group in any country, hardship gap (validity, as in build_threshold_moo.py) and B0 core-profile CatBoost balanced accuracy
    (identifiability, same protocol and seed as the threshold optimisation).

Outputs: outputs/tables/table_label_climate_switch.csv, table_label_climate_transitions.csv, table_label_income_cutoff.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib.identification.data import load_development_frame
from lib.identification.lines import CLASS_NAMES, eligible, expenditure_threshold, line_specs, make_labels
from build_threshold_moo import f1_b0, f2_gap

OUT = ROOT / "outputs/tables"
TARGET_LOW_INCOME_SHARE = 0.30


def climate_switch(dev: pd.DataFrame, specs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = specs["needqr"]
    a = make_labels(dev, "needqr", s["cutoff"], s["tau"])
    b = make_labels(dev, "needqr_noclim", s["cutoff"], s["tau"])
    assert (a.UID.to_numpy() == b.UID.to_numpy()).all()
    a["y_noclim"], a["hc_noclim"] = b.y.to_numpy(), b.hc.to_numpy()
    a["rel_log_hdd"] = np.log(a.hdd_hs1617) - a.groupby("country").hdd_hs1617.transform(lambda h: np.log(h).mean())
    rows, trans = [], []
    for scope, g in [("All five", a)] + list(a.groupby("country")):
        into = (g.hc == 1) & (g.hc_noclim == 0)                 # high cost only once degree days are in the line
        out = (g.hc == 0) & (g.hc_noclim == 1)                  # high cost only without degree days
        rows.append({"scope": scope, "households": len(g), "changed_group": float((g.y != g.y_noclim).mean()),
                     "into_high_cost": float(into.mean()), "out_of_high_cost": float(out.mean()),
                     "double_risk_with_climate": int((g.y == 3).sum()), "double_risk_without_climate": int((g.y_noclim == 3).sum()),
                     "double_risk_changed": float(((g.y == 3) != (g.y_noclim == 3)).sum() / max((g.y_noclim == 3).sum(), 1)),
                     "rel_log_hdd_into": float(g.rel_log_hdd[into].mean()), "rel_log_hdd_out": float(g.rel_log_hdd[out].mean()),
                     "corr_rel_log_hdd_thr_change": float(np.corrcoef(g.rel_log_hdd, np.log(g.thr.clip(lower=1)) -
                                                                     np.log(b.loc[g.index, "thr"].clip(lower=1)))[0, 1])})
        tab = pd.crosstab(g.y_noclim.map(CLASS_NAMES), g.y.map(CLASS_NAMES))
        for r in tab.index:
            for c in tab.columns:
                trans.append({"scope": scope, "without_climate": r, "with_climate": c, "households": int(tab.loc[r, c])})
    return pd.DataFrame(rows), pd.DataFrame(trans)


def country_cutoffs(dev: pd.DataFrame) -> dict[str, int]:
    lab = eligible(dev, "p80")
    out = {}
    for c, g in lab.groupby("country"):
        cum = g.income_bracket.value_counts(normalize=True).sort_index().cumsum()
        out[c] = int((cum - TARGET_LOW_INCOME_SHARE).abs().idxmin())
    return out


def income_cutoff(dev: pd.DataFrame, specs: dict) -> pd.DataFrame:
    cut = country_cutoffs(dev)
    print("country cut-offs (bracket <=):", cut, flush=True)
    rows = []
    for line, s in specs.items():
        e = eligible(dev, s["family"])
        thr = expenditure_threshold(e, s["family"], s["tau"])
        for variant in ("bracket <= 3 (study)", "country bottom ~30%"):
            lab = make_labels(dev, s["family"], s["cutoff"], s["tau"], thr=thr)
            if variant.startswith("country"):
                lab["li"] = (lab.income_bracket <= lab.country.map(cut)).astype(int)
                lab["y"] = lab.li + 2 * lab.hc
            counts = lab.groupby(["country", "y"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
            rec = {"line": line, "income_cutoff": variant, "households": len(lab),
                   "low_income_share_min": float(lab.groupby("country").li.mean().min()),
                   "low_income_share_max": float(lab.groupby("country").li.mean().max()),
                   "smallest_group_any_country": int(counts.to_numpy().min()),
                   "double_risk_share": float((lab.y == 3).mean()),
                   "hardship_gap": f2_gap(lab), "b0_balanced_accuracy": f1_b0(lab)}
            rec.update({f"low_income_share_{c}": float(v) for c, v in lab.groupby("country").li.mean().items()})
            rows.append(rec)
            print(f"{line:14s} {variant:22s} gap={rec['hardship_gap']:.3f} BA={rec['b0_balanced_accuracy']:.3f} "
                  f"min-group={rec['smallest_group_any_country']}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    dev = load_development_frame()
    specs = line_specs()
    sw, tr = climate_switch(dev, specs)
    sw.to_csv(OUT / "table_label_climate_switch.csv", index=False)
    tr.to_csv(OUT / "table_label_climate_transitions.csv", index=False)
    print(sw.round(3).to_string(index=False), flush=True)
    income_cutoff(dev, specs).to_csv(OUT / "table_label_income_cutoff.csv", index=False)


if __name__ == "__main__":
    main()
