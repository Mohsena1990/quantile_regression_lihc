"""Sensitivity check (b): modelled heating need and hidden under-heating (lib/identification/required_heat.py).

Questions:
  1. Does observed spending follow modelled heating need? (Spearman correlation per country)
  2. How many households spend less than half of what comparable households spend for the same need (hidden
     under-heating), and where do the four-class labels put them? An expenditure label can never call them high cost,
     so a low-income under-heater lands in "Income Risk", never "Double Risk".
  3. Is under-heating real hardship? Self-reported income difficulty (S8 >= 3) and energy-bill aid (S7) among
     low-income under-heaters against other low-income households.
  4. How much larger is the Double-Risk group if low-income hidden under-heaters are added ("extended Double Risk")?
Labels: Van Hove (vh) and the three dynamic thresholds (dyn_m1/m5/m7). Development partitions only.

Outputs: outputs/tables/table_hidden_underheating.csv (per label x country), table_hidden_underheating_by_class.csv;
outputs/figures/hidden_underheating.
--v5: the version-5 lines (Van Hove, Quantile Regression on need with and without climate, energy burden) on the five study
countries -> the same files with the suffix _v5 (figure panels: the need line).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES as COUNTRIES_4, COUNTRIES_V5, load_development_frame
from lib.identification.labels import CLASS_NAMES, add_labels
from lib.identification.required_heat import hidden_underheating
from lib.reporting.figures import save_figure

LABELS = {"vh": "raw_q80", "qr80": "qr80", "hansen": "hansen", "hansen_qr": "hansen_qr"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CLASS_COLOR = {"No Risk": "#c9c8c1", "Income Risk": "#eda100", "Expenditure Risk": "#2a78d6", "Double Risk": "#d1342f"}


def labelled(v5: bool):
    if v5:
        from lib.identification.lines_v5 import study_labels_v5
        dev = load_development_frame(extended=True)
        for label, (lab, _) in study_labels_v5(dev).items():
            if label in ("vh", "needqr", "needqr_noclim", "burden"):
                yield label, lab
        return
    dev = load_development_frame()
    if "hdd_hs1617" not in dev.columns:
        raise SystemExit("B2 climate missing: run scripts/acquire_openmeteo_b2.py and scripts/fuse_openmeteo_b2.py first")
    for label, kind in LABELS.items():
        yield label, add_labels(dev, cost_kind=kind)


def main() -> None:
    v5 = "--v5" in sys.argv
    COUNTRIES = COUNTRIES_V5 if v5 else COUNTRIES_4
    sfx, panel = ("_v5", "needqr") if v5 else ("", "hansen_qr")
    rows, by_class = [], []
    for label, lab in labelled(v5):
        lab = hidden_underheating(lab)
        lab["class_name"] = lab.y.map(CLASS_NAMES)
        for c in COUNTRIES:
            g = lab[(lab.country == c) & lab.hidden_underheating.notna()]
            hu = g.hidden_underheating.eq(1)
            low = g.li.eq(1)
            hard = g.S8.ge(3).where(g.S8.notna())
            rho = stats.spearmanr(g.required_heat_kwh, g.cost)[0]
            dr = int(g.y.eq(3).sum())
            hidden_ep = int((hu & low).sum())
            rows.append({"label": label, "country": c, "n_with_need": len(g), "spearman_need_cost": rho,
                         "hidden_underheating_rate": float(hu.mean()), "hidden_underheating_low_income_n": hidden_ep,
                         "share_of_low_income_hidden": float(hu[low].mean()),
                         "hidden_low_income_in_income_risk": float(g[hu & low].y.eq(1).mean()) if hidden_ep else np.nan,
                         "double_risk_n": dr, "extended_double_risk_n": dr + hidden_ep,
                         "double_risk_growth": (dr + hidden_ep) / dr - 1 if dr else np.nan,
                         "hardship_low_income_hidden": float(hard[hu & low].mean()), "hardship_low_income_other": float(hard[~hu & low].mean()),
                         "hardship_double_risk": float(hard[g.y.eq(3)].mean()),
                         "aid_low_income_hidden": float(g[hu & low].S7.mean()), "aid_low_income_other": float(g[~hu & low].S7.mean())})
            t = g.groupby("class_name").hidden_underheating.agg(["size", "mean"]).reset_index()
            by_class.append(t.rename(columns={"size": "n", "mean": "hidden_underheating_rate"}).assign(label=label, country=c))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"outputs/tables/table_hidden_underheating{sfx}.csv", index=False)
    bc = pd.concat(by_class, ignore_index=True)
    bc.to_csv(ROOT / f"outputs/tables/table_hidden_underheating_by_class{sfx}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.patch.set_facecolor(SURFACE)
    d = bc[bc.label == panel]
    ax = axes[0]
    for i, cls in enumerate(CLASS_NAMES.values()):
        v = d[d.class_name == cls].set_index("country").reindex(COUNTRIES).hidden_underheating_rate
        ax.bar(np.arange(len(COUNTRIES)) + (i - 1.5) * 0.2, v * 100, width=0.19, color=CLASS_COLOR[cls], label=cls)
    ax.set_xticks(range(len(COUNTRIES)), COUNTRIES)
    ax.set_ylabel("% hidden under-heaters", fontsize=8.5, color=INK2)
    ax.set_title(f"Hidden under-heating by class ({'Quantile Regression on need' if v5 else 'Hansen + Quantile Regression'} label)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    o = out[out.label == panel].set_index("country").reindex(COUNTRIES)
    x = np.arange(len(COUNTRIES))
    for j, (col, name, color) in enumerate((("hardship_low_income_hidden", "low income, hidden under-heater", "#d1342f"),
                                            ("hardship_low_income_other", "low income, other", "#eda100"),
                                            ("hardship_double_risk", "Double Risk", "#2a78d6"))):
        ax.bar(x + (j - 1) * 0.26, o[col] * 100, width=0.25, color=color, label=name)
    ax.set_xticks(x, COUNTRIES)
    ax.set_ylabel("% finding it (very) difficult on present income (S8)", fontsize=8.5, color=INK2)
    ax.set_title("Is hidden under-heating hardship?", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    for a in axes:
        a.set_facecolor(SURFACE); a.grid(axis="y", color=GRID, lw=0.7); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    save_figure(fig, ROOT / f"outputs/figures/hidden_underheating{sfx}")
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
