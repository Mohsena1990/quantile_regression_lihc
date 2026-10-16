"""Sensitivity check (b): modelled heating need and hidden under-heating (lib/identification/required_heat.py).

Questions:
  1. Does observed spending follow modelled heating need? (Spearman correlation per country)
  2. How many households spend less than half of what comparable households spend for the same need (hidden
     under-heating), and where do the four-class labels put them? An expenditure label can never call them high cost,
     so a low-income under-heater lands in "Income Risk", never "Double Risk".
  3. Is under-heating real hardship? Self-reported income difficulty (S8 >= 3) and energy-bill aid (S7) among
     low-income under-heaters against other low-income households.
  4. How much larger is the Double-Risk group if low-income hidden under-heaters are added ("extended Double Risk")?
Lines: the four study lines (lib/identification/lines.py) on the five study countries. Development households only.

Outputs: outputs/tables/table_hidden_underheating.csv (per label x country), table_hidden_underheating_by_class.csv;
outputs/figures/hidden_underheating.
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
from lib.identification.data import COUNTRIES, load_development_frame
from lib.identification.lines import CLASS_NAMES, study_labels
from lib.identification.required_heat import hidden_underheating
from lib.reporting.figures import save_figure

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CLASS_COLOR = {"No Risk": "#c9c8c1", "Income Risk": "#eda100", "Expenditure Risk": "#2a78d6", "Double Risk": "#d1342f"}


def main() -> None:
    panel = "needqr"
    rows, by_class = [], []
    for label, (lab, _) in study_labels(load_development_frame()).items():
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
    out.to_csv(ROOT / f"outputs/tables/table_hidden_underheating.csv", index=False)
    bc = pd.concat(by_class, ignore_index=True)
    bc.to_csv(ROOT / f"outputs/tables/table_hidden_underheating_by_class.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.patch.set_facecolor(SURFACE)
    d = bc[bc.label == panel]
    ax = axes[0]
    for i, cls in enumerate(CLASS_NAMES.values()):
        v = d[d.class_name == cls].set_index("country").reindex(COUNTRIES).hidden_underheating_rate
        ax.bar(np.arange(len(COUNTRIES)) + (i - 1.5) * 0.2, v * 100, width=0.19, color=CLASS_COLOR[cls], label=cls)
    ax.set_xticks(range(len(COUNTRIES)), COUNTRIES)
    ax.set_ylabel("% hidden under-heaters", fontsize=8.5, color=INK2)
    ax.set_title("Hidden under-heating by group (need line)", fontsize=10, loc="left")
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
    save_figure(fig, ROOT / f"outputs/figures/hidden_underheating")
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
