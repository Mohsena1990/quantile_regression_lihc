"""Full-page version of the income-cost plane figure: countries in rows, cost thresholds in columns."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; F = "/home/claude/work/v12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})
QC = {"HILC": "#b9b8ae", "LILC": "#eda100", "HIHC": "#2a78d6", "LIHC": "#e34948"}
TT = {"R0": "R0 fixed spending\nline (P80)", "R1": "R1 energy-burden\nline", "R2": "R2 need line", "R3": "R3 need line\n+ climate"}
L = pd.read_pickle(S + "labels.pkl")
fig, axs = plt.subplots(5, 4, figsize=(8.6, 11.6), sharex=True, sharey=True)
rng = np.random.default_rng(1)
for j, (r, e) in enumerate(L.items()):
    for i, c in enumerate(COUNTRIES):
        ax = axs[i, j]; g = e[e.country == c]
        x = g.income_bracket + rng.uniform(-0.3, 0.3, len(g)); y = np.log10((g.cost / g.thr).clip(0.05, 20))
        for q, col in QC.items():
            m = g.y.map(QUAD) == q
            ax.scatter(x[m], y[m], s=7, color=col, alpha=0.75, lw=0, label=q)
        ax.axvline(3.5, color="#0b0b0b", lw=1); ax.axhline(0, color="#0b0b0b", lw=1)
        sh = g.y.map(QUAD).value_counts(normalize=True)
        ax.text(0.04, 0.97, f"LIHC {sh.get('LIHC', 0):.0%}", transform=ax.transAxes, va="top", fontsize=10.5, color="#c92a2a",
                fontweight="bold")
        if i == 0: ax.set_title(TT[r], fontsize=11.5)
        if j == 0: ax.set_ylabel(f"{c}\n\nspending ÷ threshold", fontsize=11)
        if i == 4: ax.set_xlabel("Income bracket", fontsize=11)
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["0.1×", "1×", "10×"]); ax.set_xticks([1, 3, 5, 7, 10])
        ax.tick_params(labelsize=10)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
h, _ = axs[0, 0].get_legend_handles_labels()
fig.legend(h, ["HILC (high income, low cost)", "LILC (low income, low cost)", "HIHC (high income, high cost)",
               "LIHC (low income, high cost)"], loc="upper center", ncol=2, markerscale=3, bbox_to_anchor=(0.5, 0.97), fontsize=11)
fig.subplots_adjust(hspace=0.12, wspace=0.08, top=0.85)
fig.savefig(F + "fig_quadrants.pdf"); fig.savefig(F + "fig_quadrants.png", dpi=110); plt.close(fig)
print("ok")
