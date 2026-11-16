import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; F = "/home/claude/work/v12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})
QC = {"HILC": "#b9b8ae", "LILC": "#eda100", "HIHC": "#2a78d6", "LIHC": "#e34948"}
L = pd.read_pickle(S + "labels.pkl")
# ---------- Fig: households in the income-cost plane, four thresholds
fig, axs = plt.subplots(4, 5, figsize=(13, 10.5), sharex=True, sharey=True)
rng = np.random.default_rng(1)
for i, (r, e) in enumerate(L.items()):
    for j, c in enumerate(COUNTRIES):
        ax = axs[i, j]; g = e[e.country == c]
        x = g.income_bracket + rng.uniform(-0.3, 0.3, len(g)); y = np.log10((g.cost / g.thr).clip(0.05, 20))
        for q, col in QC.items():
            m = g.y.map(QUAD) == q
            ax.scatter(x[m], y[m], s=6, color=col, alpha=0.75, lw=0, label=q)
        ax.axvline(3.5, color="#0b0b0b", lw=1); ax.axhline(0, color="#0b0b0b", lw=1)
        sh = g.y.map(QUAD).value_counts(normalize=True)
        ax.text(0.03, 0.97, f"LIHC {sh.get('LIHC',0):.0%}", transform=ax.transAxes, va="top", fontsize=9, color="#e34948", fontweight="bold")
        if i == 0: ax.set_title(c, fontsize=11)
        if j == 0: ax.set_ylabel(RNAME[r].replace(" line", "\nline").replace("+ climate", "\n+ climate") + "\n\nspending ÷ threshold", fontsize=9)
        if i == 3: ax.set_xlabel("Income bracket")
        ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["0.1×", "1×", "10×"]); ax.set_xticks([1, 3, 5, 7, 10])
        for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, ["HILC (high income, low cost)", "LILC (low income, low cost)", "HIHC (high income, high cost)", "LIHC (low income, high cost)"],
           loc="upper center", ncol=4, markerscale=3, bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.savefig(F + "fig_quadrants.pdf"); fig.savefig(F + "fig_quadrants.png", dpi=110); plt.close(fig)
# ---------- Fig: EEA inability to keep home warm
raw = pd.read_csv("/mnt/user-data/attachments/c08a9517-68ec-5ad8-aa50-30b1cf29b014/3d99f0b8-energy_poverty.csv", skiprows=4)
raw = raw.dropna(subset=["years"]); raw["years"] = raw.years.astype(int); raw = raw[raw.years <= 2023]
col = {c: [k for k in raw.columns if k.startswith(c + "_")][0] for c in COUNTRIES + ["EU"]}
CC = {"Bulgaria": "#2a78d6", "Hungary": "#e34948", "Italy": "#1baf7a", "Poland": "#eda100", "Spain": "#8f5bd6", "EU": "#0b0b0b"}
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for c, k in col.items():
    ax.plot(raw.years, raw[k], color=CC[c], lw=2.4 if c == "EU" else 1.8, ls="--" if c == "EU" else "-")
    v = raw[k].dropna(); off = {"Spain": 1.3, "Bulgaria": -1.3, "EU": 1.0, "Italy": -0.9}.get(c, 0); ax.text(raw.years[v.index[-1]] + 0.15, v.iloc[-1] + off, f"{'EU-27' if c=='EU' else c} {v.iloc[-1]:.1f}", va="center", fontsize=9, color=CC[c])
ax.set_xlim(2014, 2026); ax.set_ylim(0, 43); ax.set_ylabel("Unable to keep home adequately warm (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(F + "fig_context_eea.pdf"); fig.savefig(F + "fig_context_eea.png", dpi=130); plt.close(fig)
print("ok")
