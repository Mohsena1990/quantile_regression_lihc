"""Figure and table: income only vs full model vs honestly selected set, and the information map per country and threshold."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
T = "/home/claude/work/v12/tables/"; F = "/home/claude/work/v12/figures/"; MS = "/home/claude/work/v12/ms/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})
N = pd.read_csv(T + "nested_best.csv"); Bf = pd.read_csv(T + "nested_block_freq.csv", skiprows=1, names=["thr", "country", "I", "H", "D", "S", "R", "C"]).rename(columns={})
M = pd.read_csv(T + "metrics_all.csv"); A = pd.read_csv(T + "block_alone.csv"); A = A[A.model == "catboost"]
CN = COUNTRIES + ["All five"]; AB = {"Bulgaria": "BG", "Hungary": "HU", "Italy": "IT", "Poland": "PL", "Spain": "ES", "All five": "All"}
fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True)
for ax, r in zip(axs.flat, RSPEC):
    x = N[N.R == r].set_index("country").reindex(CN)
    for i, c in enumerate(CN):
        ax.plot([i, i], [x.ba_income[c], x.ba_full[c]], color="#9a9890", lw=1.4, zorder=1)
        sig = x.hi[c] < 0 or x.lo[c] > 0
        ax.scatter(i + 0.18, x.ba_selected[c], marker="D", s=30, zorder=3, color="#d9480f" if sig else "#f2b48f",
                   edgecolor="#d9480f", lw=0.8)
    ax.scatter(range(6), x.ba_income, s=34, facecolor="white", edgecolor="#2a78d6", lw=1.5, zorder=2)
    ax.scatter(range(6), x.ba_full, s=38, color="#2a78d6", zorder=2)
    ax.set_xticks(range(6)); ax.set_xticklabels([AB[c] for c in CN]); ax.set_title(RNAME[r], fontsize=10.5, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True); ax.set_ylim(0.42, 0.82)
for a in axs[:, 0]: a.set_ylabel("Balanced accuracy (CatBoost, out of sample)")
h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#2a78d6", label="Income only"),
     plt.Line2D([], [], ls="", marker="o", color="#2a78d6", label="All six blocks"),
     plt.Line2D([], [], ls="", marker="D", color="#f2b48f", mec="#d9480f", label="Best set chosen in training data (= all six)"),
     plt.Line2D([], [], ls="", marker="D", color="#d9480f", label="Best set chosen in training data (significantly different)")]
fig.legend(handles=h, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.07))
fig.subplots_adjust(wspace=0.08, hspace=0.3)
fig.savefig(F + "fig_nested.pdf"); fig.savefig(F + "fig_nested.png", dpi=110); plt.close(fig)

# ---- information map table
NAME = {"H": "household", "D": "dwelling", "S": "heating", "R": "region", "C": "climate"}
rows = []
for c in CN:
    cells = []
    for r in RSPEC:
        n = N[(N.R == r) & (N.country == c)].iloc[0]
        a = A[(A.R == r) & (A.country == c) & A.sig & (A.gain > 0)].sort_values("gain", ascending=False)
        f = Bf[(Bf.thr == r) & (Bf.country == c)].iloc[0]
        top = [f"{NAME[b]}{'$^\\dagger$' if f[b] >= 0.8 else ''}" for b in a.block[:3]]
        mm = M[(M.R == r) & (M.country == c) & (M.model == "catboost")].set_index("set").balanced_accuracy
        cell = f"{mm['I']:.2f}$\\rightarrow${mm['I+H+D+S+R+C']:.2f} \\newline " + (", ".join(top) if top else "none")
        _ = f"{n.ba_income:.2f}$\\rightarrow${n.ba_full:.2f} \\newline " + (", ".join(top) if top else "none")
        cells.append(cell)
    rows.append(f"{c} & " + " & ".join(cells) + r" \\")
tex = r"""\begin{table}[!htbp]
\centering
\caption{Information map: balanced accuracy with income only and with all six blocks, and the blocks that significantly improve on
income when added on their own, largest first, by country and cost threshold (CatBoost, out of sample).}
\label{tab:infomap}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{L{1.6cm} L{3.3cm} L{3.3cm} L{3.3cm} L{3.3cm}}
\toprule
Country & R0 fixed spending & R1 energy burden & R2 need & R3 need + climate \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\par\smallskip{\footnotesize\raggedright
\par First line: income only $\rightarrow$ all six blocks. Second line: blocks with a significant gain over income alone (paired
bootstrap, 95\%), at most three. $^\dagger$: the block was also chosen in at least four of the five training folds when the best
combination of blocks was selected inside the training data. Climate is measured per region, so its gain on its own partly reflects
where a household lives.\par}
\end{table}
"""
open(MS + "tab_infomap.tex", "w").write(tex); print(tex)
