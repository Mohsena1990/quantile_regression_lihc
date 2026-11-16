"""Best combination of blocks per country and threshold: all metrics, block influence (Shapley) and honest check (nested)."""
import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import *
from metrics import all_metrics, boot_ci, _ba
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
F = "/home/claude/work/v12/figures/"; MS = "/home/claude/work/v12/ms/"
L = pd.read_pickle(S + "labels.pkl"); sh = pickle.load(open(S + "v12/shapley.pkl", "rb"))
B = pd.read_csv(T + "best_subset.csv"); B = B[B.model == "catboost"]
N = pd.read_csv(T + "nested_best.csv"); SH = pd.read_csv(T + "shapley_blocks.csv"); SH = SH[SH.model == "catboost"]
P = {}
for o in sh:
    for (s, m), p in o["proba"].items():
        if m == "catboost":
            P.setdefault((o["r"], s), np.zeros((len(L[o["r"]]), 4)))[o["te"]] = p
CN = COUNTRIES + ["All five"]; rows = []
for r, e in L.items():
    y = e.y.to_numpy()
    for c in CN:
        ix = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
        b = B[(B.R == r) & (B.country == c)].iloc[0]; best = tuple(b.best_set.split("+"))
        for lab, s in (("Income only", ("I",)), ("Best combination", best), ("All six", tuple(ORDER))):
            p = P[(r, s)][ix]; m = all_metrics(y[ix], p); lo, hi = boot_ci(y[ix], p.argmax(1), _ba, B=400, seed=1)
            rows.append({"R": r, "country": c, "set_role": lab, "set": "+".join(s), **m, "ba_lo": lo, "ba_hi": hi})
D = pd.DataFrame(rows); D.to_csv(T + "best_report.csv", index=False)

# ---- table: best combination per country and threshold
NAME = {"I": "Inc", "H": "Hh", "D": "Dw", "S": "Heat", "R": "Reg", "C": "Clim"}
lines = []
for r in RSPEC:
    first = True
    for c in CN:
        g = D[(D.R == r) & (D.country == c)].set_index("set_role")
        bst = g.loc["Best combination"]; inc = g.loc["Income only"]; full = g.loc["All six"]
        n = N[(N.R == r) & (N.country == c)].iloc[0]
        sh_ = SH[(SH.R == r) & (SH.country == c) & (SH.block != "I")].sort_values("shapley_ba", ascending=False).iloc[0]
        lines.append(f"{RNAME[r].split(' ')[0] if first else ''} & {c} & {inc.balanced_accuracy:.3f} & "
                     f"{' + '.join(NAME[k] for k in bst.set.split('+'))} & \\textbf{{{bst.balanced_accuracy:.3f}}} & "
                     f"{bst.f1_macro:.3f} & {bst.mcc:.3f} & {bst.roc_auc_macro:.3f} & {bst.precision_LIHC:.2f} & {bst.recall_LIHC:.2f} & "
                     f"{full.balanced_accuracy:.3f} & {n.ba_selected:.3f} & {NAME[sh_.block]} \\\\")
        first = False
    lines.append(r"\midrule" if r != "R3" else "")
tex = r"""\begin{sidewaystable}
\centering
\caption{Best combination of information blocks for each country and cost threshold: all evaluation metrics on held-out households
(CatBoost, five-fold cross-validation).}
\label{tab:best}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llcl ccccccc cc}
\toprule
 & & Income & \multicolumn{7}{c}{Best combination} & All six & Check on & Most influential \\
\cmidrule(lr){4-10}
 & Country & only (BA) & Blocks & BA & F1 & MCC & AUC & LIHC prec. & LIHC rec. & & new households & block after income \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\par\smallskip{\footnotesize\raggedright
\par BA: balanced accuracy (chance 0.25); F1: macro F1; MCC: Matthews correlation; AUC: macro ROC AUC. Blocks: Inc income, Hh household,
Dw dwelling fabric, Heat heating system and fuel, Reg region, Clim climate. Best combination: highest balanced accuracy among the 32
combinations that contain income. Because the best of 32 is chosen on the same households on which it is scored, its accuracy is
slightly optimistic; ``check on new households'' repeats the choice inside the training data only and scores it on households not used
to choose it (Section~\ref{sec:best}). Most influential block: largest fair share (Shapley value) after income.\par}
\end{sidewaystable}
"""
open(MS + "tab_best.tex", "w").write(tex)

# ---- figure: block influence (Shapley) per country, best combination outlined
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300})
fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4))
BL = list(ORDER); LBL = ["Income", "Household", "Dwelling", "Heating", "Region", "Climate"]
for ax, r in zip(axs.flat, RSPEC):
    M = SH[SH.R == r].pivot_table(index="country", columns="block", values="shapley_ba").reindex(index=CN, columns=BL)
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=0.10, aspect="auto")
    for i, c in enumerate(CN):
        best = B[(B.R == r) & (B.country == c)].best_set.iloc[0].split("+")
        for j, b in enumerate(BL):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.07 else "#1a1a1a")
            if b in best:
                ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec="#d9480f", lw=1.8))
    ax.set_xticks(range(6)); ax.set_xticklabels(LBL, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(CN, fontsize=9); ax.set_title(RNAME[r], fontsize=10.5, loc="left")
    for s in ax.spines.values(): s.set_visible(False)

fig.text(0.45, -0.02, "Orange boxes: blocks in the best combination for that country and threshold", ha="center", fontsize=9)
fig.subplots_adjust(hspace=0.45, wspace=0.35, right=0.88)
cax = fig.add_axes([0.91, 0.3, 0.015, 0.4]); fig.colorbar(im, cax=cax, extend="max", label="Fair share of balanced accuracy (Shapley)")
fig.savefig(F + "fig_bestmap.pdf"); fig.savefig(F + "fig_bestmap.png", dpi=110); plt.close(fig)
print(D[D.set_role == "Best combination"][["R", "country", "set", "balanced_accuracy", "ba_lo", "ba_hi", "f1_macro", "mcc", "precision_LIHC", "recall_LIHC"]].round(3).to_string(index=False))
