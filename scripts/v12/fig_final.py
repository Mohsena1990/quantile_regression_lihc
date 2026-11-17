"""Publication versions of Figs 7-11 (block alone, steps, Shapley, direction, best map) at print size with large text.
Figures are drawn at 7.0 in wide, the width at which they are printed (slightly wider than the text block), so 9-10 pt text stays
9-10 pt on paper."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from common import *
T = "/home/claude/work/v12/tables/"; F = "/home/claude/work/v12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "savefig.bbox": "tight",
                     "savefig.dpi": 600, "legend.frameon": False, "pdf.fonttype": 42, "axes.linewidth": 0.8})
W = 7.0
CC = {"Bulgaria": "#2a78d6", "Hungary": "#e34948", "Italy": "#1baf7a", "Poland": "#eda100", "Spain": "#8f5bd6", "All five": "#0b0b0b"}
AB = {"Bulgaria": "BG", "Hungary": "HU", "Italy": "IT", "Poland": "PL", "Spain": "ES", "All five": "All"}
CN = COUNTRIES + ["All five"]
SHORT = {"I": "Income", "H": "Household", "D": "Dwelling", "S": "Heating\nsystem & fuel", "R": "Region", "C": "Climate"}
TITLE = {"R0": "R0  fixed spending line (P80)", "R1": "R1  energy-burden line", "R2": "R2  need line", "R3": "R3  need line + climate"}
HL = "#d9480f"


def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color="#e4e3df", lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(F + name + ".pdf"); fig.savefig(F + name + ".png", dpi=200); plt.close(fig); print(name)


# ---------------- Fig 7: each block on its own (gain over income alone) --------------------------------------------
BA = pd.read_csv(T + "block_alone.csv"); BA = BA[BA.model == "catboost"]
blocks = ["H", "D", "S", "R", "C"]
fig, axs = plt.subplots(2, 2, figsize=(W, 6.6), sharex=True, sharey=True)
for ax, r in zip(axs.flat, RSPEC):
    x = BA[BA.R == r]
    g = x.pivot(index="block", columns="country", values="gain").reindex(index=blocks, columns=CN)
    s = x.pivot(index="block", columns="country", values="sig").reindex(index=blocks, columns=CN)
    im = ax.imshow(g.to_numpy(), cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    for j in range(len(CN)):
        col = g.iloc[:, j]; best = col.idxmax()
        for i, b in enumerate(blocks):
            v = g.iloc[i, j]; sig = bool(s.iloc[i, j])
            ax.text(j, i, f"{v:.2f}".replace("-0.00", "0.00") + ("*" if sig else ""), ha="center", va="center", fontsize=7.9,
                    color="white" if abs(v) > 0.12 else "#1a1a1a", fontweight="bold" if sig else "normal")
        if s.loc[best].iloc[j] and g.loc[best].iloc[j] > 0:
            ax.add_patch(Rectangle((j - 0.48, blocks.index(best) - 0.46), 0.96, 0.92, fill=False, ec=HL, lw=2))
    ax.set_xticks(range(len(CN))); ax.set_xticklabels([AB[c] for c in CN])
    ax.set_yticks(range(len(blocks))); ax.set_yticklabels([{"H": "Household", "D": "Dwelling", "S": "Heating", "R": "Region", "C": "Climate"}[b] for b in blocks])
    ax.set_title(TITLE[r], loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.subplots_adjust(wspace=0.06, hspace=0.25, right=0.86)
cax = fig.add_axes([0.89, 0.25, 0.022, 0.5]); cb = fig.colorbar(im, cax=cax)
cb.set_label("Gain in balanced accuracy over income alone"); cb.outline.set_visible(False)
fig.text(0.47, 0.02, "* significant at 5% (paired bootstrap)     orange box: largest significant gain in that country",
         ha="center", fontsize=8.8, color="#3d3c39")
save(fig, "fig_block_alone")

# ---------------- Fig 8: blocks added in policy order -----------------------------------------------------------------
M = pd.read_csv(T + "metrics_all.csv")
STEPS = ["I", "I+H", "I+H+D", "I+H+D+S", "I+H+D+S+R", "I+H+D+S+R+C"]
LAB = ["Income", "+ Household", "+ Dwelling", "+ Heating", "+ Region", "+ Climate"]
fig, axs = plt.subplots(2, 2, figsize=(W, 6.9), sharey=True, sharex=True)
for ax, r in zip(axs.flat, RSPEC):
    x = M[M.R == r]
    for c in CN:
        v = x[(x.model == "catboost") & (x.country == c)].set_index("set").balanced_accuracy.reindex(STEPS)
        a = c == "All five"
        ax.plot(range(6), v, "-o", color=CC[c], lw=2.6 if a else 1.5, ms=5 if a else 3.8, zorder=3 if a else 2, label=c)
        k = int(np.nanargmax(v.to_numpy()))
        ax.scatter([k], [v.iloc[k]], s=95, facecolor="none", edgecolor=CC[c], lw=1.6, zorder=4)
        pl = x[(x.model == "catboost") & (x.country == c) & (x.set == "ALL_placebo")].balanced_accuracy
        ax.scatter([5.3], pl, marker="x", color=CC[c], s=26, lw=1.3, zorder=4)
    v = x[(x.model == "logit") & (x.country == "All five")].set_index("set").balanced_accuracy.reindex(STEPS)
    ax.plot(range(6), v, ls=":", color="#0b0b0b", lw=1.8, label="All five, logistic regression")
    ax.axhline(0.25, color="#b9b8ae", lw=0.8, ls="--")
    ax.set_xticks(range(6)); ax.set_xticklabels(LAB, rotation=35, ha="right")
    ax.set_title(TITLE[r], loc="left", fontweight="bold"); clean(ax); ax.set_ylim(0.42, 0.82); ax.set_xlim(-0.3, 5.55)
for a in axs[:, 0]: a.set_ylabel("Balanced accuracy (out of sample)")
h, l = axs[0, 0].get_legend_handles_labels()
h += [Line2D([], [], ls="", marker="o", mfc="none", mec="#3d3c39", ms=9, mew=1.4), Line2D([], [], ls="", marker="x", color="#3d3c39", ms=6)]
l += ["Highest point for the country", "Full model with placebo climate"]
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
fig.subplots_adjust(wspace=0.06, hspace=0.32)
save(fig, "fig_steps")

# ---------------- Fig 9: Shapley fair shares ------------------------------------------------------------------------
Sh = pd.read_csv(T + "shapley_blocks.csv"); Sh = Sh[Sh.model == "catboost"]
BC = {"H": "#b9b8ae", "D": "#8f5bd6", "S": "#eda100", "R": "#2a78d6", "C": "#1baf7a"}
fig, axs = plt.subplots(2, 2, figsize=(W, 6.4), sharex=True, sharey=True)
for ax, r in zip(axs.flat, RSPEC):
    x = Sh[Sh.R == r]
    for k, c in enumerate(CN):
        z = x[x.country == c].set_index("block").shapley_ba
        lp, ln = 0, 0; top = z.drop("I").idxmax()
        for b in ["H", "D", "S", "R", "C"]:
            v = z[b]
            kw = dict(color=BC[b], edgecolor=HL if b == top else "white", lw=1.8 if b == top else 0.5, height=0.72,
                      zorder=3 if b == top else 2)
            if v >= 0: ax.barh(k, v, left=lp, **kw); lp += v
            else: ax.barh(k, v, left=ln, **kw); ln += v
        ax.text(lp + 0.004, k, f"+ income {z['I']:.2f}", va="center", fontsize=8.4, color="#3d3c39")
    ax.set_yticks(range(len(CN))); ax.set_yticklabels(CN); ax.axvline(0, color="#0b0b0b", lw=0.7)
    ax.set_title(TITLE[r], loc="left", fontweight="bold"); ax.set_xlim(-0.02, 0.30); clean(ax, grid="x")
for a in axs[1]: a.set_xlabel("Fair share of balanced accuracy (Shapley)")
axs[0, 0].invert_yaxis()
h = [Rectangle((0, 0), 1, 1, fc=BC[b], ec="white") for b in ["H", "D", "S", "R", "C"]] + [Rectangle((0, 0), 1, 1, fc="white", ec=HL, lw=1.8)]
l = [SHORT[b].replace("\n", " ") for b in ["H", "D", "S", "R", "C"]] + ["Largest block after income"]
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
fig.subplots_adjust(wspace=0.16, hspace=0.3)
save(fig, "fig_shapley")

# ---------------- Fig 10: direction of effects (SHAP) ---------------------------------------------------------------
D = pd.read_csv(T + "direction_shap.csv")
fig, axs = plt.subplots(2, 1, figsize=(W, 6.8))
w = 0.16
ax = axs[0]; order = ["<45 m2", "45-64", "65-90", "91-120", "121-200", ">200 m2"]
x = D[(D.R == "R0") & (D.variable == "H3") & (D.n >= 10)]
for k, c in enumerate(COUNTRIES):
    z = x[x.country == c].set_index("value").mean_shap_LIHC.reindex(order)
    ax.bar(np.arange(6) + (k - 2) * w, z, width=w, color=CC[c], label=c)
ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_xticks(range(6))
ax.set_xticklabels(["< 45 m²", "45–64 m²", "65–90 m²", "91–120 m²", "121–200 m²", "> 200 m²"])
ax.set_title("(a)  R0 fixed spending line: floor area of the dwelling", loc="left", fontweight="bold", pad=16)
ax.set_ylabel("Effect on chance of LIHC (SHAP)\n< 0 lowers     > 0 raises"); clean(ax)
fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
ax = axs[1]; fuels = ["electricity", "gas_lpg", "district_heating", "wood", "coal_coke"]
fl = {"electricity": "Electricity", "gas_lpg": "Gas / LPG", "district_heating": "District heating", "wood": "Wood", "coal_coke": "Coal"}
x = D[(D.R == "R2") & (D.variable == "primary_heating_fuel") & (D.n >= 15)]
for k, c in enumerate(COUNTRIES):
    z = x[x.country == c].set_index("value").mean_shap_LIHC.reindex(fuels)
    ax.bar(np.arange(5) + (k - 2) * w, z, width=w, color=CC[c])
ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([fl[f] for f in fuels])
ax.set_title("(b)  R2 need line: main heating fuel (where at least 15 households)", loc="left", fontweight="bold", pad=16)
ax.set_ylabel("Effect on chance of LIHC (SHAP)\n< 0 lowers     > 0 raises"); clean(ax)
fig.subplots_adjust(hspace=0.38)
save(fig, "fig_direction")

# ---------------- Fig 11: block influence with best combination -----------------------------------------------------
B = pd.read_csv(T + "best_subset.csv"); B = B[B.model == "catboost"]
fig, axs = plt.subplots(2, 2, figsize=(W, 6.6), sharex=True, sharey=True)
BL = list(ORDER); LBL = ["Income", "Household", "Dwelling", "Heating", "Region", "Climate"]
for ax, r in zip(axs.flat, RSPEC):
    Mx = Sh[Sh.R == r].pivot_table(index="country", columns="block", values="shapley_ba").reindex(index=CN, columns=BL)
    im = ax.imshow(Mx.values, cmap="Blues", vmin=0, vmax=0.10, aspect="auto")
    for i, c in enumerate(CN):
        bb = B[(B.R == r) & (B.country == c)].iloc[0]
        best = bb.best_set.split("+")
        for j, b in enumerate(BL):
            v = Mx.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.6, color="white" if v > 0.065 else "#1a1a1a",
                    fontweight="bold" if b in best else "normal")
            if b in best:
                ax.add_patch(Rectangle((j - 0.46, i - 0.43), 0.92, 0.86, fill=False, ec=HL, lw=1.8))
        ax.text(6.15, i, f"{bb.ba_best:.2f}", va="center", ha="center", fontsize=8.6, color=HL, fontweight="bold")
    ax.set_xticks(range(7)); ax.set_xticklabels(LBL + ["Best BA"], rotation=35, ha="right"); ax.set_xlim(-0.5, 6.6)
    [t.set_color(HL) for t in ax.get_xticklabels()[-1:]]
    ax.set_yticks(range(6)); ax.set_yticklabels(CN); ax.set_title(TITLE[r], loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.subplots_adjust(hspace=0.36, wspace=0.08, bottom=0.17)
cax = fig.add_axes([0.3, 0.035, 0.4, 0.018]); cb = fig.colorbar(im, cax=cax, extend="max", orientation="horizontal")
cb.set_label("Fair share of balanced accuracy (Shapley)"); cb.outline.set_visible(False)
fig.text(0.5, -0.045, "Orange boxes and bold values: blocks in the best combination;  best BA: its balanced accuracy",
         ha="center", fontsize=8.8, color="#3d3c39")
[t.set_color(HL) for a in axs[1] for t in a.get_xticklabels()[-1:]]
save(fig, "fig_bestmap")
