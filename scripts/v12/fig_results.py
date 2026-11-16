import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
T = "/home/claude/work/v12/tables/"; F = "/home/claude/work/v12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})
CC = {"Bulgaria": "#2a78d6", "Hungary": "#e34948", "Italy": "#1baf7a", "Poland": "#eda100", "Spain": "#8f5bd6", "All five": "#0b0b0b"}
M = pd.read_csv(T + "metrics_all.csv")
STEPS = ["I", "I+H", "I+H+D", "I+H+D+S", "I+H+D+S+R", "I+H+D+S+R+C"]
LAB = ["Income", "+ Household", "+ Dwelling", "+ Heating system", "+ Region", "+ Climate"]
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
# ---- Fig: policy-order steps by country (CatBoost), with logit for All five
fig, axs = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
for ax, r in zip(axs, RSPEC):
    x = M[(M.R == r)]
    for c in COUNTRIES + ["All five"]:
        v = x[(x.model == "catboost") & (x.country == c)].set_index("set").balanced_accuracy.reindex(STEPS)
        ax.plot(range(6), v, "-o", color=CC[c], lw=2.6 if c == "All five" else 1.5, ms=4, label=c, zorder=3 if c == "All five" else 2)
        pl = x[(x.model == "catboost") & (x.country == c) & (x.set == "ALL_placebo")].balanced_accuracy
        ax.scatter([5.25], pl, marker="x", color=CC[c], s=22, zorder=4)
    v = x[(x.model == "logit") & (x.country == "All five")].set_index("set").balanced_accuracy.reindex(STEPS)
    ax.plot(range(6), v, ls=":", color="#0b0b0b", lw=1.6, label="All five, logistic regression")
    ax.set_xticks(range(6)); ax.set_xticklabels(LAB, fontsize=8.5, rotation=40, ha="right"); ax.set_title(RNAME[r], fontsize=10.5, loc="left")
    ax.axhline(0.25, color="#b9b8ae", lw=0.8, ls="--"); clean(ax); ax.set_ylim(0.42, 0.82)
axs[0].set_ylabel("Balanced accuracy (CatBoost, out of sample)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=7, fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
fig.text(0.5, -0.12, "x: full model with placebo climate (climate swapped between regions of the same country)", ha="center", fontsize=8.5)
fig.subplots_adjust(wspace=0.08)
fig.savefig(F + "fig_steps.pdf"); fig.savefig(F + "fig_steps.png", dpi=110); plt.close(fig)
print("ok")
# ---- Fig: direction of effects
D = pd.read_csv(T + "direction_shap.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.2]})
ax = axs[0]; order = ["<45 m2", "45-64", "65-90", "91-120", "121-200", ">200 m2"]
x = D[(D.R == "R0") & (D.variable == "H3") & (D.n >= 10)]
w = 0.15
for k, c in enumerate(COUNTRIES):
    z = x[x.country == c].set_index("value").mean_shap_LIHC.reindex(order)
    ax.bar(np.arange(6) + (k - 2) * w, z, width=w, color=CC[c], label=c)
ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_xticks(range(6)); ax.set_xticklabels(order, fontsize=9)
ax.set_title("(a) R0 fixed spending line: floor area", loc="left", fontsize=10.5); ax.set_ylabel("Mean contribution to LIHC prediction (SHAP)")
ax.set_xlabel("Floor area of the dwelling"); clean(ax); ax.legend(fontsize=8.5)
ax = axs[1]; fuels = ["electricity", "gas_lpg", "district_heating", "wood", "coal_coke"]
fl = {"electricity": "Electricity", "gas_lpg": "Gas / LPG", "district_heating": "District heating", "wood": "Wood", "coal_coke": "Coal"}
x = D[(D.R == "R2") & (D.variable == "primary_heating_fuel") & (D.n >= 15)]
for k, c in enumerate(COUNTRIES):
    z = x[x.country == c].set_index("value").mean_shap_LIHC.reindex(fuels)
    ax.bar(np.arange(5) + (k - 2) * w, z, width=w, color=CC[c])
ax.axhline(0, color="#0b0b0b", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([fl[f] for f in fuels], fontsize=9)
ax.set_title("(b) R2 need line: main heating fuel", loc="left", fontsize=10.5); ax.set_xlabel("Main heating fuel (shown where at least 15 households)"); clean(ax)
fig.subplots_adjust(wspace=0.2)
fig.savefig(F + "fig_direction.pdf"); fig.savefig(F + "fig_direction.png", dpi=110); plt.close(fig)
print("dir ok")
# ---- Fig: Shapley fair shares (CatBoost), non-income blocks, with income annotated
Sh = pd.read_csv(T + "shapley_blocks.csv"); Sh = Sh[Sh.model == "catboost"]
BC = {"H": "#b9b8ae", "D": "#8f5bd6", "S": "#eda100", "R": "#2a78d6", "C": "#1baf7a"}
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
cs = COUNTRIES + ["All five"]
for ax, r in zip(axs, RSPEC):
    x = Sh[Sh.R == r]
    for k, c in enumerate(cs):
        z = x[x.country == c].set_index("block").shapley_ba
        left_pos, left_neg = 0, 0
        for b in ["H", "D", "S", "R", "C"]:
            v = z[b]
            if v >= 0: ax.barh(k, v, left=left_pos, color=BC[b], edgecolor="white", lw=0.5, label=BLOCK_NAME[b] if k == 0 else None); left_pos += v
            else: ax.barh(k, v, left=left_neg, color=BC[b], edgecolor="white", lw=0.5); left_neg += v
        ax.text(left_pos + 0.004, k, f"income {z['I']:.2f}", va="center", fontsize=7.5, color="#3d3c39")
    ax.set_yticks(range(len(cs))); ax.set_yticklabels(cs); ax.invert_yaxis(); ax.axvline(0, color="#0b0b0b", lw=0.7)
    ax.set_title(RNAME[r], loc="left", fontsize=10); ax.set_xlim(-0.02, 0.27)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=9)
fig.text(0.5, -0.03, "Fair share of balanced accuracy (Shapley value of each block, CatBoost); income share printed", ha="center", fontsize=9.5)
fig.subplots_adjust(wspace=0.08)
fig.savefig(F + "fig_shapley.pdf"); fig.savefig(F + "fig_shapley.png", dpi=110); plt.close(fig)
print("shap ok")
# ---- Fig: each block on its own (with income), CatBoost, heatmap
BA = pd.read_csv(T + "block_alone.csv"); BA = BA[BA.model == "catboost"]
blocks = ["H", "D", "S", "R", "C"]; cols = COUNTRIES + ["All five"]
fig, axs = plt.subplots(2, 2, figsize=(11, 6.6))
for ax, r in zip(axs.flat, RSPEC):
    x = BA[BA.R == r]
    g = x.pivot(index="block", columns="country", values="gain").reindex(index=blocks, columns=cols)
    s = x.pivot(index="block", columns="country", values="sig").reindex(index=blocks, columns=cols)
    im = ax.imshow(g.to_numpy(), cmap="RdBu_r", vmin=-0.22, vmax=0.22, aspect="auto")
    for i in range(len(blocks)):
        for j in range(len(cols)):
            v = g.iloc[i, j]
            ax.text(j, i, f"{v:.2f}" + ("*" if s.iloc[i, j] else ""), ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.13 else "#0b0b0b", fontweight="bold" if s.iloc[i, j] else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["BG", "HU", "IT", "PL", "ES", "All"], fontsize=9)
    ax.set_yticks(range(len(blocks))); ax.set_yticklabels([BLOCK_NAME[b] for b in blocks] if r in ("R0", "R2") else [], fontsize=9)
    ax.set_title(RNAME[r], loc="left", fontsize=10)
fig.subplots_adjust(wspace=0.08, hspace=0.35); cb = fig.colorbar(im, ax=axs, shrink=0.7, pad=0.02); cb.set_label("Gain over income alone\n(balanced accuracy)", fontsize=9)
fig.savefig(F + "fig_block_alone.pdf"); fig.savefig(F + "fig_block_alone.png", dpi=110); plt.close(fig)
print("alone ok")
