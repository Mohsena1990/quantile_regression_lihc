import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from common import *
T = "/home/claude/work/v12/tables/"; O = "/home/claude/work/v12/ms/"
M = pd.read_csv(T + "metrics_all.csv"); C = pd.read_csv(T + "comparisons.csv"); E = pd.read_csv(T + "region_eta2.csv")
MN = {"logit": "Logistic regression", "catboost": "CatBoost", "xgboost": "XGBoost"}
f = lambda v, d=3: f"{v:.{d}f}"
# ---- headline table: full model, all five, all metrics
x = M[(M.set == "I+H+D+S+R+C") & (M.country == "All five")]
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{All evaluation metrics of the full model (six blocks), five countries together, out of sample.}",
     r"\label{tab:headline}", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}", r"\begin{adjustbox}{max width=\textwidth}",
     r"\begin{tabular}{llccccccccccccc}", r"\toprule",
     r"Threshold & Model & Acc. & \makecell{Bal.\\acc. (95\% CI)} & \makecell{F1\\macro} & \makecell{F1\\wtd.} & $\kappa$ & MCC & AUC & \makecell{Log\\loss} & Brier & \makecell{Prec.\\LIHC} & \makecell{Rec.\\LIHC} & \makecell{F1\\LIHC} & \makecell{Train\\bal. acc.} \\",
     r"\midrule"]
for r in RSPEC:
    for m in ("logit", "catboost", "xgboost"):
        z = x[(x.R == r) & (x.model == m)].iloc[0]
        L.append(f"{r if m=='logit' else ''} & {MN[m]} & {f(z.accuracy)} & {f(z.balanced_accuracy)} ({f(z.ba_lo,2)}--{f(z.ba_hi,2)}) & {f(z.f1_macro)} & {f(z.f1_weighted)} & {f(z.kappa)} & {f(z.mcc)} & {f(z.roc_auc_macro)} & {f(z.log_loss)} & {f(z.brier)} & {f(z.precision_LIHC)} & {f(z.recall_LIHC)} & {f(z.f1_LIHC)} & {f(z.train_ba)} \\\\")
    if r != "R3": L.append(r"\addlinespace")
i = M[(M.set == "I") & (M.country == "All five") & (M.model == "catboost")].set_index("R").balanced_accuracy
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\smallskip{\footnotesize\raggedright",
      r"\par Acc.: accuracy; Bal. acc.: balanced accuracy (chance 0.25); $\kappa$: Cohen's kappa; MCC: Matthews correlation; AUC: macro one-vs-rest ROC AUC; Prec./Rec.: precision/recall. Train bal. acc.: balanced accuracy on the training households (mean over folds); the difference to the test score measures overfitting. Income alone (CatBoost): "
      + ", ".join(f"{r} {i[r]:.3f}" for r in RSPEC) + r". Country-level metrics for every model and information set are given in the supplementary data.\par}", r"\end{table}"]
open(O + "tab_headline.tex", "w").write("\n".join(L))
# ---- country table: balanced accuracy, LIHC precision/recall, catboost full; income-only BA
x = M[(M.set == "I+H+D+S+R+C") & (M.model == "catboost")]
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Full model by country (CatBoost): balanced accuracy and LIHC precision and recall, with income alone for comparison.}",
     r"\label{tab:countryba}", r"\footnotesize", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Threshold & Metric & Bulgaria & Hungary & Italy & Poland & Spain & All five \\", r"\midrule"]
cs = COUNTRIES + ["All five"]
for r in RSPEC:
    io = M[(M.set == "I") & (M.model == "catboost") & (M.R == r)].set_index("country").balanced_accuracy
    z = x[x.R == r].set_index("country")
    L.append(f"{r} & Bal. acc., income only & " + " & ".join(f(io[c]) for c in cs) + r" \\")
    L.append(f" & Bal. acc., full model & " + " & ".join(f"{f(z.loc[c,'balanced_accuracy'])}" for c in cs) + r" \\")
    L.append(f" & LIHC precision & " + " & ".join(f(z.loc[c, 'precision_LIHC'], 2) for c in cs) + r" \\")
    L.append(f" & LIHC recall & " + " & ".join(f(z.loc[c, 'recall_LIHC'], 2) for c in cs) + r" \\")
    if r != "R3": L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(O + "tab_countryba.tex", "w").write("\n".join(L))
# ---- block significance summary
lab = [("step: +Household", "Add household"), ("step: +Dwelling", "Add dwelling fabric"), ("step: +Heating system & fuel", "Add heating system \\& fuel"),
       ("step: +Region", "Add region"), ("step: +Climate", "Add climate"), ("climate vs placebo", "Real vs placebo climate"),
       ("climate when region unknown", "Add climate, region unknown"), ("drop Income", "Remove income")]
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{What each block adds: change in balanced accuracy (five countries together, CatBoost) and number of significant gains and losses over three models and six scopes (five countries and all together).}",
     r"\label{tab:blocksig}", r"\footnotesize", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lcccc}", r"\toprule",
     r"Comparison & R0 fixed spending & R1 energy burden & R2 need & R3 need + climate \\", r"\midrule"]
for k, name in lab:
    cells = []
    for r in RSPEC:
        d = C[(C.comparison == k) & (C.R == r)]
        cb = d[(d.model == "catboost") & (d.country == "All five")].iloc[0]
        sign = -1 if k == "drop Income" else 1
        dd = sign * cb.d_ba
        up = int(((d.sig) & (d.d_ba > 0)).sum()); dn = int(((d.sig) & (d.d_ba < 0)).sum())
        if k == "drop Income": up, dn = dn, up
        cells.append(f"${dd:+.3f}$ ({up}+, {dn}$-$)")
    L.append(name + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\smallskip{\footnotesize\raggedright",
      r"\par Steps follow the policy order income $\rightarrow$ household $\rightarrow$ dwelling $\rightarrow$ heating system $\rightarrow$ region $\rightarrow$ climate. Significance: paired household bootstrap, 95\%; 18 comparisons per cell. ``Remove income'' is reported as the loss when income is removed from the full model.\par}", r"\end{table}"]
open(O + "tab_blocksig.tex", "w").write("\n".join(L))
# ---- eta table
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Model-free regional signal: excess between-region share of variance of high cost ($\eta^2$ minus its chance value) within each country.}",
     r"\label{tab:eta}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule", r"Threshold & Bulgaria & Hungary & Italy & Poland & Spain \\", r"\midrule"]
for r in RSPEC:
    z = E[E.R == r].set_index("country")
    L.append(r + " & " + " & ".join(f"{z.loc[c,'eta2_excess']:.3f}{'' if z.loc[c,'p']<0.05 else '$^{ns}$'}" for c in COUNTRIES) + r" \\")
reg = E.groupby("country").regions.first()
L += [r"\midrule", "Regions (NUTS2) & " + " & ".join(str(reg[c]) for c in COUNTRIES) + r" \\", r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip{\footnotesize\raggedright", r"\par $\eta^2$: between-region share of the variance of the high-cost indicator; chance value from 500 random reallocations of households to regions. $^{ns}$: not significant at 5\%.\par}", r"\end{table}"]
open(O + "tab_eta.tex", "w").write("\n".join(L))
print("ok")
