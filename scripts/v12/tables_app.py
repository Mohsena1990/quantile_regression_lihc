import sys; sys.path.insert(0, '.')
import pandas as pd
from common import *
T = "/home/claude/work/v12/tables/"; O = "/home/claude/work/v12/ms/"
I = pd.read_csv(T + "imbalance.csv"); I = I[I.country == "All five"]
HN = {"balanced_weights": "Balanced class weights (main)", "no_weights": "No weights", "oversample": "Random oversampling", "undersample": "Random undersampling"}
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Handling class imbalance: full model (CatBoost), five countries together.}", r"\label{tab:imbalance}", r"\footnotesize",
     r"\begin{tabular}{lcccc}", r"\toprule", r"Handling & R0 & R1 & R2 & R3 \\", r"\midrule"]
for met, nm in [("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "Macro F1"), ("precision_LIHC", "LIHC precision"), ("recall_LIHC", "LIHC recall")]:
    L.append(r"\multicolumn{5}{l}{\textit{" + nm + r"}} \\")
    for h, hn in HN.items():
        z = I[I.handling == h].set_index("R")[met]
        L.append(hn + " & " + " & ".join(f"{z[r]:.3f}" for r in RSPEC) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(O + "tab_imbalance.tex", "w").write("\n".join(L))
C = pd.read_csv(T + "country_training.csv"); M = pd.read_csv(T + "metrics_all.csv")
P = M[(M.set == "I+H+D+S+R+C") & (M.country != "All five")]
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Pooled training (main) versus training a separate model in each country: balanced accuracy of the full model.}", r"\label{tab:countrytrain}", r"\footnotesize",
     r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{llccccc}", r"\toprule", r"Threshold & Model and training & Bulgaria & Hungary & Italy & Poland & Spain \\", r"\midrule"]
for r in RSPEC:
    for m, mn in (("catboost", "CatBoost"), ("logit", "Logistic regression")):
        p = P[(P.R == r) & (P.model == m)].set_index("country").balanced_accuracy
        c = C[(C.R == r) & (C.model == m)].set_index("country").balanced_accuracy
        L.append(f"{r if m=='catboost' else ''} & {mn}, pooled & " + " & ".join(f"{p[k]:.3f}" for k in COUNTRIES) + r" \\")
        L.append(f" & {mn}, country only & " + " & ".join(f"{c[k]:.3f}" for k in COUNTRIES) + r" \\")
    if r != "R3": L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(O + "tab_countrytrain.tex", "w").write("\n".join(L))
U = pd.read_csv(T + "tau80.csv"); U = U[U.country == "All five"]
M2 = M[(M.country == "All five") & (M.R.isin(["R2", "R3"])) & (M.model.isin(["catboost", "logit"]))]
STEPS = ["I", "I+H", "I+H+D", "I+H+D+S", "I+H+D+S+R", "I+H+D+S+R+C"]
SN = ["Income", "+ Household", "+ Dwelling", "+ Heating system", "+ Region", "+ Climate"]
L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Sensitivity to the level of the need lines: 60th (main) versus 80th conditional percentile, balanced accuracy in policy order, five countries together.}", r"\label{tab:tau}", r"\footnotesize",
     r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{lcccccccc}", r"\toprule",
     r" & \multicolumn{4}{c}{R2 need line} & \multicolumn{4}{c}{R3 need line + climate} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r"Information & CatBoost 0.60 & CatBoost 0.80 & Logit 0.60 & Logit 0.80 & CatBoost 0.60 & CatBoost 0.80 & Logit 0.60 & Logit 0.80 \\", r"\midrule"]
for s, sn in zip(STEPS, SN):
    cells = []
    for r in ("R2", "R3"):
        for m in ("catboost", "logit"):
            cells.append(f"{M2[(M2.R==r)&(M2.model==m)&(M2.set==s)].balanced_accuracy.iloc[0]:.3f}")
            cells.append(f"{U[(U.R==r)&(U.model==m)&(U.set==s)].balanced_accuracy.iloc[0]:.3f}")
    L.append(sn + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(O + "tab_tau.tex", "w").write("\n".join(L))
print("ok")
