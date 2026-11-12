"""LaTeX tables for the methodology document, built from outputs/tables/m_*.csv so numbers cannot be mistyped.
Run from the repo root after src/make_method_results.py:  .venv/bin/python report/make_method_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
OUT = ROOT / "report" / "tables"
MINUS = "$-$"
STACK = lambda a, b: r"\shortstack[r]{" + a + r"\\" + b + "}"


def num(v, dec=2, plus=False):
    if pd.isna(v):
        return "--"
    s = f"{abs(v):.{dec}f}"
    if float(s) == 0:
        return s
    return (MINUS if v < 0 else ("+" if plus else "")) + s


def ci(lo, hi, dec=2):
    return f"[{num(lo, dec, True)}, {num(hi, dec, True)}]"


def write(name, header, rows, colspec, pad=4):
    lines = [r"\footnotesize\setlength{\tabcolsep}{" + f"{pad}pt" + "}", r"\begin{tabular}{" + colspec + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n")


# ---- event study
e = pd.read_csv(T / "m_event_summary.csv")
rows = []
for _, r in e.iterrows():
    nm = "All eight, pooled" if r["scope"].startswith("All") else r["scope"]
    rows.append([nm, f"{int(r['n'])}", num(r["pre-trend 2017"], 2, True), num(r["2018"], 2, True), num(r["2022"], 2, True), ci(r["2022 lo"], r["2022 hi"], 1),
                 num(r["2025"], 2, True), ci(r["2025 lo"], r["2025 hi"], 1), f"{r['boot/naive width (2022)']:.1f}"])
write("t_m_event.tex", ["Scope", "ZIPs", STACK("2017", "(pre-trend)"), STACK("2018", "(pre-trend)"), "2022", STACK("95\\% block", "interval"), "2025",
                        STACK("95\\% block", "interval"), STACK("Block/ordinary", "width")], rows, "lrrrrlrll")

# ---- exposure model: coefficient and four interval types
m = pd.read_csv(T / "m_mixed_model.csv")
rows = []
for out in ["2020-22", "2023-25"]:
    for ctrl in ["controls", "controls + pre-2020 growth"]:
        q = m[(m["outcome"] == out) & (m["controls"] == ctrl)].set_index("model")
        h, c, mx, bs = q.loc["OLS (HC1)"], q.loc["OLS, clustered by county"], q.loc["Mixed (county random intercept)"], q.loc["OLS, spatial block bootstrap"]
        rows.append([out.replace("-", "--"), "Controls" if ctrl == "controls" else "+ pre-2020", num(h["wfh_per5pp"], 2, True),
                     num(bs["se"], 3), ci(h["lo"], h["hi"]), ci(c["lo"], c["hi"]), ci(bs["lo"], bs["hi"]),
                     f"{num(mx['wfh_per5pp'], 2, True)} {ci(mx['lo'], mx['hi'])}"])
write("t_m_mixed.tex", ["Outcome", "Controls", STACK("OLS", "coefficient"), STACK("Bootstrap", "SE"), STACK("Plain OLS", "(HC1)"),
                        STACK("OLS, clustered", "by county"), STACK("OLS, spatial block", "bootstrap (BCa)"),
                        STACK("Mixed model", "(county effect)")], rows, "llrrrrrr", pad=1.8)

# ---- the four bootstrap intervals side by side
rows = []
for out in ["2020-22", "2023-25"]:
    for ctrl in ["controls", "controls + pre-2020 growth"]:
        bs = m[(m["outcome"] == out) & (m["controls"] == ctrl) & (m["model"] == "OLS, spatial block bootstrap")].iloc[0]
        ivs = {"percentile": (bs["pct_lo"], bs["pct_hi"]), "basic": (bs["basic_lo"], bs["basic_hi"]),
               "symmetric": (bs["sym_lo"], bs["sym_hi"]), "BCa": (bs["lo"], bs["hi"])}
        excl = [k for k, (lo, hi) in ivs.items() if lo > 0 or hi < 0]
        rows.append([out.replace("-", "--"), "Controls" if ctrl == "controls" else "+ pre-2020"]
                    + [ci(*ivs["percentile"]), ci(*ivs["basic"], dec=3), ci(*ivs["symmetric"]), ci(*ivs["BCa"])] + [", ".join({"percentile": "pct.", "basic": "basic", "symmetric": "sym.", "BCa": "BCa"}[k] for k in excl)])
write("t_m_boot.tex", ["Outcome", "Controls", "Percentile", "Basic", "Symmetric", "BCa", "Excludes zero"], rows, "lllllll", pad=3)

# ---- Moran's I
mo = pd.read_csv(T / "m_moran.csv")
rows = [[r["period"].replace("2020-22", "2020--22").replace("2023-25", "2023--25"), r["variable"].capitalize(), num(r["morans_I"], 2),
         "$<$0.001" if r["p_value"] < 0.001 else num(r["p_value"], 3)] for _, r in mo.iterrows()]
write("t_m_moran.tex", ["Period", "Variable", "Moran's $I$", "$p$-value"], rows, "llrr")

# ---- cross-validation scores (feature set D)
try:
    s = pd.read_csv(T / "m_cv_scores.csv")
    rows = []
    for tg, lab in [("ann_cum_2019_2022_dm", "2020--22"), ("ann_cum_2022_2025_dm", "2023--25")]:
        q = s[(s["target"] == tg) & s["features"].str.startswith("D")]
        rows.append(r"\multicolumn{5}{l}{\textit{Growth in " + lab + r" (predict-the-average RMSE " + f"{q['baseline_RMSE'].iloc[0]:.2f}" + r")}} \\")
        for _, r in q.iterrows():
            rows.append([r["model"], num(r["RMSE"], 2), num(r["MAE"], 2), num(r["R2_leave_one_metro_out"], 2),
                         num(r["R2_random_5fold"], 2), ""][:5])
    write("t_m_cv.tex", ["Model", "RMSE", "MAE", STACK("$R^2$, leave", "one metro out"), STACK("$R^2$, random", "5-fold")],
          rows, "lrrrr")
    # feature-set ablation, leave one metro out
    rows = []
    for tg, lab in [("ann_cum_2019_2022_dm", "2020--22"), ("ann_cum_2022_2025_dm", "2023--25")]:
        for mod in ["Ridge", "Lasso", "Random forest", "Gradient boosting"]:
            q = s[(s["target"] == tg) & (s["model"] == mod)].sort_values("features")
            rows.append([lab, mod] + [num(v, 3) for v in q["R2_leave_one_metro_out"]])
    write("t_m_ablation.tex", ["Outcome", "Model", STACK("A: place,", "housing"), STACK("B: + income,", "jobs"),
                               STACK("C: + WFH", "share"), STACK("D: + pre-2020", "growth")], rows, "llrrrr")
except FileNotFoundError:
    pass

# ---- clustering methods
c = pd.read_csv(T / "m_cluster_methods.csv")
a = pd.read_csv(T / "m_cluster_ari.csv").set_index("k")
rows = []
for k in [2, 3, 4, 5]:
    q = c[c["k"] == k].set_index("method")
    rows.append([str(k), num(q.loc["k-means", "silhouette"], 2), num(q.loc["Ward hierarchical", "silhouette"], 2),
                 num(q.loc["Gaussian mixture", "silhouette"], 2), num(a.loc[k, "k-means vs Ward"], 2),
                 num(a.loc[k, "k-means vs mixture"], 2), num(a.loc[k, "Ward vs mixture"], 2)])
write("t_m_clusters.tex", ["$k$", STACK("Silhouette,", "k-means"), STACK("Silhouette,", "Ward"), STACK("Silhouette,", "mixture"),
                           STACK("Agreement (ARI)", "k-means/Ward"), STACK("Agreement (ARI)", "k-means/mix."),
                           STACK("Agreement (ARI)", "Ward/mix.")], rows, "rrrrrrr")
print("tables written")
