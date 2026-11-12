"""Check that every number quoted in report/Team10_Results.tex matches the saved results.

    .venv/bin/python src/check_results_numbers.py

New results come from outputs/tables/r_*.csv (src/make_results.py); results carried over from the methodology come from
outputs/tables/m_*.csv; numbers quoted from the EDA report are checked against report/Team10_EDA.tex. Exits with an error
if any check fails.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
norm = lambda s: re.sub(r"\s+", " ", s.replace("$-$", "-").replace("$+$", "+").replace("--", "~"))
tex = norm((ROOT / "report" / "Team10_Results.tex").read_text())
eda = norm((ROOT / "report" / "Team10_EDA.tex").read_text())
ok = bad = 0


def check(label, phrase, where=None):
    global ok, bad
    phrase = re.sub(r"\s+", " ", phrase)
    if phrase in (tex if where is None else where):
        ok += 1
    else:
        bad += 1
        print(f"FAILED  {label}: not found -> {phrase!r}")


def claim(label, cond):
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
        print(f"FAILED  {label}")


sg = lambda v, d=2: ("-" if v < 0 and float(f"{abs(v):.{d}f}") != 0 else "+") + f"{abs(v):.{d}f}"
un = lambda v, d=2: f"{v:.{d}f}"
iv = lambda lo, hi: f"{sg(lo)} to {sg(hi)}"

# ---------------------------------------------------------------- event study, all balanced ZIPs (methodology)
e = pd.read_csv(T / "m_event_summary.csv").set_index("scope")
P = e.loc["All eight (within-metro)"]
check("pooled 2022", f"the gradient coefficient is {sg(P['2022'])} in 2022 (interval {iv(P['2022 lo'], P['2022 hi'])})")
check("pooled 2025", f"and {sg(P['2025'])} in 2025 ({un(P['2025 lo'])} to {un(P['2025 hi'])})")
check("pooled change", f"the change from 2022 to 2025 is {sg(P['2025 minus 2022'])} ({un(P['diff lo'])} to {un(P['diff hi'])})")
check("pooled change in table", f"(2022 to 2025 change {sg(P['2025 minus 2022'])})")
A = e.loc["Atlanta"]
check("Atlanta", f"from {sg(A['2022'])} in 2022 to {sg(A['2025'])} in 2025 (interval {un(A['2025 lo'])} to {un(A['2025 hi'])})")
Tm = e.loc["Tampa"]
check("Tampa 2022", f"({sg(Tm['2022'])}, {un(Tm['2022 lo'])} to {un(Tm['2022 hi'])})")
claim("Tampa 2025 not clear", Tm["2025 lo"] < 0 < Tm["2025 hi"])
check("Nashville pre-trend", f"its 2017 coefficient is {sg(e.loc['Nashville', 'pre-trend 2017'])}")
claim("Austin interval wide, includes zero", e.loc["Austin", "2025 lo"] < 0 < e.loc["Austin", "2025 hi"] and e.loc["Austin", "2025"] > e.loc["Austin", "2022"])
claim("Phoenix and Las Vegas below zero in 2025", e.loc["Phoenix", "2025"] < 0 and e.loc["Las Vegas", "2025"] < 0)
ev = pd.read_csv(T / "m_event_study.csv").set_index(["scope", "year"])
claim("Dallas spike 2021, Charlotte spike 2020, both near zero by 2025",
      ev.loc[("Dallas", 2021), "beta"] > 1.5 and ev.loc[("Charlotte", 2020), "beta"] > 1.5
      and abs(e.loc["Dallas", "2025"]) < 0.6 and abs(e.loc["Charlotte", "2025"]) < 0.6)
excl = lambda r: (r.boot_lo > 0 or r.boot_hi < 0, r.basic_lo > 0 or r.basic_hi < 0,
                  r.beta - 1.96 * r.boot_se > 0 or r.beta + 1.96 * r.boot_se < 0)
clear25 = [m for m in e.index if not m.startswith("All") and all(excl(ev.loc[(m, 2025)]))]
claim("Atlanta only metro clear in 2025 under every interval type", clear25 == ["Atlanta"])
w = e["boot/naive width (2022)"]
claim("ordinary intervals up to three times too narrow", round(w.max(), 1) == 3.0)

# ---------------------------------------------------------------- without flagged ZIP codes
f = pd.read_csv(T / "r_flagfree_event_summary.csv").set_index("scope")
F = f.loc["All eight (within-metro)"]
check("flag-free n", f"leaves {int(F['n'])} ZIP codes")
check("flag-free Atlanta", f"Atlanta reaches {sg(f.loc['Atlanta', '2025'])} in 2025 ({un(f.loc['Atlanta', '2025 lo'])} to {un(f.loc['Atlanta', '2025 hi'])})")
check("flag-free pooled", f"the pooled coefficient is {sg(F['2025'])} ({un(F['2025 lo'])} to {un(F['2025 hi'])})")
check("flag-free Phoenix", f"({sg(f.loc['Phoenix', '2025'])}, interval {iv(f.loc['Phoenix', '2025 lo'], f.loc['Phoenix', '2025 hi'])})")
claim("Phoenix decline disappears", f.loc["Phoenix", "2025 lo"] < 0 < f.loc["Phoenix", "2025 hi"])
check("flag-free Las Vegas", f"({sg(f.loc['Las Vegas', '2025'])}, {sg(f.loc['Las Vegas', '2025 lo'])} to {sg(f.loc['Las Vegas', '2025 hi'])})")
claim("Las Vegas clearly negative", f.loc["Las Vegas", "2025 hi"] < 0)
claim("Atlanta stronger without flags", f.loc["Atlanta", "2025"] > A["2025"] and F["2025"] > P["2025"])

# ---------------------------------------------------------------- rents and the price-to-rent ratio
rs = pd.read_csv(T / "r_rent_event_summary.csv")
rt = pd.read_csv(T / "r_rent_event_study.csv")
pool = lambda o: rs[(rs.outcome == o) & rs.scope.str.startswith("All")].iloc[0]
yr = lambda o, y: rt[(rt.outcome == o) & rt.scope.str.startswith("All") & (rt.year == y)].iloc[0]
n = int(pool("rent")["n"])
check("rent n", f"Only {n} ZIP codes have a rent in every December")
sys.path.insert(0, str(ROOT / "src"))
import methods as M  # noqa: E402
import results as R  # noqa: E402
gd, _ = M.load()
rz = gd.set_index("zip").loc[R.annual_log_rents(gd).index]
n30, all30 = int((rz.dist_nearest_mi >= 30).sum()), int((gd.loc[gd.balanced, "dist_nearest_mi"] >= 30).sum())
check("outer ring counts", f"({n30} of them are 30 or more miles out, against {all30} in the full sample)")
r20, p20 = yr("rent", 2020), yr("price", 2020)
check("rent 2020", f"the rent gradient rose ({sg(r20.beta)}, {un(r20.boot_lo)} to {un(r20.boot_hi)})")
check("price 2020", f"the price gradient did not ({sg(p20.beta)}, {iv(p20.boot_lo, p20.boot_hi)})")
claim("rents moved first", r20.boot_lo > 0 and p20.boot_lo < 0 < p20.boot_hi)
ratio = pool("price-to-rent ratio")
check("ratio 2022", f"rose with distance ({sg(ratio['2022'])}, {un(ratio['2022 lo'])} to {un(ratio['2022 hi'])})")
check("ratio change", f"from 2022 to 2025 is {sg(ratio['2025 minus 2022'])} ({iv(ratio['diff lo'], ratio['diff hi'])})")
claim("ratio change includes zero", ratio["diff lo"] < 0 < ratio["diff hi"])
claim("rent gradient did not fall back", pool("rent")["2025"] >= pool("rent")["2022"])

# ---------------------------------------------------------------- work-from-home exposure
x = pd.read_csv(T / "r_exposure_specs.csv")
s_ = lambda o, sp, col: x[(x.outcome == o) & (x.spec == sp)][col].iloc[0]
check("no controls", f"in 2020~22 ({sg(s_('2020-22', 'No controls', 'estimate'))} per 5 points of work-from-home share) and faster in 2023~25 ({sg(s_('2023-25', 'No controls', 'estimate'))})")
check("income flip", f"to {sg(s_('2020-22', '+ income and home value', 'estimate'))} in 2020~22 and {sg(s_('2023-25', '+ income and home value', 'estimate'))} in 2023~25")
claim("distance changes almost nothing", all(abs(s_(o, "+ distance", "estimate") - s_(o, "No controls", "estimate")) < 0.05 for o in ("2020-22", "2023-25")))
claim("signs flip at income and value", all(s_(o, "No controls", "estimate") * s_(o, "+ income and home value", "estimate") < 0 for o in ("2020-22", "2023-25")))
claim("housing and pre-trend change little", all(abs(s_(o, sp, "estimate") - s_(o, "+ income and home value", "estimate")) < 0.1
                                                   for o in ("2020-22", "2023-25") for sp in ("+ housing type", "+ pre-2020 growth")))
ctrl = ["+ income and home value", "+ housing type", "+ pre-2020 growth"]
claim("BCa excludes zero in every controlled spec", all(s_(o, sp, "bca_lo") * s_(o, sp, "bca_hi") > 0 for o in ("2020-22", "2023-25") for sp in ctrl))
claim("symmetric includes zero for 2020-22 once housing type is added",
      all(s_("2020-22", sp, "sym_lo") < 0 < s_("2020-22", sp, "sym_hi") for sp in ctrl[1:]) and s_("2020-22", ctrl[0], "sym_lo") > 0)
check("interaction", f"the exposure-by-distance interaction is {sg(s_('2020-22', '+ housing type, with WFH x distance', 'estimate'))} for 2020~22 "
      f"(interval {iv(s_('2020-22', '+ housing type, with WFH x distance', 'bca_lo'), s_('2020-22', '+ housing type, with WFH x distance', 'bca_hi'))}) and "
      f"{sg(s_('2023-25', '+ housing type, with WFH x distance', 'estimate'))} for 2023~25 "
      f"({iv(s_('2023-25', '+ housing type, with WFH x distance', 'bca_lo'), s_('2023-25', '+ housing type, with WFH x distance', 'bca_hi'))})")
check("sample", f"{int(x['n'].iloc[0]):,} ZIP codes.}}")
check("EDA income corr", "+0.66 with income", eda)
check("EDA value corr", "+0.74 with home value", eda)
check("quoted corr", "rank correlation of +0.66 with income and +0.74 with home value")
b = s_("2020-22", "+ housing type", "estimate")
check("main controlled estimate", f"Our main controlled estimate (with income, home value and housing type), {sg(b)} points of annual growth per 5 points")
check("per point", f"is about {b / 5:.2f} points a year per point, roughly {3 * b / 5:.1f} percent over three years")
cv = pd.read_csv(T / "m_cv_scores.csv")
claim("prediction 10 to 30 percent", 0.09 < cv.R2_leave_one_metro_out.min() and cv.R2_leave_one_metro_out.max() < 0.30)
piv = cv.pivot_table(index=["target", "model"], columns=cv.features.str[0], values="R2_leave_one_metro_out")
claim("WFH changes accuracy by no more than 0.02", (piv["C"] - piv["B"]).abs().max() <= 0.02)

# ---------------------------------------------------------------- price-path groups
prof = pd.read_csv(T / "cluster_profile.csv").set_index("cluster")
comp = pd.read_csv(T / "cluster_metro_composition.csv", index_col=0)
check("group sizes", f"Group~1 ({prof.loc[1, 'zips']} ZIP codes)")
check("group 2", f"Group~2 ({prof.loc[2, 'zips']}, about half of them in Dallas)")
check("group 3", f"Group~3 ({prof.loc[3, 'zips']}, three-quarters of them in Austin)")
check("group 4", f"Group~4 ({prof.loc[4, 'zips']}) is the dense urban core of Atlanta and Dallas, a median of {prof.loc[4, 'median_distance_mi']:.1f} miles")
check("group 4 detached", f"with only {100 * prof.loc[4, 'median_sf_detached']:.0f}\\% detached houses")
row = lambda g: comp.iloc[g - 1]
claim("group compositions", round(row(2)["Dallas"]) == 49 and round(row(3)["Austin"]) == 74 and row(4)["Atlanta"] + row(4)["Dallas"] >= 70)
import viz_eda  # noqa: E402
panel = pd.read_parquet(ROOT / "data" / "clean" / "zhvi_analysis_panel.parquet", columns=["zip", "date", "index_dec2019"])
Pm = viz_eda.path_matrix(panel, gd)
_, lab, _ = viz_eda.cluster_paths(Pm)
mean_path = Pm.groupby(lab.loc[Pm.index].to_numpy() + 1).mean()           # groups numbered 1-4 as in the figure
peak, end = mean_path.max(axis=1), mean_path.iloc[:, -1]
fall = peak - end
claim("group 1 ends highest (rose most, held gains)", end.idxmax() == 1)
claim("group 3 has the highest peak and the largest fall", peak.idxmax() == 3 and fall.idxmax() == 3)
claim("group 4 gained least", end.idxmin() == 4 and peak.idxmin() == 4)
check("EDA silhouette", "(0.36 at four", eda)
check("EDA PC1", "71\\%", eda)
check("EDA wedges", "five of the eight metros the highest-value ZIP codes are concentrated on one side of the center ($p<0.01$ in each)", eda)
check("quoted wedges", "five of the eight metros the highest-value ZIP codes are concentrated on one side of the center ($p<0.01$ in each")
cd = pd.read_csv(T / "r_cluster_determinants.csv").set_index("features")
pr = cd["pseudo_r2"]
check("metro pseudo R2", f"(pseudo $R^2$ {pr['Metro']:.2f}; it alone classifies {100 * cd.loc['Metro', 'accuracy']:.0f}\\% of ZIP codes correctly, against {100 * cd['baseline_accuracy'].iloc[0]:.0f}\\%")
check("others", f"Housing type comes next ({pr['Housing type']:.2f}), then distance ({pr['Distance']:.2f}), home value ({pr['Home value, Dec 2019']:.2f}), "
      f"work-from-home share ({pr['Work-from-home share']:.2f}) and income (under 0.01)")
claim("income under 0.01", pr["Income"] < 0.01)
claim("ranking", list(pr.drop("All of the above").sort_values(ascending=False).index) ==
      ["Metro", "Housing type", "Distance", "Home value, Dec 2019", "Work-from-home share", "Income"])
check("determinants n", f"{int(cd['n'].iloc[0]):,} ZIP codes with complete data")

# ---------------------------------------------------------------- other counts quoted
check("balanced sample", "The main sample is the 1,082 ZIP codes")
flags = pd.read_csv(ROOT / "data" / "clean" / "zip_flags.csv")
check("unreliable WFH count", f"{int(flags.wfh_unreliable.sum())} ZIP codes have unreliable work-from-home estimates")
check("rent count in challenges", f"rents cover only {n} ZIP codes over the full period")

# ---------------------------------------------------------------- consistency with the earlier reports
xr = pd.read_csv(T / "r_exposure_specs_reliable.csv")
r_ = lambda o, sp, col: xr[(xr.outcome == o) & (xr.spec == sp)][col].iloc[0]
check("reliable count", f"Dropping the {int(x['n'].iloc[0]) - int(xr['n'].iloc[0])} ZIP codes with an unreliable work-from-home estimate")
check("reliable estimates", f"({sg(r_('2020-22', '+ housing type', 'estimate'))} for 2020~22 and {sg(r_('2023-25', '+ housing type', 'estimate'))} for 2023~25 with housing type)")
claim("reliable: sign still flips", all(r_(o, "No controls", "estimate") * r_(o, "+ income and home value", "estimate") < 0 for o in ("2020-22", "2023-25")))
claim("reliable: estimates grow slightly", abs(r_("2020-22", "+ housing type", "estimate")) > abs(s_("2020-22", "+ housing type", "estimate"))
      and abs(r_("2023-25", "+ housing type", "estimate")) > abs(s_("2023-25", "+ housing type", "estimate")))
claim("reliable: 2020-22 symmetric still includes zero", r_("2020-22", "+ housing type", "sym_lo") < 0 < r_("2020-22", "+ housing type", "sym_hi"))
check("EDA rent n", "On the 478 ZIP codes with rents at both ends", eda)
check("EDA rent slopes", "growth no longer rose with distance (-0.07) while rent growth still did (+0.33", eda)
check("quoted EDA rent contrast", "on 478 ZIP codes with rents at both ends")
check("quoted EDA slopes", "still rose with distance (+0.33) while price growth no longer did (-0.07)")
claim("price, rent and ratio changes after 2022 all include zero",
      all(pool(o)["diff lo"] < 0 < pool(o)["diff hi"] for o in ("price", "rent", "price-to-rent ratio")))
check("EDA no delayed group", "no delayed-appreciation group", eda)
check("EDA resampling", "not always to resampling", eda)

print(f"PASSED {ok}, FAILED {bad}")
sys.exit(1 if bad else 0)
