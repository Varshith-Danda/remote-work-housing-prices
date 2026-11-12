"""Analyses added for the final results submission (report/Team10_Results.tex). They are the follow-ups the methodology
document listed, built on the same data, sample and code as before (methods.py):

  rent_event_study      - the event study on rents (ZORI) next to prices on the same ZIP codes: the mortgage-rate check
  flag_free_event_study - the event study without ZIP codes that carry a data-quality flag
  block_bootstrap_coef  - spatial block bootstrap (percentile, basic, symmetric, BCa) for any OLS coefficient
  exposure_specs        - the work-from-home coefficient from no controls to full controls, plus exposure x distance
  cluster_determinants  - what separates the k-means price-path groups (multinomial logit, one feature group at a time)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import methods as M
import viz
import viz_eda

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
SEED = M.SEED


# ----------------------------------------------------------------------------- 1. rents vs prices
def annual_log_rents(gd):
    """ZIP x year matrix of 100 * log(rent) every December 2017-2025, for balanced-sample ZIPs with a rent in every one of them."""
    z = pd.read_parquet(CLEAN / "zori_analysis_panel.parquet", columns=["zip", "date", "log_zori"])
    z = z[(z["date"].dt.month == 12) & (z["date"].dt.year.isin(M.YEARS))]
    m = z.assign(year=z["date"].dt.year).pivot(index="zip", columns="year", values="log_zori") * 100
    keep = gd.loc[gd["balanced"], "zip"]
    return m.loc[m.index.intersection(keep)].dropna(how="any")


def rent_event_study(gd, B=1000):
    """The event study on rents, and on prices for exactly the same ZIP codes, so the two can be compared."""
    R = annual_log_rents(gd)
    P = M.annual_log_values(gd).loc[R.index]
    tabs, sums = [], []
    # price minus rent (in logs) is the price-to-rent ratio: its coefficient is the price coefficient minus the rent
    # coefficient, so its interval is an interval for the difference between the two gradients
    for name, L in [("price", P), ("rent", R), ("price-to-rent ratio", P - R)]:
        t, s_ = M.event_study(gd, B=B, L=L)
        tabs.append(t.assign(outcome=name))
        sums.append(s_.assign(outcome=name))
    return pd.concat(tabs, ignore_index=True), pd.concat(sums, ignore_index=True)


# ----------------------------------------------------------------------------- 2. without flagged ZIP codes
def flag_free_event_study(gd, B=1000):
    """The price event study on balanced ZIP codes with none of the four data-quality flags used in the EDA."""
    flags = pd.read_csv(CLEAN / "zip_flags.csv", dtype={"zip": str})
    bad = set(flags.loc[flags[viz_eda.CORE_FLAGS].any(axis=1), "zip"])
    L = M.annual_log_values(gd)
    L = L.loc[[z for z in L.index if z not in bad]]
    return M.event_study(gd, B=B, L=L)


# ----------------------------------------------------------------------------- 3. exposure specifications
def block_bootstrap_coef(x, ycol, cols, coef, B=1000):
    """OLS of ycol on cols plus metro fixed effects; spatial block bootstrap (grid cells redrawn within each metro, as in
    methods.event_study). Returns the estimate, bootstrap SE and four 95% intervals: percentile, basic, symmetric, BCa
    (acceleration from a jackknife that leaves out one grid cell at a time)."""
    from scipy import stats
    rng = np.random.default_rng(SEED)
    blocks = M._blocks(x).to_numpy()
    metro = x["metro_short"].to_numpy()
    cells = {m: pd.Series(np.arange(len(x))[metro == m]).groupby(blocks[metro == m]).apply(list).tolist() for m in sorted(set(metro))}
    X = np.c_[np.ones(len(x)), x[cols].to_numpy(float), pd.get_dummies(x["metro_short"], drop_first=True).astype(float).to_numpy()]
    y = x[ycol].to_numpy(float)
    j = 1 + cols.index(coef)
    fit = lambda X_, y_: np.linalg.lstsq(X_, y_, rcond=None)[0][j]
    est = fit(X, y)
    d = np.empty(B)
    for b in range(B):
        idx = []
        for m, cl in cells.items():
            pick = rng.integers(0, len(cl), len(cl))
            idx.extend(i for k in pick for i in cl[k])
        idx = np.array(idx)
        d[b] = fit(X[idx], y[idx])
    se = d.std(ddof=1)
    pct = np.percentile(d, [2.5, 97.5])
    jk = np.array([fit(X[blocks != bk], y[blocks != bk]) for bk in np.unique(blocks)])
    dev = jk.mean() - jk
    accel = (dev ** 3).sum() / (6 * ((dev ** 2).sum()) ** 1.5)
    z0 = stats.norm.ppf((d < est).mean())
    za = stats.norm.ppf([0.025, 0.975])
    bca = np.percentile(d, 100 * stats.norm.cdf(z0 + (z0 + za) / (1 - accel * (z0 + za))))
    return {"estimate": est, "boot_se": se, "pct_lo": pct[0], "pct_hi": pct[1], "basic_lo": 2 * est - pct[1],
            "basic_hi": 2 * est - pct[0], "sym_lo": est - 1.96 * se, "sym_hi": est + 1.96 * se, "bca_lo": bca[0], "bca_hi": bca[1]}


def exposure_specs(xd, B=1000):
    """The work-from-home coefficient (per 5 points) as controls are added, and the exposure x distance interaction."""
    x = M._exposure_frame(xd)
    x["wfh_x_dist"] = x["wfh5"] * x["dist10"]
    specs = [("No controls", ["wfh5"]),
             ("+ distance", ["wfh5", "dist10"]),
             ("+ income and home value", ["wfh5", "dist10", "log_income_z", "log_value_z"]),
             ("+ housing type", M.EXPOSURE_BASE),
             ("+ pre-2020 growth", M.EXPOSURE_BASE + ["pre"])]
    rows = []
    for out, lab in M.EXPOSURE_OUTCOMES:
        for name, cols in specs:
            rows.append({"outcome": lab, "spec": name, "term": "wfh5", **block_bootstrap_coef(x, out, cols, "wfh5", B)})
        cols = M.EXPOSURE_BASE + ["wfh_x_dist"]
        rows.append({"outcome": lab, "spec": "+ housing type, with WFH x distance", "term": "wfh_x_dist",
                     **block_bootstrap_coef(x, out, cols, "wfh_x_dist", B)})
    return pd.DataFrame(rows).assign(n=len(x))


# ----------------------------------------------------------------------------- 4. what separates the price-path groups
def cluster_determinants(gd, xd):
    """k-means groups (k = 4, the EDA's choice) predicted by one feature group at a time with a multinomial logit.
    McFadden pseudo R-squared = 1 - LL(model) / LL(group shares only). Also accuracy against always guessing the largest group."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    panel = pd.read_parquet(CLEAN / "zhvi_analysis_panel.parquet", columns=["zip", "date", "index_dec2019"])
    P = viz_eda.path_matrix(panel, gd)
    _, labels, k = viz_eda.cluster_paths(P)
    x = xd.set_index("zip").join(labels, how="inner").dropna(subset=["log_income", "log_value19", "sfd_pct", "mf20_pct"])
    y = x["cluster"].to_numpy()
    groups = {"Metro": pd.get_dummies(x["metro_short"], drop_first=True).astype(float),
              "Distance": x[["dist_nearest_mi"]],
              "Work-from-home share": x[["wfh_pct"]],
              "Housing type": x[["sfd_pct", "mf20_pct"]],
              "Home value, Dec 2019": x[["log_value19"]],
              "Income": x[["log_income"]]}
    p0 = np.bincount(y) / len(y)
    ll0 = np.log(p0[y]).sum()
    rows = []
    for name, F in list(groups.items()) + [("All of the above", pd.concat(groups.values(), axis=1))]:
        Z = StandardScaler().fit_transform(F.to_numpy(float))
        m = LogisticRegression(C=1e6, max_iter=5000).fit(Z, y)
        pr = m.predict_proba(Z)
        ll = np.log(pr[np.arange(len(y)), np.searchsorted(m.classes_, y)]).sum()
        rows.append({"features": name, "pseudo_r2": 1 - ll / ll0, "accuracy": (m.predict(Z) == y).mean()})
    out = pd.DataFrame(rows)
    out["baseline_accuracy"] = p0.max()
    out["n"] = len(y)
    out["k"] = k
    return out
