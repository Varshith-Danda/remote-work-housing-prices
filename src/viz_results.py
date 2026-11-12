"""Figures for the final results submission. Same style, fonts and palette as the EDA and methodology figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from viz import BLUE_700, C1, C2, FIG_W, INK, METRO_ORDER, fs

GREY = "#898781"
SCOPES = ["All eight (within-metro)"] + METRO_ORDER


def _label(s):
    return "All eight, pooled" if s.startswith("All") else s


def summary_figure(full: pd.DataFrame, flagfree: pd.DataFrame):
    """One row per metro: the gradient coefficient in 2022 and 2025 (all balanced ZIPs) and in 2025 without flagged ZIPs,
    each with its spatial block-bootstrap 95% interval."""
    fig, ax = plt.subplots(figsize=(FIG_W, 5.4))
    f, g = full.set_index("scope"), flagfree.set_index("scope")
    series = [("2022, all ZIPs", f, "2022", -0.24, dict(facecolors=INK["surface"], edgecolors=C1, linewidths=1.8), C1),
              ("2025, all ZIPs", f, "2025", 0.0, dict(color=BLUE_700, edgecolors=INK["surface"], linewidths=1.0), BLUE_700),
              ("2025, without flagged ZIPs", g, "2025", 0.24, dict(color=C2, marker="D", edgecolors=INK["surface"], linewidths=1.0), C2)]
    for i, s in enumerate(SCOPES):
        y = len(SCOPES) - 1 - i
        if i == 0:
            ax.axhspan(y - 0.45, y + 0.45, color="#f1efe6", zorder=0)
        for lab, t, col, off, kw, lc in series:
            ax.hlines(y + off, t.loc[s, f"{col} lo"], t.loc[s, f"{col} hi"], color=lc, lw=1.6, zorder=2)
            ax.scatter([t.loc[s, col]], [y + off], s=46, zorder=3, label=lab if i == 0 else None, **kw)
    ax.axvline(0, color=INK["axis"], lw=1.1, zorder=1)
    ax.set_yticks(range(len(SCOPES)))
    ax.set_yticklabels([_label(s) for s in SCOPES[::-1]])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Gradient coefficient: extra % gained since Dec 2019 by a ZIP code 10 miles farther out")
    ax.legend(loc="lower right", frameon=False, fontsize=fs(9.8))
    viz._heading(ax, "Where the price gradient ended up, by metro",
                 "Lines are spatial block-bootstrap 95% intervals. Right of zero: farther ZIP codes gained more.")
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.17, right=0.98)
    return fig


def rent_figure(tab: pd.DataFrame):
    """Left: pooled within-metro gradient by year for prices and rents on the same ZIP codes. Right: the price-to-rent ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.9))
    t = tab[tab["scope"].str.startswith("All")]
    n = int(t["n"].iloc[0])
    for ax, items, title, sub in [
            (axes[0], [("price", C1, "Prices"), ("rent", C2, "Rents")], "Prices and rents", f"Same {n} ZIP codes, pooled within metro"),
            (axes[1], [("price-to-rent ratio", BLUE_700, "Price-to-rent ratio")], "Price-to-rent ratio",
             "Price coefficient minus rent coefficient")]:
        ax.axhline(0, color=INK["axis"], lw=1.0)
        ax.axvline(2019, color=INK["axis"], lw=1.0, ls=(0, (3, 3)))
        ax.axvspan(2019.5, 2022.5, color="#f1efe6", zorder=0)
        for out, col, lab in items:
            q = t[t["outcome"] == out].sort_values("year")
            ax.fill_between(q["year"], q["boot_lo"], q["boot_hi"], color=col, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(q["year"], q["beta"], color=col, lw=2.0, marker="o", markersize=5, markerfacecolor=col,
                    markeredgecolor=INK["surface"], markeredgewidth=1.0, label=lab, zorder=4)
        ax.set_xticks([2017, 2019, 2021, 2023, 2025])
        ax.set_ylabel("% per 10 mi out, vs Dec 2019")
        viz._heading(ax, title, sub)
        if len(items) > 1:
            ax.legend(loc="upper left", frameon=False, fontsize=fs(9.8))
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.12, left=0.09, right=0.985)
    return fig


def exposure_figure(ex: pd.DataFrame):
    """The work-from-home coefficient as controls are added. Thick line = BCa interval (our main interval);
    thin grey line = symmetric interval from the same bootstrap, to show where the verdict depends on the method."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.9), sharey=True)
    specs = ["No controls", "+ distance", "+ income and home value", "+ housing type", "+ pre-2020 growth"]
    for ax, (out, title) in zip(axes, [("2020-22", "Growth in 2020–22"), ("2023-25", "Growth in 2023–25")]):
        q = ex[(ex["outcome"] == out) & (ex["term"] == "wfh5")].set_index("spec").loc[specs]
        y = np.arange(len(specs))[::-1]
        ax.axvline(0, color=INK["axis"], lw=1.1)
        ax.hlines(y - 0.16, q["sym_lo"], q["sym_hi"], color=GREY, lw=1.0, zorder=2, label="symmetric interval")
        ax.hlines(y, q["bca_lo"], q["bca_hi"], color=C1, lw=2.4, zorder=3, label="BCa interval")
        ax.scatter(q["estimate"], y, s=48, color=BLUE_700, edgecolors=INK["surface"], linewidths=1.0, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(specs)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        viz._heading(ax, title, "Controls added from top to bottom")
    fig.text(0.62, 0.135, "Coefficient: points of annual growth per +5 points of work-from-home share", ha="center",
             fontsize=fs(11), color=INK["secondary"])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", ncol=2, frameon=False, fontsize=fs(9.5))
    fig.subplots_adjust(wspace=0.14, top=0.84, bottom=0.24, left=0.25, right=0.985)
    return fig


def cluster_determinants_figure(cd: pd.DataFrame):
    """How much of the k-means group membership each feature group explains on its own (McFadden pseudo R-squared)."""
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    q = cd[cd["features"] != "All of the above"].sort_values("pseudo_r2")
    y = np.arange(len(q))
    ax.barh(y, q["pseudo_r2"], color=C1, height=0.62, zorder=3)
    for yy, v in zip(y, q["pseudo_r2"]):
        ax.text(v + 0.006, yy, f"{v:.2f}" if v >= 0.01 else "<0.01", va="center", fontsize=fs(9.5), color=INK["secondary"])
    ax.set_yticks(y)
    ax.set_yticklabels(q["features"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(0.45, q["pseudo_r2"].max() * 1.15))
    ax.set_xlabel("Pseudo R² of a multinomial logit using only that feature group")
    viz._heading(ax, "What separates the four price-path groups?", f"Each bar is a separate model; {int(cd['n'].iloc[0]):,} ZIP codes")
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.22, right=0.97)
    return fig
