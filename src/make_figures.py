"""Figures for the paper: cumulative wealth curves for the main comparison
and a Sharpe-ratio bar chart across all covariate ablations."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rets = pd.read_csv("results/portfolio_returns.csv", parse_dates=["date"])
summary = pd.read_csv("results/main_comparison.csv")
os.makedirs("results/figures", exist_ok=True)

LABELS = {
    "A": "Arm A (all 5 covariates)",
    "B": "Arm B (no covariates)",
    "A_changes": "Arm A (covariates as changes)",
}

# ---- Figure 1: wealth curves, main comparison ----
fig, ax = plt.subplots(figsize=(8, 4.5))

for arm, style in [("B", "-"), ("A", "-"), ("A_changes", "--")]:
    r = rets[(rets["arm"] == arm) & (rets["strategy"] == "MV")].sort_values("date")
    wealth = np.concatenate([[1.0], np.cumprod(1 + r["return"].values)])
    dates = [r["date"].iloc[0] - pd.Timedelta(days=30)] + list(r["date"])
    ax.plot(dates, wealth, style, linewidth=1.8, label=LABELS[arm])

ew = rets[(rets["arm"] == "B") & (rets["strategy"] == "EW")].sort_values("date")
wealth_ew = np.concatenate([[1.0], np.cumprod(1 + ew["return"].values)])
dates_ew = [ew["date"].iloc[0] - pd.Timedelta(days=30)] + list(ew["date"])
ax.plot(dates_ew, wealth_ew, ":", color="grey", linewidth=1.8,
        label="Equal-weight baseline")

ax.set_ylabel("Cumulative wealth (start = 1.0)")
ax.set_title("Portfolio growth, 2023-2026 (monthly rebalancing)")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/wealth_curves.png", dpi=200)
print("Saved results/figures/wealth_curves.png")

# ---- Figure 2: Sharpe by ablation ----
mv = summary[summary["strategy"] == "MV"].copy()
order = ["B", "A_wo_US10Y_yield", "A_wo_USD_index", "A_wo_VIX_volatility",
         "A_changes", "A", "A_wo_Gold", "A_wo_Oil_WTI"]
mv = mv.set_index("arm").reindex([a for a in order if a in set(mv["arm"])])

pretty = {"B": "no covariates", "A": "all 5 covariates",
          "A_changes": "changes, not levels"}
names = [pretty.get(a, a.replace("A_wo_", "without ")) for a in mv.index]
colors = ["#1E2761" if a in ("A", "B") else "#9AA6D6" for a in mv.index]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(names, mv["ann_sharpe"], color=colors)
ax.axvline(mv.loc["B", "ann_sharpe"], color="grey", linestyle=":",
           label="Arm B baseline")
ax.set_xlabel("Annualised Sharpe ratio")
ax.set_title("Mean-variance portfolios: covariate ablations")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("results/figures/ablation_sharpe.png", dpi=200)
print("Saved results/figures/ablation_sharpe.png")
