"""Weight analysis: does covariate information change how the optimizer allocates?

Computes, per arm, the average weight held in each ETF and three concentration
measures. Feeds Section 6.5 of the paper.
"""

import numpy as np
import pandas as pd

w = pd.read_csv("results/weights.csv", parse_dates=["date"])
mv = w[w["strategy"] == "MV"]

MAIN_ARMS = ["B", "A", "A_changes"]
arms = [a for a in MAIN_ARMS if a in set(mv["arm"])] + \
       sorted(a for a in set(mv["arm"]) if a not in MAIN_ARMS)

# ---- concentration measures, computed per rebalancing date then averaged ----
rows = []
for arm in arms:
    sub = mv[mv["arm"] == arm]
    hhi, max_w, n_held = [], [], []

    for _, grp in sub.groupby("date"):
        weights = grp["weight"].values
        hhi.append(np.sum(weights ** 2))        # Herfindahl index
        max_w.append(weights.max())
        n_held.append((weights > 0.01).sum())   # positions above 1%

    rows.append({
        "arm": arm,
        "mean_HHI": round(float(np.mean(hhi)), 4),
        "effective_positions": round(float(np.mean([1 / h for h in hhi])), 2),
        "mean_max_weight": round(float(np.mean(max_w)), 3),
        "mean_positions_above_1pct": round(float(np.mean(n_held)), 2),
    })

conc = pd.DataFrame(rows)

# ---- average weight per ETF per arm ----
avg = (mv.groupby(["arm", "etf"])["weight"].mean()
         .unstack("arm")
         .reindex(columns=arms)
         .round(4))

print("=== CONCENTRATION (mean-variance portfolios) ===")
print(conc.to_string(index=False))
print("\nHHI: 1 = everything in one ETF, 0.091 = perfectly equal across 11 ETFs.")
print("Effective positions: 1/HHI, i.e. how many ETFs the portfolio behaves like.")

print("\n=== AVERAGE WEIGHT PER ETF ===")
print(avg.to_string())

conc.to_csv("results/weight_concentration.csv", index=False)
avg.to_csv("results/avg_weight_per_etf.csv")
print("\nSaved results/weight_concentration.csv and results/avg_weight_per_etf.csv")
