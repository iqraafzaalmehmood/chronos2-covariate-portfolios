"""Paired bootstrap for the Sharpe-ratio difference between the two MV
portfolios (Arm A with covariates vs Arm B without)."""

import numpy as np
import pandas as pd

N_BOOT = 10_000
rng = np.random.default_rng(42)

rets = pd.read_csv("results/portfolio_returns.csv", parse_dates=["date"])
mv = rets[rets["strategy"] == "MV"].pivot(index="date", columns="arm",
                                          values="return").dropna()
a, b = mv["A"].values, mv["B"].values
n = len(a)


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


obs_diff = sharpe(a) - sharpe(b)

diffs = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n, n)          # same months for both arms (paired)
    diffs[i] = sharpe(a[idx]) - sharpe(b[idx])

ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
frac_positive = (diffs > 0).mean()

print(f"Months compared            : {n}")
print(f"Sharpe A-MV (covariates)   : {sharpe(a):.3f}")
print(f"Sharpe B-MV (no covariates): {sharpe(b):.3f}")
print(f"Observed difference (A-B)  : {obs_diff:.3f}")
print(f"95% bootstrap CI           : [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"Fraction of samples with A > B: {frac_positive:.1%}")
print("\nReading: if the CI contains 0, the gap is not statistically")
print("established at the 5% level on this sample - report it as such.")
