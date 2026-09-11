"""Multivariate variant: all 11 ETFs are forecast jointly as one group, so
Chronos-2's group attention can use cross-series structure.

Runs both arms under identical conditions:
  A_multi - joint targets + 5 macro covariates
  B_multi - joint targets only

This differs from forecast.py / forecast_armA.py, which forecast each ETF
separately (n_variates = 1) and therefore never exercise group attention.
"""

import os
import numpy as np
import pandas as pd
from chronos import BaseChronosPipeline

TEST_START = "2023-01-01"
TEST_END = "2026-07-31"
HORIZON = 21
CONTEXT_DAYS = 512
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
covs = pd.read_csv("data/covariates.csv", index_col=0, parse_dates=True)
etfs = list(prices.columns)
cov_names = list(covs.columns)

month_starts = (
    prices.loc[TEST_START:TEST_END]
    .groupby(pd.Grouper(freq="MS"))
    .apply(lambda g: g.index.min())
    .dropna()
    .tolist()
)

print(f"{len(month_starts)} forecast dates, {len(etfs)} ETFs forecast jointly")

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")

rows = {"A_multi": [], "B_multi": []}

for i, date in enumerate(month_starts, 1):
    hist = prices.loc[prices.index < date].tail(CONTEXT_DAYS)
    cov_hist = covs.loc[hist.index]

    # shape (n_variates, context) -> one multivariate item instead of 11 items
    target = hist[etfs].values.T

    specs = {
        "B_multi": [{"target": target}],
        "A_multi": [{
            "target": target,
            "past_covariates": {n: cov_hist[n].values for n in cov_names},
        }],
    }

    for arm, inputs in specs.items():
        quantiles, means = pipeline.predict_quantiles(
            inputs, prediction_length=HORIZON, quantile_levels=QUANTILE_LEVELS
        )
        q_all = np.asarray(quantiles[0])        # (n_variates, HORIZON, 9)
        m_all = np.asarray(means[0])            # (n_variates, HORIZON)

        for j, etf in enumerate(etfs):
            for day in range(HORIZON):
                row = {"date": date.date(), "etf": etf, "day_ahead": day + 1,
                       "mean": float(m_all[j, day])}
                for k, level in enumerate(QUANTILE_LEVELS):
                    row[f"q{int(level * 100)}"] = float(q_all[j, day, k])
                rows[arm].append(row)

    print(f"[{i}/{len(month_starts)}] {date.date()} done")

os.makedirs("results", exist_ok=True)
for arm, r in rows.items():
    out = pd.DataFrame(r)
    fname = f"results/forecasts_arm{arm}.csv"
    out.to_csv(fname, index=False)
    print(f"Saved {len(out)} rows -> {fname}")
