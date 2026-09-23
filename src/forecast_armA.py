"""Arm A forecasting loop: identical to Arm B except each ETF target is
accompanied by the 5 macro series as past-only covariates."""

import os
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

print(f"{len(month_starts)} forecast dates from "
      f"{month_starts[0].date()} to {month_starts[-1].date()}")

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")

rows = []
for i, date in enumerate(month_starts, 1):
    # Use only data available before the forecast date to avoid look-ahead bias.
    hist = prices.loc[prices.index < date].tail(CONTEXT_DAYS)
    cov_hist = covs.loc[hist.index]          # same dates as the target window

    inputs = [
        {
            "target": hist[etf].values,
            "past_covariates": {name: cov_hist[name].values for name in cov_names},
        }
        for etf in etfs
    ]

    quantiles, means = pipeline.predict_quantiles(
        inputs, prediction_length=HORIZON, quantile_levels=QUANTILE_LEVELS
    )

    for j, etf in enumerate(etfs):
        q = quantiles[j][0]
        m = means[j][0] if means[j].ndim > 1 else means[j]
        for day in range(HORIZON):
            row = {"date": date.date(), "etf": etf, "day_ahead": day + 1,
                   "mean": float(m[day])}
            for k, level in enumerate(QUANTILE_LEVELS):
                row[f"q{int(level * 100)}"] = float(q[day, k])
            rows.append(row)

    print(f"[{i}/{len(month_starts)}] {date.date()} done")

out = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
out.to_csv("results/forecasts_armA.csv", index=False)
print(f"\nSaved {len(out)} forecast rows -> results/forecasts_armA.csv")
