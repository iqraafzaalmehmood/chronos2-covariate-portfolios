"""Arm B forecasting loop: generating zero-shot Chronos-2 forecasts
month by month for all 11 ETFs using prices only"""

import os
import pandas as pd
from chronos import BaseChronosPipeline

TEST_START = "2023-01-01"
TEST_END = "2026-07-31"      # last forecast date; leaves a month of real
                             # prices after it for later evaluation
HORIZON = 21                 # trading days ahead (~one month)
CONTEXT_DAYS = 512           # history shown to the model at each date
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
etfs = list(prices.columns)

# rebalance dates = first trading day of each month in the test window
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
    # history strictly BEFORE the forecast date -> no look-ahead
    hist = prices.loc[prices.index < date].tail(CONTEXT_DAYS)

    inputs = [hist[etf].values for etf in etfs]
    quantiles, means = pipeline.predict_quantiles(
        inputs, prediction_length=HORIZON, quantile_levels=QUANTILE_LEVELS
    )

    for j, etf in enumerate(etfs):
        q = quantiles[j][0]          # shape (HORIZON, 9)
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
out.to_csv("results/forecasts_armB.csv", index=False)
print(f"\nSaved {len(out)} forecast rows -> results/forecasts_armB.csv")
