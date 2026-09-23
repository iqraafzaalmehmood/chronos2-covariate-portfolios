"""Leave-one-out ablations: reruning the Arm A forecasting loop five times and
each time excluding one of the covariates to see which signal drives the results."""

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
all_covs = list(covs.columns)

month_starts = (
    prices.loc[TEST_START:TEST_END]
    .groupby(pd.Grouper(freq="MS"))
    .apply(lambda g: g.index.min())
    .dropna()
    .tolist()
)

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")
os.makedirs("results", exist_ok=True)

for excluded in all_covs:
    used = [c for c in all_covs if c != excluded]
    print(f"\n=== Ablation: WITHOUT {excluded} (using {len(used)} covariates) ===")

    rows = []
    for i, date in enumerate(month_starts, 1):
        hist = prices.loc[prices.index < date].tail(CONTEXT_DAYS)
        cov_hist = covs.loc[hist.index, used]

        inputs = [
            {"target": hist[etf].values,
             "past_covariates": {name: cov_hist[name].values for name in used}}
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

        if i % 10 == 0 or i == len(month_starts):
            print(f"  [{i}/{len(month_starts)}] {date.date()} done")

    out = pd.DataFrame(rows)
    fname = f"results/forecasts_armA_wo_{excluded}.csv"
    out.to_csv(fname, index=False)
    print(f"  Saved {len(out)} rows -> {fname}")

print("\nAll five ablations complete.")
