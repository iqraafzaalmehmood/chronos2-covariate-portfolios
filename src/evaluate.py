"""Final evaluation: portfolio metrics (Sharpe, max drawdown, total return,
turnover) for every arm/strategy, plus forecast accuracy per arm."""

import numpy as np
import pandas as pd

HORIZON = 21

prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
rets = pd.read_csv("results/portfolio_returns.csv", parse_dates=["date"])


def portfolio_metrics(r):
    r = r.sort_values("date")
    monthly = r["return"].values
    wealth = np.cumprod(1 + monthly)
    peak = np.maximum.accumulate(wealth)
    return {
        "total_return_pct": round((wealth[-1] - 1) * 100, 2),
        "ann_sharpe": round(monthly.mean() / monthly.std(ddof=1) * np.sqrt(12), 3),
        "max_drawdown_pct": round(((wealth / peak) - 1).min() * 100, 2),
        "avg_turnover": round(r["turnover"].mean(), 3),
        "months": len(monthly),
    }


rows = []
for (arm, strat), grp in rets.groupby(["arm", "strategy"]):
    rows.append({"arm": arm, "strategy": strat, **portfolio_metrics(grp)})
summary = pd.DataFrame(rows).sort_values(["strategy", "arm"])


def forecast_accuracy(arm):
    """Median-forecast error and quantile (pinball) loss at the 21-day horizon."""
    fc = pd.read_csv(f"results/forecasts_arm{arm}.csv", parse_dates=["date"])
    fc = fc[fc["day_ahead"] == HORIZON]
    qcols = [c for c in fc.columns if c.startswith("q")]
    levels = [int(c[1:]) / 100 for c in qcols]

    abs_err, pinball = [], []
    for date, grp in fc.groupby("date"):
        future = prices.loc[prices.index >= date]
        if len(future) < HORIZON:
            continue
        realized = future.iloc[HORIZON - 1]
        for _, row in grp.iterrows():
            y = realized[row["etf"]]
            abs_err.append(abs(row["q50"] - y) / y)
            for c, lv in zip(qcols, levels):
                diff = y - row[c]
                pinball.append(max(lv * diff, (lv - 1) * diff) / y)
    return {"arm": arm,
            "median_MAPE_pct": round(np.mean(abs_err) * 100, 3),
            "mean_quantile_loss_pct": round(np.mean(pinball) * 100, 3)}


acc = pd.DataFrame([forecast_accuracy("A"), forecast_accuracy("B"),
                    forecast_accuracy("A_changes")])

print("\n=== PORTFOLIO PERFORMANCE (A = with covariates, B = without) ===")
print(summary.to_string(index=False))
print("\n=== FORECAST ACCURACY AT 21-DAY HORIZON ===")
print(acc.to_string(index=False))

summary.to_csv("results/main_comparison.csv", index=False)
acc.to_csv("results/forecast_accuracy.csv", index=False)
print("\nSaved results/main_comparison.csv and results/forecast_accuracy.csv")
