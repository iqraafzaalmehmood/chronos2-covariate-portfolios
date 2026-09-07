"""Turn Arm A and Arm B forecasts into portfolio weights (mean-variance and
equal-weight), simulate monthly rebalancing, and save weights + returns."""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HORIZON = 21
RISK_AVERSION = 5.0
COV_WINDOW = 252            # trailing days for the covariance matrix

prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
etfs = list(prices.columns)
daily_ret = prices.pct_change()


def mean_variance_weights(mu, sigma, gamma=RISK_AVERSION):
    """Long-only max of w'mu - gamma/2 w'Sigma w with weights summing to 1."""
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_utility(w):
        return -(w @ mu - 0.5 * gamma * w @ sigma @ w)

    res = minimize(
        neg_utility, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    return res.x if res.success else w0


def run_arm(arm):
    fc = pd.read_csv(f"results/forecasts_arm{arm}.csv", parse_dates=["date"])
    # expected 1-month return per ETF: median forecast at day 21 vs last price
    end_of_month = fc[fc["day_ahead"] == HORIZON]

    dates = sorted(end_of_month["date"].unique())
    weight_rows, return_rows = [], []
    prev_w = {"MV": None, "EW": None}

    for t, date in enumerate(dates):
        last_price = prices.loc[prices.index < date].iloc[-1]
        snap = end_of_month[end_of_month["date"] == date].set_index("etf")
        mu = np.array([snap.loc[e, "q50"] / last_price[e] - 1.0 for e in etfs])

        hist = daily_ret.loc[daily_ret.index < date].tail(COV_WINDOW)
        sigma = hist.cov().values * HORIZON        # scale daily cov to 1 month

        weights = {
            "MV": mean_variance_weights(mu, sigma),
            "EW": np.ones(len(etfs)) / len(etfs),
        }

        # realized holding-period return: this rebalance date to the next one
        # (for the last date: HORIZON trading days ahead)
        buy = prices.loc[date]
        if t + 1 < len(dates):
            sell = prices.loc[dates[t + 1]]
        else:
            future = prices.loc[prices.index > date]
            sell = future.iloc[min(HORIZON, len(future)) - 1]
        realized = (sell / buy - 1.0).values

        for strat, w in weights.items():
            port_ret = float(w @ realized)
            turnover = float(np.abs(w - prev_w[strat]).sum()) if prev_w[strat] is not None else 0.0
            prev_w[strat] = w
            return_rows.append({"date": date.date(), "arm": arm, "strategy": strat,
                                "return": port_ret, "turnover": turnover})
            for e, wi in zip(etfs, w):
                weight_rows.append({"date": date.date(), "arm": arm,
                                    "strategy": strat, "etf": e, "weight": float(wi)})

    return pd.DataFrame(weight_rows), pd.DataFrame(return_rows)


all_w, all_r = [], []
for arm in ["A", "B", "A_changes"]:
    w, r = run_arm(arm)
    all_w.append(w)
    all_r.append(r)
    print(f"Arm {arm}: {r['date'].nunique()} months simulated")

os.makedirs("results", exist_ok=True)
pd.concat(all_w).to_csv("results/weights.csv", index=False)
pd.concat(all_r).to_csv("results/portfolio_returns.csv", index=False)
print("Saved results/weights.csv and results/portfolio_returns.csv")
