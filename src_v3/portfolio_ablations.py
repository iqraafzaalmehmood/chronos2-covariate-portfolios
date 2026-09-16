import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import (
    PRICE_FILE, COVARIATE_FILE, RESULTS_DIR, HORIZON, COV_WINDOW,
    RISK_AVERSION, MAX_WEIGHT, TRANSACTION_COST_BPS
)
from forecast_utils import get_horizon_end

def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")

def get_arms():
    covs = pd.read_csv(COVARIATE_FILE, index_col=0, parse_dates=True)
    return ["A"] + [f"A_wo_{safe_name(c)}" for c in covs.columns]

def mean_variance_weights(mu, sigma, gamma=RISK_AVERSION, max_weight=MAX_WEIGHT):
    n = len(mu)
    x0 = np.full(n, 1.0 / n)

    def objective(w):
        return -(w @ mu - 0.5 * gamma * (w @ sigma @ w))

    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints
    )
    return result.x if result.success else x0

def calc_turnover(w, prev_w):
    if prev_w is None:
        return 0.0
    return float(np.abs(w - prev_w).sum())

def main():
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True).sort_index()
    daily_returns = prices.pct_change()
    etfs = list(prices.columns)
    all_rows = []

    for arm in get_arms():
        path = RESULTS_DIR / f"forecasts_{arm}.csv"
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        fc = pd.read_csv(path, parse_dates=["date"])
        end_fc = fc.loc[fc["day_ahead"] == HORIZON].copy()
        dates = sorted(end_fc["date"].unique())

        prev_weights = {"EW": None, "MV_CAP": None}

        for date in dates:
            date = pd.Timestamp(date)
            sell_date = get_horizon_end(prices, date, HORIZON)
            if sell_date is None:
                continue

            rows = end_fc.loc[end_fc["date"] == date].set_index("ticker").reindex(etfs)
            if rows["q50"].isna().any():
                continue

            buy_price = prices.loc[date, etfs]
            sell_price = prices.loc[sell_date, etfs]

            expected_return = rows["q50"].to_numpy() / buy_price.to_numpy() - 1.0
            realized_asset_return = sell_price.to_numpy() / buy_price.to_numpy() - 1.0

            trailing = daily_returns.loc[:date, etfs].dropna().tail(COV_WINDOW)
            if len(trailing) < 30:
                continue

            sigma = trailing.cov().to_numpy() * HORIZON

            strategies = {
                "EW": np.full(len(etfs), 1.0 / len(etfs)),
                "MV_CAP": mean_variance_weights(
                    expected_return, sigma, max_weight=MAX_WEIGHT
                ),
            }

            for strategy, weights in strategies.items():
                turnover = calc_turnover(weights, prev_weights[strategy])
                gross_ret = float(weights @ realized_asset_return)
                cost = (TRANSACTION_COST_BPS / 10000.0) * turnover
                net_ret = gross_ret - cost

                row = {
                    "arm": arm,
                    "strategy": strategy,
                    "date": date,
                    "sell_date": sell_date,
                    "gross_return": gross_ret,
                    "transaction_cost": cost,
                    "net_return": net_ret,
                    "turnover": turnover,
                }

                for ticker, weight in zip(etfs, weights):
                    row[f"w_{ticker}"] = float(weight)

                all_rows.append(row)
                prev_weights[strategy] = weights.copy()

    detail = pd.DataFrame(all_rows)
    detail.to_csv(RESULTS_DIR / "ablation_portfolio_returns.csv", index=False)

    summary_rows = []
    periods_per_year = 252.0 / HORIZON

    for (arm, strategy), g in detail.groupby(["arm", "strategy"]):
        r = g["net_return"].to_numpy()
        wealth = np.cumprod(1.0 + r)
        running_max = np.maximum.accumulate(wealth)
        drawdown = wealth / running_max - 1.0
        sd = np.std(r, ddof=1)

        ann_sharpe = (
            np.sqrt(periods_per_year) * np.mean(r) / sd
            if len(r) > 1 and sd > 0
            else np.nan
        )

        summary_rows.append({
            "arm": arm,
            "strategy": strategy,
            "total_return_pct": (wealth[-1] - 1.0) * 100 if len(wealth) else np.nan,
            "ann_sharpe": ann_sharpe,
            "max_drawdown_pct": np.min(drawdown) * 100 if len(drawdown) else np.nan,
            "avg_turnover": g["turnover"].mean(),
            "periods": len(g),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "ablation_portfolio_summary.csv", index=False)

    print("\nAblation portfolio results:\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
