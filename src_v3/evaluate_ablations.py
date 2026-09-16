import numpy as np
import pandas as pd

from config import PRICE_FILE, COVARIATE_FILE, RESULTS_DIR, HORIZON
from forecast_utils import get_horizon_end

def pinball_loss(y, qhat, q):
    e = y - qhat
    return max(q * e, (q - 1.0) * e)

def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")

def get_arms():
    covs = pd.read_csv(COVARIATE_FILE, index_col=0, parse_dates=True)
    return ["A"] + [f"A_wo_{safe_name(c)}" for c in covs.columns]

def main():
    prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True).sort_index()
    etfs = list(prices.columns)
    detail_rows = []

    for arm in get_arms():
        path = RESULTS_DIR / f"forecasts_{arm}.csv"
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        fc = pd.read_csv(path, parse_dates=["date"])
        end_fc = fc.loc[fc["day_ahead"] == HORIZON].copy()

        for _, row in end_fc.iterrows():
            date = pd.Timestamp(row["date"])
            ticker = row["ticker"]
            if ticker not in etfs or date not in prices.index:
                continue

            end_date = get_horizon_end(prices, date, HORIZON)
            if end_date is None:
                continue

            p0 = float(prices.loc[date, ticker])
            y = float(prices.loc[end_date, ticker])
            pred_price = float(row["q50"])
            pred_ret = pred_price / p0 - 1.0
            actual_ret = y / p0 - 1.0

            detail_rows.append({
                "arm": arm,
                "date": date,
                "ticker": ticker,
                "actual_price": y,
                "pred_price": pred_price,
                "price_ape": abs(pred_price - y) / abs(y),
                "pinball_q10_scaled": pinball_loss(y, float(row["q10"]), 0.1) / abs(y),
                "pinball_q50_scaled": pinball_loss(y, float(row["q50"]), 0.5) / abs(y),
                "pinball_q90_scaled": pinball_loss(y, float(row["q90"]), 0.9) / abs(y),
                "actual_return": actual_ret,
                "pred_return": pred_ret,
                "return_abs_error": abs(pred_ret - actual_ret),
                "return_sq_error": (pred_ret - actual_ret) ** 2,
                "direction_correct": float(np.sign(pred_ret) == np.sign(actual_ret)),
            })

    detail = pd.DataFrame(detail_rows)

    rank_rows = []
    for (arm, date), g in detail.groupby(["arm", "date"]):
        if len(g) >= 3:
            rank_rows.append({
                "arm": arm,
                "date": date,
                "rank_ic": g["pred_return"].corr(g["actual_return"], method="spearman"),
            })

    rank_df = pd.DataFrame(rank_rows)
    summary_rows = []

    for arm, g in detail.groupby("arm"):
        ric = rank_df.loc[rank_df["arm"] == arm, "rank_ic"] if not rank_df.empty else pd.Series(dtype=float)
        summary_rows.append({
            "arm": arm,
            "median_MAPE_pct": 100 * g["price_ape"].median(),
            "mean_quantile_loss_pct": 100 * g[
                ["pinball_q10_scaled", "pinball_q50_scaled", "pinball_q90_scaled"]
            ].to_numpy().mean(),
            "return_MAE_pct": 100 * g["return_abs_error"].mean(),
            "return_RMSE_pct": 100 * np.sqrt(g["return_sq_error"].mean()),
            "directional_accuracy_pct": 100 * g["direction_correct"].mean(),
            "mean_rank_IC": ric.mean() if len(ric) else np.nan,
        })

    summary = pd.DataFrame(summary_rows).sort_values("arm")
    detail.to_csv(RESULTS_DIR / "ablation_forecast_evaluation_detail.csv", index=False)
    rank_df.to_csv(RESULTS_DIR / "ablation_forecast_rank_ic.csv", index=False)
    summary.to_csv(RESULTS_DIR / "ablation_forecast_accuracy.csv", index=False)

    print("\nAblation forecast results:\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
