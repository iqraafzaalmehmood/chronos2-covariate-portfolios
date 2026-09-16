import pandas as pd
from chronos import BaseChronosPipeline

from config import PRICE_FILE, COVARIATE_FILE, RESULTS_DIR, HORIZON, CONTEXT_DAYS, QUANTILES
from forecast_utils import load_prices_covariates, make_nonoverlapping_origins, tensor_to_numpy

MODEL_NAME = "amazon/chronos-2"

def safe_name(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")

def main():
    prices, covs = load_prices_covariates(PRICE_FILE, COVARIATE_FILE)
    etfs = list(prices.columns)
    all_covariates = list(covs.columns)
    dates = make_nonoverlapping_origins(prices)

    print("Covariates found:")
    for c in all_covariates:
        print(" -", c)

    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype="auto"
    )

    experiments = {
        f"A_wo_{safe_name(omitted)}": [c for c in all_covariates if c != omitted]
        for omitted in all_covariates
    }

    for arm_name, selected_covariates in experiments.items():
        print(f"\n=== Running {arm_name} ===")
        rows = []

        for date in dates:
            hist = prices.loc[prices.index <= date].tail(CONTEXT_DAYS)
            cov_hist = covs.loc[hist.index, selected_covariates]

            inputs = [{
                "target": hist[etf].to_numpy(),
                "past_covariates": {
                    name: cov_hist[name].to_numpy()
                    for name in selected_covariates
                },
            } for etf in etfs]

            quantiles, means = pipeline.predict_quantiles(
                inputs,
                prediction_length=HORIZON,
                quantile_levels=QUANTILES,
            )

            q_np = tensor_to_numpy(quantiles)
            mean_np = tensor_to_numpy(means)

            for j, etf in enumerate(etfs):
                for h in range(HORIZON):
                    rows.append({
                        "date": date,
                        "ticker": etf,
                        "day_ahead": h + 1,
                        "q10": float(q_np[j, 0, h, 0]),
                        "q50": float(q_np[j, 0, h, 1]),
                        "q90": float(q_np[j, 0, h, 2]),
                        "mean": float(mean_np[j, 0, h]),
                    })

        out = pd.DataFrame(rows)
        out_path = RESULTS_DIR / f"forecasts_{arm_name}.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {len(out):,} rows to {out_path}")

if __name__ == "__main__":
    main()
