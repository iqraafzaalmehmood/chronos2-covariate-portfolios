import pandas as pd
from chronos import BaseChronosPipeline

from config import (
    PRICE_FILE, COVARIATE_FILE, RESULTS_DIR,
    HORIZON, CONTEXT_DAYS, QUANTILES
)
from forecast_utils import (
    load_prices_covariates, make_nonoverlapping_origins, tensor_to_numpy
)


MODEL_NAME = "amazon/chronos-2"


def main():
    # RESULTS_DIR is created at import time in config.py, before any file writes.
    prices, covs = load_prices_covariates(PRICE_FILE, COVARIATE_FILE)
    etfs = list(prices.columns)
    cov_names = list(covs.columns)
    dates = make_nonoverlapping_origins(prices)

    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    rows = []
    macro_rows = []

    for date in dates:
        hist = prices.loc[prices.index <= date].tail(CONTEXT_DAYS)
        cov_hist = covs.loc[hist.index]

        # Step 1: forecast each macro series t+1,...,t+21 using data through t.
        macro_inputs = [cov_hist[name].to_numpy() for name in cov_names]

        macro_q, _ = pipeline.predict_quantiles(
            macro_inputs,
            prediction_length=HORIZON,
            quantile_levels=[0.5],
        )
        macro_q_np = tensor_to_numpy(macro_q)

        future_cov = {}
        for j, name in enumerate(cov_names):
            vals = macro_q_np[j, 0, :, 0]
            future_cov[name] = vals

            for h, value in enumerate(vals, start=1):
                macro_rows.append({
                    "date": date,
                    "covariate": name,
                    "day_ahead": h,
                    "q50": float(value),
                })

        # Step 2: ETF forecasts use both past covariates and estimated future covariates.
        etf_inputs = []
        for etf in etfs:
            etf_inputs.append({
                "target": hist[etf].to_numpy(),
                "past_covariates": {
                    name: cov_hist[name].to_numpy()
                    for name in cov_names
                },
                "future_covariates": {
                    name: future_cov[name]
                    for name in cov_names
                },
            })

        quantiles, means = pipeline.predict_quantiles(
            etf_inputs,
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

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "forecasts_C.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(
        RESULTS_DIR / "macro_forecasts_C.csv", index=False
    )

    print(f"Saved ETF forecasts to {RESULTS_DIR / 'forecasts_C.csv'}")
    print(f"Saved macro forecasts to {RESULTS_DIR / 'macro_forecasts_C.csv'}")


if __name__ == "__main__":
    main()
