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
    prices, _ = load_prices_covariates(PRICE_FILE, COVARIATE_FILE)
    etfs = list(prices.columns)
    dates = make_nonoverlapping_origins(prices)

    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    rows = []

    for date in dates:
        # IMPORTANT: information through decision date t is available.
        hist = prices.loc[prices.index <= date].tail(CONTEXT_DAYS)

        inputs = [hist[etf].to_numpy() for etf in etfs]

        quantiles, means = pipeline.predict_quantiles(
            inputs,
            prediction_length=HORIZON,
            quantile_levels=QUANTILES,
        )

        q_np = tensor_to_numpy(quantiles)
        mean_np = tensor_to_numpy(means)
        

        for j, etf in enumerate(etfs):
            # Chronos returns [series, horizon, quantile] for quantiles.
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
    out.to_csv(RESULTS_DIR / "forecasts_B.csv", index=False)
    print(f"Saved {len(out):,} rows to {RESULTS_DIR / 'forecasts_B.csv'}")


if __name__ == "__main__":
    main()
