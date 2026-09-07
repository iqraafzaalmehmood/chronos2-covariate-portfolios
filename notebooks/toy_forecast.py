import pandas as pd
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")

prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
context = prices["XLK"].tail(500)

quantiles, means = pipeline.predict_quantiles(
    [context.values], prediction_length=21
)

print("quantiles shape:", quantiles[0].shape)
print("median path:", quantiles[0][0, :, 4])   # the 0.5 quantile = best guess
print("mean forecast:", means[0])