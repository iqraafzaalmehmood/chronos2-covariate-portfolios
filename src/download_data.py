"""Download daily ETF and macro covariate data from Yahoo Finance,
align calendars, and save clean CSVs plus a short data report."""

import os
import pandas as pd
import yfinance as yf

ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

COVARIATES = {
    "^VIX": "VIX_volatility",
    "^TNX": "US10Y_yield",
    "CL=F": "Oil_WTI",
    "GC=F": "Gold",
    "DX-Y.NYB": "USD_index",
}

START = "2012-01-01"


def download_closes(tickers):
    data = yf.download(tickers=tickers, start=START, auto_adjust=True, progress=True)
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])
    return closes


print("Downloading ETF prices ...")
etf = download_closes(ETFS)

print("Downloading covariates ...")
cov = download_closes(list(COVARIATES.keys())).rename(columns=COVARIATES)

# ETF trading days are the master calendar. Covariates trade on other
# calendars, so reindex and forward-fill gaps (past values only, max 5 days).
cov = cov.reindex(etf.index)
gaps_before = cov.isna().sum()
cov = cov.ffill(limit=5)
gaps_after = cov.isna().sum()

# keep only the period where all series exist (XLC starts mid-2018)
start_day = max(etf.dropna().index.min(), cov.dropna().index.min())
etf, cov = etf.loc[start_day:], cov.loc[start_day:]

bad_rows = int(etf.isna().any(axis=1).sum())
keep = ~etf.isna().any(axis=1)
etf, cov = etf.loc[keep], cov.loc[keep]

checks = [
    ("Same number of rows", len(etf) == len(cov)),
    ("Identical calendars", etf.index.equals(cov.index)),
    ("No missing ETF prices", int(etf.isna().sum().sum()) == 0),
    ("No missing covariates", int(cov.isna().sum().sum()) == 0),
    ("All ETF prices positive", bool((etf > 0).all().all())),
    ("VIX in range 5-100", bool(cov["VIX_volatility"].between(5, 100).all())),
]

os.makedirs("data", exist_ok=True)
etf.to_csv("data/etf_prices.csv")
cov.to_csv("data/covariates.csv")

with open("data/data_report.txt", "w") as f:
    f.write(f"Sample period : {etf.index.min().date()} to {etf.index.max().date()}\n")
    f.write(f"Trading days  : {len(etf)}\n")
    f.write(f"ETFs          : {', '.join(etf.columns)}\n")
    f.write(f"Covariates    : {', '.join(cov.columns)}\n\n")
    f.write("Covariate gaps before forward-fill:\n" + gaps_before.to_string() + "\n\n")
    f.write("Covariate gaps after forward-fill (limit=5):\n" + gaps_after.to_string() + "\n\n")
    f.write(f"ETF rows dropped: {bad_rows}\n\n")
    f.write("Notes: master calendar = ETF trading days; covariates forward-filled\n")
    f.write("(past values only, no back-fill); sample starts when all series exist.\n\n")
    for name, ok in checks:
        f.write(f"[{'PASS' if ok else 'FAIL'}] {name}\n")

print(f"\nSaved {len(etf)} trading days "
      f"({etf.index.min().date()} to {etf.index.max().date()})")
print("OK" if all(ok for _, ok in checks) else "WARNING: check data_report.txt")
