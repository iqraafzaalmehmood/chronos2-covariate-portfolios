from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILE = DATA_DIR / "etf_prices.csv"
COVARIATE_FILE = DATA_DIR / "covariates.csv"

TEST_START = "2023-01-01"
TEST_END = "2026-07-31"

# Fixed-horizon design:
# use information available through decision date t and forecast t+1,...,t+21.
HORIZON = 21
CONTEXT_DAYS = 512

QUANTILES = [0.1, 0.5, 0.9]

RISK_AVERSION = 5.0
COV_WINDOW = 252
MAX_WEIGHT = 0.25
TRANSACTION_COST_BPS = 10.0

SEED = 42
