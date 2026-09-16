import numpy as np
import pandas as pd

from config import HORIZON, TEST_START, TEST_END


def load_prices_covariates(price_file, covariate_file):
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True).sort_index()
    covs = pd.read_csv(covariate_file, index_col=0, parse_dates=True).sort_index()

    # Keep common trading dates only. No future filling is introduced here.
    common = prices.index.intersection(covs.index)
    prices = prices.loc[common].copy()
    covs = covs.loc[common].copy()

    return prices, covs


def make_nonoverlapping_origins(prices, test_start=TEST_START, test_end=TEST_END, horizon=HORIZON):
    """
    Build non-overlapping fixed-horizon forecast origins.

    If t is an origin, the model uses data through t and predicts the next
    `horizon` observed trading days. The next origin is exactly the sell date
    from the previous period, which prevents overlapping portfolio returns.
    """
    eligible = prices.loc[test_start:test_end].index
    if len(eligible) == 0:
        return []

    all_idx = prices.index
    origins = []

    # first observed trading day on/after TEST_START
    pos = all_idx.get_indexer([eligible[0]])[0]

    while True:
        if pos + horizon >= len(all_idx):
            break

        origin = all_idx[pos]
        sell = all_idx[pos + horizon]

        if origin > pd.Timestamp(test_end) or sell > pd.Timestamp(test_end):
            break

        origins.append(origin)
        pos += horizon

    return origins


def tensor_to_numpy(x):
    """
    Robustly convert Chronos outputs to NumPy whether they are torch tensors
    or already NumPy-like objects.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def get_horizon_end(prices, date, horizon=HORIZON):
    """
    Exact t+horizon observed-trading-day endpoint.
    """
    loc = prices.index.get_loc(date)
    if isinstance(loc, slice):
        loc = loc.start
    end_pos = loc + horizon
    if end_pos >= len(prices.index):
        return None
    return prices.index[end_pos]
