import numpy as np
import pandas as pd

from config import RESULTS_DIR, HORIZON, SEED


N_BOOT = 10000
BLOCK_LENGTH = 3


def sharpe(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if sd == 0:
        return np.nan
    periods_per_year = 252.0 / HORIZON
    return np.sqrt(periods_per_year) * x.mean() / sd


def circular_block_bootstrap_diff(a, b, n_boot=N_BOOT, block_length=BLOCK_LENGTH, seed=SEED):
    """
    Paired circular block bootstrap for Sharpe(A) - Sharpe(B).
    Preserves short-range serial dependence better than IID resampling.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b)

    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)

    for k in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            block = [(start + j) % n for j in range(block_length)]
            idx.extend(block)

        idx = np.asarray(idx[:n])
        diffs[k] = sharpe(a[idx]) - sharpe(b[idx])

    return diffs


def compare(df, arm1, arm2, strategy="MV_CAP"):
    x = df[(df["arm"] == arm1) & (df["strategy"] == strategy)][["date", "net_return"]]
    y = df[(df["arm"] == arm2) & (df["strategy"] == strategy)][["date", "net_return"]]

    m = x.merge(y, on="date", suffixes=(f"_{arm1}", f"_{arm2}"))
    a = m[f"net_return_{arm1}"].to_numpy()
    b = m[f"net_return_{arm2}"].to_numpy()

    obs = sharpe(a) - sharpe(b)
    boot = circular_block_bootstrap_diff(a, b)

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_two_sided = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))

    return {
        "arm_1": arm1,
        "arm_2": arm2,
        "strategy": strategy,
        "periods": len(m),
        "sharpe_diff": obs,
        "bootstrap_ci_2.5": ci_low,
        "bootstrap_ci_97.5": ci_high,
        "bootstrap_p_two_sided": min(float(p_two_sided), 1.0),
    }


def main():
    df = pd.read_csv(RESULTS_DIR / "portfolio_returns.csv", parse_dates=["date"])

    rows = [
        compare(df, "A", "B"),
        compare(df, "C", "B"),
        compare(df, "C", "A"),
    ]

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "significance.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
