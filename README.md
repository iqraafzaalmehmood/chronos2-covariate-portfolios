# Do macro covariates improve foundation-model portfolios?

This project investigates whether adding macroeconomic covariates to **Chronos-2**
improves zero-shot ETF forecasts and, importantly, whether any forecasting improvements
translate into better portfolio outcomes.

The experiment uses 11 US sector ETFs and five macroeconomic covariates:

- VIX volatility
- US 10-year Treasury yield
- WTI crude oil
- Gold
- US dollar index

## Research design

The revised experiment compares three forecasting arms.

- **Arm B — Baseline:** ETF price history only.
- **Arm A — Historical covariates:** ETF price history plus historical macroeconomic
  covariates available through the decision date.
- **Arm C — Future-covariate extension:** ETF price history and historical macro
  covariates, plus Chronos-2 forecasts of the macro variables over the future
  prediction horizon.

Arm A versus Arm B is the **primary comparison**.

Arm C is a secondary experiment testing whether forecasting the future macroeconomic
paths before forecasting ETF prices provides additional value.

Chronos-2 is used zero-shot without fine-tuning.

## Timing and backtest design

At each decision date `t`, information available through `t` is used.

Chronos-2 forecasts ETF prices for:

`t+1, ..., t+21`

The portfolio is formed using the price at `t` and evaluated at `t+21`.

Therefore, the forecasting horizon and portfolio holding horizon are both exactly
21 trading days.

The evaluation contains 42 non-overlapping portfolio periods.

This design avoids using realized future information and keeps the forecast horizon,
expected-return calculation, and investment horizon aligned.

## Main results

### Forecasting performance

| Arm | Median MAPE (%) | Return MAE (%) | Return RMSE (%) | Directional accuracy (%) | Mean rank IC |
|---|---:|---:|---:|---:|---:|
| A | 2.957 | 3.976 | 5.297 | 49.57 | 0.050 |
| B | 3.028 | 4.034 | 5.509 | 53.90 | 0.006 |
| C | 2.967 | 3.945 | 5.466 | 51.95 | 0.107 |

Historical macro covariates (Arm A) improve several forecast-error measures relative
to the no-covariate baseline (Arm B), although Arm B has higher directional accuracy.

Arm C achieves the lowest return MAE and highest cross-sectional rank IC, but does
not dominate the other arms across all forecasting metrics.

### Portfolio performance

The forecast-driven portfolio uses long-only mean-variance optimization with a
maximum weight of 25% per ETF, a trailing 252-trading-day covariance estimate,
risk-aversion parameter gamma = 5, and 10 basis points of transaction costs.

| Arm | Strategy | Total return (%) | Annualized Sharpe | Max drawdown (%) | Avg. turnover |
|---|---|---:|---:|---:|---:|
| A | MV_CAP | 65.36 | 1.098 | -12.76 | 1.012 |
| B | MV_CAP | 44.92 | 0.839 | -16.40 | 0.756 |
| C | MV_CAP | 63.46 | 1.050 | -16.08 | 0.962 |
| — | Equal Weight | 72.43 | 1.274 | -14.74 | 0.000 |

Arm A therefore produces higher observed portfolio return and Sharpe ratio than the
no-covariate Arm B.

However, the simple equal-weight benchmark outperforms all three forecast-driven
portfolios in total return and Sharpe ratio.

## Statistical significance

A paired circular block bootstrap with 10,000 resamples is used to test differences
in portfolio Sharpe ratios while accounting for time dependence.

For the primary comparison, Arm A versus Arm B:

- Sharpe difference: **0.264**
- 95% bootstrap CI: **[-0.202, 0.735]**
- two-sided p-value: **0.2396**

Thus, Arm A shows an economically meaningful improvement over Arm B in this backtest,
but the improvement is **not statistically significant at the conventional 5% level**.

The results therefore should not be interpreted as conclusive evidence that macro
covariates improve portfolio performance.

## Covariate ablation

Leave-one-covariate-out experiments are used to investigate which variables are
associated with the performance of Arm A.

| Model | Total return (%) | Annualized Sharpe |
|---|---:|---:|
| All covariates | 65.36 | 1.098 |
| Without Gold | 58.85 | 1.027 |
| Without Oil | 49.32 | 0.930 |
| Without US10Y | 63.63 | 1.090 |
| Without USD | 63.43 | 1.075 |
| Without VIX | 64.29 | 1.111 |

Removing Oil produces the largest deterioration in portfolio performance, while
removing Gold also reduces total return and Sharpe.

The effects of US10Y and USD are smaller. VIX produces mixed results: removing it
slightly reduces total return but slightly increases the Sharpe ratio.

These ablations are descriptive and are not interpreted as causal evidence.

## Interpretation

The results highlight the distinction between **statistical forecast accuracy** and
**economic value**.

Adding historical macroeconomic covariates improves several forecasting metrics and
is associated with substantially better mean-variance portfolio performance relative
to the Chronos-2 price-only baseline.

However:

1. the Sharpe improvement is not statistically significant;
2. the equal-weight benchmark remains stronger than the forecast-driven portfolios;
3. forecasting future macro covariates (Arm C) does not improve overall portfolio
   performance beyond using historical covariates (Arm A); and
4. ablation results suggest that the economic contribution differs substantially
   across macro variables.

The evidence therefore supports a cautious conclusion: macro covariates can affect
the economic usefulness of Chronos-2 forecasts, but the observed portfolio improvement
is not statistically conclusive in this sample.

## Setup

### 1. Install Python

Python 3.11+ is recommended.

```bash
python --version
```

### 2. Clone the repository

```bash
git clone https://github.com/iqraafzaalmehmood/chronos2-covariate-portfolios.git
cd chronos2-covariate-portfolios
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install "chronos-forecasting>=2.0"
```

Run commands from the repository root.

## Running the V3 experiment

### Forecasting

```bash
python src_v3/forecast.py
python src_v3/forecast_armA.py
python src_v3/forecast_armC.py
```

### Forecast evaluation

```bash
python src_v3/evaluate.py
```

### Portfolio backtest

```bash
python src_v3/portfolio.py
```

### Statistical significance

```bash
python src_v3/significance.py
```

### Covariate ablations

```bash
python src_v3/forecast_ablations.py
python src_v3/evaluate_ablations.py
python src_v3/portfolio_ablations.py
```

### Figures

```bash
python src_v3/make_figures.py
```

Final summary outputs and figures are written to:

```text
results_v3/
```

Large intermediate forecast files are generated locally and excluded from Git.

## Repository layout

```text
src/          original/legacy implementation
src_v3/       revised experimental implementation used for the current results
results/      outputs from the original implementation
results_v3/   final V3 summary results and figures
notebooks/    small experimental/smoke-test notebooks
docs/         project documentation
data/         local input data (gitignored)
```

## Methodological notes

- **No look-ahead:** forecasts use information available through decision date `t`
  and predict `t+1` through `t+21`.
- **Aligned horizons:** expected and realized portfolio returns both use `P_t` as
  their starting price and `P_(t+21)` as the horizon endpoint.
- **Frozen model:** Chronos-2 is used zero-shot without fine-tuning.
- **Primary experiment:** Arm A versus Arm B.
- **Secondary experiment:** Arm C evaluates forecast future macro covariates.
- **Portfolio constraints:** long-only with a maximum 25% allocation to each ETF.
- **Transaction costs:** 10 basis points.
- **Benchmark:** equal-weight allocation across the 11 ETFs.
- **Inference:** paired circular block bootstrap is used for Sharpe-ratio comparisons.

## Reproducibility

The repository keeps the revised source code, summary result tables, significance
results, ablation summaries, and figures under version control.

Large forecast-level intermediate CSV files and local data are excluded because they
can be regenerated from the source scripts.