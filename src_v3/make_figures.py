from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v3"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def bar(df, x, y, title, ylabel, filename):
    ax = df.plot(kind="bar", x=x, y=y, legend=False, figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("Results:", RESULTS)
    print("Figures:", FIGURES)

    p = RESULTS / "main_comparison.csv"
    if p.exists():
        df = pd.read_csv(p)
        mv = df[df["strategy"] == "MV_CAP"].copy()
        bar(mv, "arm", "total_return_pct", "Portfolio Total Return: A vs B vs C",
            "Total return (%)", "figure_1_total_return_ABC.png")
        bar(mv, "arm", "ann_sharpe", "Portfolio Sharpe Ratio: A vs B vs C",
            "Annualized Sharpe ratio", "figure_2_sharpe_ABC.png")
    else:
        print("Missing:", p.name)

    p = RESULTS / "portfolio_returns.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        mv = df[df["strategy"] == "MV_CAP"].copy()
        plt.figure(figsize=(9, 5))
        for arm, g in mv.groupby("arm"):
            g = g.sort_values("date")
            wealth = (1 + g["net_return"]).cumprod()
            plt.plot(g["date"], wealth, label=f"Arm {arm}")
        plt.title("Cumulative Portfolio Value: A vs B vs C")
        plt.xlabel("Date")
        plt.ylabel("Growth of €1 invested")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURES / "figure_3_cumulative_portfolio_ABC.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("Missing:", p.name)

    p = RESULTS / "forecast_accuracy.csv"
    if p.exists():
        df = pd.read_csv(p)
        bar(df, "arm", "return_MAE_pct", "Return Forecast Error: A vs B vs C",
            "Return MAE (%) — lower is better", "figure_4_return_MAE_ABC.png")
    else:
        print("Missing:", p.name)

    labels = {
        "A": "All covariates",
        "A_wo_Oil_WTI": "Without Oil",
        "A_wo_USD_index": "Without USD",
        "A_wo_Gold": "Without Gold",
        "A_wo_US10Y_yield": "Without US10Y",
        "A_wo_VIX_volatility": "Without VIX",
    }

    p = RESULTS / "ablation_portfolio_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        mv = df[df["strategy"] == "MV_CAP"].copy()
        mv["experiment"] = mv["arm"].map(labels).fillna(mv["arm"])
        bar(mv, "experiment", "total_return_pct", "Covariate Ablation: Total Return",
            "Total return (%)", "figure_5_ablation_total_return.png")
        bar(mv, "experiment", "ann_sharpe", "Covariate Ablation: Sharpe Ratio",
            "Annualized Sharpe ratio", "figure_6_ablation_sharpe.png")
    else:
        print("Missing:", p.name)

    p = RESULTS / "ablation_forecast_accuracy.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["experiment"] = df["arm"].map(labels).fillna(df["arm"])
        bar(df, "experiment", "return_MAE_pct", "Covariate Ablation: Return Forecast Error",
            "Return MAE (%) — lower is better", "figure_7_ablation_return_MAE.png")
    else:
        print("Missing:", p.name)

    print("\nDone. Open:", FIGURES)

if __name__ == "__main__":
    main()
