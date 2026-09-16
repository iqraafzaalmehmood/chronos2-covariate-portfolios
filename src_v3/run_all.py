import subprocess
import sys


STEPS = [
    "forecast.py",        # Arm B: no covariates
    "forecast_armA.py",   # Arm A: historical covariates
    "forecast_armC.py",   # Arm C: historical + forecast future covariates
    "evaluate.py",
    "portfolio.py",
    "significance.py",
]


def main():
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, f"src_v3/{step}"], check=True)


if __name__ == "__main__":
    main()
