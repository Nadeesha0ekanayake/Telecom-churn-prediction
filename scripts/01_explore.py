"""Step 1 runner: data loading & exploration (Path a, local).

Thin orchestrator — all real logic lives in src/. Run from the project root:

    ./venv/bin/python scripts/01_explore.py

Prints a section-by-section EDA report to stdout and writes figures to
reports/figures/.
"""
import sys
from pathlib import Path

# Make the project root importable so `from src import ...` works when this
# file is run directly (Python only adds the script's own dir to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader, eda  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    df = data_loader.load_raw()

    section("1. OVERVIEW")
    eda.overview(df)

    section("2. MISSING VALUES (NaN nulls + blank/whitespace strings)")
    report = eda.missing_report(df)
    print(report.to_string() if len(report) else "No missing values detected.")

    section("3. DUPLICATES")
    eda.duplicate_report(df)

    section("4. TARGET BALANCE (Churn)")
    eda.target_balance(df)

    section("5. NUMERIC SUMMARY")
    eda.numeric_summary(df)

    section("6. CATEGORICAL SUMMARY")
    eda.categorical_summary(df)

    section("7. FIGURES")
    for path in (
        eda.plot_target(df),
        eda.plot_churn_by_category(df, "Contract"),
        eda.plot_numeric_by_churn(df),
        eda.plot_correlation(df),
    ):
        print(f"saved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
