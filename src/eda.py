"""Exploratory data analysis for the Telco churn dataset (Step 1, Path a).

Pure pandas/matplotlib/seaborn — no notebook or Databricks dependency — so the
same functions run locally now and, unchanged, on Databricks later.

Text functions print a section report. Plot functions save a PNG to
reports/figures/ and return its path.
"""
from __future__ import annotations

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: save figures, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402

sns.set_theme(style="whitegrid")

# Columns that are numeric in meaning (TotalCharges arrives as text — see below)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with TotalCharges coerced to float (blanks -> NaN)."""
    tmp = df.copy()
    tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
    return tmp


# --------------------------------------------------------------------------- #
# Text reports
# --------------------------------------------------------------------------- #
def overview(df: pd.DataFrame) -> None:
    """Print shape, dtypes and memory footprint."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    print("Dtypes:")
    print(df.dtypes.to_string())
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"\nMemory: {mem:.2f} MB")


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report NaN nulls AND blank/whitespace strings.

    pandas does not treat a whitespace string (" ") as null, so a plain
    isna() misses the real gaps in object columns. We check both.
    """
    nan_nulls = df.isna().sum()
    blanks = {
        col: (df[col].astype(str).str.strip() == "").sum()
        for col in df.select_dtypes(include="object")
    }
    blanks = pd.Series(blanks, dtype="int64").reindex(df.columns, fill_value=0)
    report = pd.DataFrame({"nan_nulls": nan_nulls, "blank_strings": blanks})
    report["total_missing"] = report["nan_nulls"] + report["blank_strings"]
    return report[report["total_missing"] > 0]


def duplicate_report(df: pd.DataFrame) -> None:
    """Print full-row and id-level duplicate counts."""
    print(f"Full duplicate rows : {df.duplicated().sum()}")
    print(
        f"Duplicate {config.ID_COL}: {df[config.ID_COL].duplicated().sum()} "
        f"(unique ids: {df[config.ID_COL].nunique():,} of {len(df):,})"
    )


def target_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return the churn class balance."""
    out = pd.DataFrame(
        {
            "count": df[config.TARGET].value_counts(),
            "proportion": df[config.TARGET].value_counts(normalize=True).round(4),
        }
    )
    print(out.to_string())
    return out


def numeric_summary(df: pd.DataFrame) -> None:
    """Describe the numeric columns (TotalCharges coerced first)."""
    tmp = _coerce_numeric(df)
    num_cols = tmp.select_dtypes(include="number").columns
    print(tmp[num_cols].describe().T.to_string())


def categorical_summary(df: pd.DataFrame, max_levels: int = 15) -> None:
    """Print level counts for each *true* categorical column.

    Excludes the id and the numeric-in-meaning columns (e.g. TotalCharges,
    which is stored as text and would otherwise look like a 6,531-level
    category). High-cardinality columns are truncated to the top `max_levels`.
    """
    cats = df.select_dtypes(include="object").columns.drop(
        [config.ID_COL, *NUMERIC_COLS], errors="ignore"
    )
    for col in cats:
        n = df[col].nunique()
        counts = df[col].value_counts(dropna=False)
        print(f"\n{col}  ({n} levels)")
        print(counts.head(max_levels).to_string())
        if n > max_levels:
            print(f"... (+{n - max_levels} more levels)")


# --------------------------------------------------------------------------- #
# Plots  (each returns the saved path)
# --------------------------------------------------------------------------- #
def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_target(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = df[config.TARGET].value_counts().index
    sns.countplot(data=df, x=config.TARGET, order=order, hue=config.TARGET,
                  legend=False, ax=ax)
    ax.set_title("Churn distribution")
    for c in ax.containers:
        ax.bar_label(c)
    return _save(fig, "01_target_distribution.png")


def plot_churn_by_category(df: pd.DataFrame, col: str = "Contract"):
    """Stacked churn-rate bars for one categorical driver."""
    tab = pd.crosstab(df[col], df[config.TARGET], normalize="index")
    fig, ax = plt.subplots(figsize=(6, 4))
    tab.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
    ax.set_ylabel("proportion")
    ax.set_title(f"Churn rate by {col}")
    ax.legend(title=config.TARGET, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, "02_churn_by_contract.png")


def plot_numeric_by_churn(df: pd.DataFrame):
    """KDE of each numeric feature split by churn."""
    tmp = _coerce_numeric(df)
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.kdeplot(data=tmp, x=col, hue=config.TARGET, common_norm=False,
                    fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{col} by churn")
    return _save(fig, "03_numeric_by_churn.png")


def plot_correlation(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    num = _coerce_numeric(df).select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlations")
    return _save(fig, "04_correlation_heatmap.png")
