# 01 · Data Loading & Exploration (Step 1, Path α)

> Learning notes: **what we found and why each check matters.** Code:
> `src/data_loader.py` (load), `src/eda.py` (checks + plots),
> `scripts/01_explore.py` (runner). Figures in `reports/figures/`.
>
> Run it yourself: `./venv/bin/python scripts/01_explore.py`

## Why explore *before* modelling?

You cannot choose sensible cleaning, encoding, or models without first knowing the
shape of the data, what's missing, whether the target is balanced, and which features
look predictive. Every decision in Steps 2–5 traces back to something found here.

## What we did & found

### 1. Overview — `overview()`
- **7,043 rows × 21 columns**, ~7.8 MB in memory. Small — pandas is plenty; no need for
  Spark locally (a point we'll revisit on Databricks).
- One row per customer. Columns: 1 id, 1 target (`Churn`), 3 numeric, the rest categorical.
- **Red flag in dtypes:** `TotalCharges` loaded as `object` (text), not a number — even
  though it's clearly a dollar amount. That's the thread we pull on next.

### 2. Missing values — `missing_report()`
- A plain `isna()` finds **zero** nulls. But `TotalCharges` has **11 blank/whitespace
  strings** — which pandas does *not* count as null.
- **Lesson:** missing data can hide as empty strings in text columns. We explicitly check
  for `" "` in every object column, not just `NaN`.
- **Root cause (verified):** all 11 blanks have `tenure = 0` and all 11 `Churn = No` —
  i.e. brand-new customers who haven't been billed a cumulative total yet. Not corrupt
  data. → **Step 2 plan:** coerce `TotalCharges` to numeric and impute these 11 as `0`
  (a tenure-0 customer has accrued $0), rather than dropping them.

### 3. Duplicates — `duplicate_report()`
- **0** full-duplicate rows; **0** duplicate `customerID` (7,043 unique of 7,043). Clean.

### 4. Target balance — `target_balance()`
- **Churn = No: 5,174 (73.5%)** · **Yes: 1,869 (26.5%)**.
- **Imbalanced** (roughly 3:1). Consequences we'll act on:
  - **Accuracy is misleading** — predicting "No" for everyone already scores 73.5%. We'll
    lead with **AUC-ROC, precision, recall, F1**, not accuracy.
  - We'll use **stratified** splits (preserve the 26.5% in train/test) and later try
    `class_weight` / SMOTE / threshold tuning (the Step 2 options).

### 5. Numeric summary — `numeric_summary()`
| feature | note |
|---|---|
| `tenure` | 0–72 months, mean ~32. Months a customer has stayed. |
| `MonthlyCharges` | \$18.25–\$118.75, mean ~\$65. |
| `TotalCharges` | after coercion: 7,032 non-null (the 11 blanks → NaN), \$18.8–\$8,684. |
| `SeniorCitizen` | stored 0/1 — **numeric in storage but categorical in meaning**; don't scale it like a continuous variable. |

### 6. Categorical summary — `categorical_summary()`
- Most are 2–4 levels. Note a recurring encoding pattern: service columns use
  **"No internet service"** / **"No phone service"** as a third level (1,526 and 682 rows).
  These are logically "No", but they also encode a dependency (no internet ⇒ no online
  security, etc.). Something to decide in Step 2 (collapse to "No" vs keep the distinction).
- This function originally tried to summarise `TotalCharges` too and printed **6,531
  levels** — a concrete symptom of the text-vs-number bug. Fixed by excluding the
  numeric-in-meaning columns and capping printed levels.

### 7. Figures — `reports/figures/`
- **`01_target_distribution.png`** — the 73.5 / 26.5 imbalance, visually.
- **`02_churn_by_contract.png`** — the strongest categorical signal in the data:
  **month-to-month ~43% churn**, one-year ~11%, two-year ~3%. Contract length is a
  dominant driver.
- **`03_numeric_by_churn.png`** — churners cluster at **low tenure** and **high monthly
  charges**; non-churners show a second peak at long tenure (~70 months). Both features
  are clearly predictive.
- **`04_correlation_heatmap.png`** — **`TotalCharges` ≈ `tenure` × `MonthlyCharges`**
  (corr **0.83** and **0.65**). Near-redundant. Matters for the logistic-regression
  baseline (multicollinearity inflates coefficient variance); trees are unaffected.

## Decisions carried into Step 2

1. Coerce `TotalCharges` → float; impute the 11 tenure-0 blanks as `0`.
2. Treat `SeniorCitizen` as categorical, not a scaled number.
3. Decide how to handle the "No internet/phone service" third level.
4. Use **stratified** train/test split; plan for class imbalance.
5. Watch `TotalCharges` redundancy for the linear model (candidate to drop or regularise).
6. Headline metrics = AUC-ROC / precision / recall / F1 (not accuracy).
