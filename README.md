# Data Profiler

Automated tabular data quality profiler, declarative schema contract validator, and prioritized remediation planner for CSV and pandas DataFrames.

## Features

- **`DataProfiler`**: Infers semantic column types (`numeric`, `categorical`, `boolean`, `datetime`, `identifier`), computes null counts/rates, cardinality, summary quantiles (`min`, `q25`, `median`, `q75`, `max`, `iqr`), skewness, IQR and z-score outlier counts, exact duplicate rows, constant/all-null columns, and optional `--target` diagnostics (class imbalance and Pearson/conditional-purity target leakage warnings).
- **`DataContract` & `ContractValidator`**: Validates datasets against declarative column rules (`not_null`, `min_value`, `max_value`, `allowed_values`, `unique`, `max_null_rate`), supports JSON serialization, and automatically infers baseline contracts from clean reference DataFrames via `infer_contract(df)`.
- **`RemediationPlanner` & `apply_remediation()`**: Generates prioritized, executable remediation plans (`drop_duplicates`, `quarantine_contract_violations`, `drop_constant_columns`, `impute_median`, `impute_mode`, `winsorize_iqr`) and optionally outputs a remediated CSV.
- **Bundled Sample Dataset**: Includes `data/sample_dirty.csv` with missing values, exact duplicates, negative/extreme numeric anomalies, category typos, and a constant column.

## Project Structure

```text
data-profiler/
├── config.py             # Config thresholds and default remediation mapping
├── main.py               # CLI entry point (defaults to data/sample_dirty.csv)
├── pyproject.toml        # Pytest configuration
├── requirements.txt      # Minimal runtime/test dependencies
├── data/
│   └── sample_dirty.csv  # Realistic dirty tabular sample dataset
├── src/
│   ├── __init__.py       # Public package exports
│   ├── models.py         # Typed dataclasses for profiles, contracts, and plans
│   ├── profiler.py       # DataProfiler implementation
│   ├── contracts.py      # ContractValidator and infer_contract
│   ├── remediation.py    # RemediationPlanner and apply_remediation
│   ├── pipeline.py       # End-to-end DataQualityPipeline orchestration
│   └── utils.py          # CSV/JSON I/O and summary formatting
└── tests/
    ├── test_profiler.py
    ├── test_contracts.py
    ├── test_remediation.py
    └── test_cli.py
```

## Installation

```bash
pip install -r requirements.txt
```

## CLI Usage

Run directly against the bundled `data/sample_dirty.csv` dataset:

```bash
python main.py
```

Full options:

```bash
python main.py \
  --input-data data/sample_dirty.csv \
  --contract path/to/contract.json \
  --target churned \
  --output-report outputs/report.json \
  --clean-output outputs/cleaned.csv
```

### Example CLI Output (`python main.py`)

```text
=== Data Profiler Report ===
Rows: 53 | Columns: 8 | Duplicates: 3 (5.7%) | Missing cells: 10 (2.4%)
Constant columns: source_system

Column             Type           Nulls   Card  IQR Out  Z Out Key Stats
----------------------------------------------------------------------------------
customer_id        identifier     0(0%)     50        0      0 mode=CUST-1003
signup_date        datetime       1(2%)     49        0      0 2024-01-05 .. 2024-09-06
age                numeric        3(6%)     33        2      1 med=45.00 [-5.00, 245.00]
annual_income      numeric        4(8%)     46        2      1 med=75936.39 [-15000.00, 4500000.00]
plan_tier          categorical    2(4%)      5        0      0 mode=Basic
is_active          boolean        0(0%)      2        0      0 mode=True
source_system      categorical    0(0%)      1        0      0 mode=crm_prod
churned            numeric        0(0%)      2        0      0 med=0.00 [0.00, 1.00]

Warnings:
  - (WARNING) duplicate_rows: Found 3 exact duplicate row(s) (5.7%).
  - (WARNING) high_missing_rate: [age] Column 'age' has 3 missing values (5.7%).
  - (WARNING) outliers_detected: [age] Column 'age' has 2 IQR outlier(s) and 1 z-score outlier(s).
  - (WARNING) high_missing_rate: [annual_income] Column 'annual_income' has 4 missing values (7.5%).
  - (WARNING) outliers_detected: [annual_income] Column 'annual_income' has 2 IQR outlier(s) and 1 z-score outlier(s).
  - (INFO) constant_column: [source_system] Column 'source_system' has only 1 distinct non-null value.

Remediation Plan (7 prioritized actions):
  1. [drop_duplicates] __dataset__ (affected_rows=3): Remove 3 exact duplicate row(s).
  2. [drop_constant_columns] source_system (affected_rows=53): Drop zero-information constant/all-null column(s): source_system.
  3. [impute_median] age (affected_rows=3): Impute 3 missing value(s) in numeric column 'age' with median (45).
  4. [winsorize_iqr] age (affected_rows=2): Winsorize 2 IQR outlier(s) in 'age' to [12.12, 79.12].
  5. [impute_median] annual_income (affected_rows=4): Impute 4 missing value(s) in numeric column 'annual_income' with median (7.594e+04).
  6. [winsorize_iqr] annual_income (affected_rows=2): Winsorize 2 IQR outlier(s) in 'annual_income' to [3.667e+04, 1.114e+05].
  7. [impute_mode] plan_tier (affected_rows=2): Impute 2 missing value(s) in categorical column 'plan_tier' with mode ('Basic').
```

## Python API Example

```python
import pandas as pd
from src import (
    ContractValidator,
    DataProfiler,
    RemediationPlanner,
    apply_remediation,
    infer_contract,
)

df = pd.read_csv("data/sample_dirty.csv")
profile = DataProfiler().profile(df, target="churned")
contract = infer_contract(clean_reference_df)
report = ContractValidator().validate(df, contract)
plan = RemediationPlanner().plan(profile, contract_report=report)
result = apply_remediation(df, plan)
```

## Running Tests

```bash
pytest -v
```
