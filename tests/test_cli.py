"""CLI smoke tests for data-profiler."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from main import main
from src.models import ColumnRule, DataContract

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_cli_default_smoke_run(capsys):
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Data Profiler Report" in captured.out
    assert "source_system" in captured.out
    assert "Remediation Plan" in captured.out


def test_cli_with_contract_target_and_outputs(tmp_path):
    contract = DataContract(
        name="sample_contract",
        rules=[
            ColumnRule(column="customer_id", not_null=True, unique=True),
            ColumnRule(column="age", min_value=18, max_value=100),
            ColumnRule(column="annual_income", min_value=0, max_value=500000),
            ColumnRule(column="plan_tier", allowed_values=["Basic", "Pro", "Enterprise"]),
        ],
    )
    contract_file = tmp_path / "contract.json"
    contract.to_json_file(contract_file)

    report_file = tmp_path / "report.json"
    clean_file = tmp_path / "cleaned.csv"

    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--input-data",
            str(PROJECT_ROOT / "data" / "sample_dirty.csv"),
            "--contract",
            str(contract_file),
            "--target",
            "churned",
            "--output-report",
            str(report_file),
            "--clean-output",
            str(clean_file),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "FAILED" in proc.stdout
    assert "class_imbalance" in proc.stdout
    assert report_file.exists()
    assert clean_file.exists()

    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["profile"]["row_count"] == 53
    assert payload["contract_validation"]["passed"] is False
    assert payload["remediation_execution"]["rows_after"] < 53

    cleaned_df = pd.read_csv(clean_file)
    assert cleaned_df.duplicated().sum() == 0
    assert "source_system" not in cleaned_df.columns
    assert cleaned_df["age"].between(18, 100).all()
