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


class _FakeProfile:
    def __init__(self, severities):
        self.warnings = [type("W", (), {"severity": s})() for s in severities]


class _FakeReport:
    def __init__(self, passed):
        self.passed = passed


class _FakeResult:
    def __init__(self, severities=(), contract_passed=True):
        self.profile = _FakeProfile(severities)
        self.contract_report = _FakeReport(contract_passed)


def test_fail_on_none_always_succeeds():
    from main import exit_code_for

    assert exit_code_for(_FakeResult(severities=["high"]), "none") == 0
    assert exit_code_for(_FakeResult(contract_passed=False), "none") == 0


def test_fail_on_contract_gates_on_contract_violations():
    from main import exit_code_for

    assert exit_code_for(_FakeResult(contract_passed=False), "contract") == 1
    # Warnings alone do not gate at the "contract" level.
    assert exit_code_for(_FakeResult(severities=["high"]), "contract") == 0


def test_fail_on_high_gates_on_high_severity_warnings():
    from main import exit_code_for

    assert exit_code_for(_FakeResult(severities=["high"]), "high") == 1
    assert exit_code_for(_FakeResult(severities=["info", "warning"]), "high") == 0
    assert exit_code_for(_FakeResult(contract_passed=False), "high") == 1


def test_fail_on_any_gates_on_any_warning():
    from main import exit_code_for

    assert exit_code_for(_FakeResult(severities=["info"]), "any") == 1
    assert exit_code_for(_FakeResult(severities=["warning"]), "any") == 1
    assert exit_code_for(_FakeResult(severities=[]), "any") == 0


def test_fail_on_rejects_unknown_level():
    import pytest

    from main import exit_code_for

    with pytest.raises(ValueError, match="fail_on must be one of"):
        exit_code_for(_FakeResult(), "bogus")
