"""Unit tests for I/O, JSON conversion, and report summary helpers."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    format_report_summary,
    load_data,
    save_dataframe,
    save_report,
    to_json_safe,
)


class TestLoadData:
    def test_missing_file_reports_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="absent.csv"):
            load_data(tmp_path / "absent.csv")

    def test_reads_csv_into_dataframe(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n")
        df = load_data(path)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 1


class TestSaveDataframe:
    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "out.csv"
        returned = save_dataframe(pd.DataFrame({"x": [1, 2]}), out)
        assert returned == out
        assert out.exists()
        assert len(pd.read_csv(out)) == 2


class TestToJsonSafe:
    def test_nan_and_inf_become_null(self):
        payload = to_json_safe({"a": float("nan"), "b": np.float64("inf"), "c": 1.5})
        assert payload["a"] is None
        assert payload["b"] is None
        assert payload["c"] == 1.5

    def test_numpy_scalars_become_python_scalars(self):
        payload = to_json_safe({"i": np.int64(3), "b": np.bool_(True), "f": np.float64(0.25)})
        assert payload == {"i": 3, "b": True, "f": 0.25}

    def test_nested_collections_are_converted(self):
        payload = to_json_safe({"xs": [np.int64(1), (np.float64(2.0),)]})
        assert payload == {"xs": [1, [2.0]]}

    def test_result_is_json_serializable(self):
        text = json.dumps(to_json_safe({"n": np.float64("nan"), "t": pd.Timestamp("2020-01-01")}))
        assert json.loads(text)["n"] is None


class TestSaveReport:
    def test_writes_valid_json(self, tmp_path):
        out = tmp_path / "reports" / "profile.json"
        returned = save_report({"rows": 2, "bad": float("nan")}, out)
        assert returned == out
        payload = json.loads(out.read_text())
        assert payload["rows"] == 2
        assert payload["bad"] is None


def _minimal_profile():
    return {
        "row_count": 10,
        "column_count": 1,
        "duplicate_row_count": 0,
        "duplicate_row_rate": 0.0,
        "total_missing_cells": 1,
        "missing_cell_rate": 0.1,
        "constant_columns": [],
        "all_null_columns": [],
        "warnings": [
            {"severity": "info", "category": "imbalance", "column": "y", "message": "skewed"}
        ],
        "columns": {
            "x": {
                "semantic_type": "numeric",
                "null_count": 1,
                "null_rate": 0.1,
                "cardinality": 8,
                "iqr_outlier_count": 0,
                "zscore_outlier_count": 0,
                "median": 5.0,
                "min": 1.0,
                "max": 9.0,
                "mode": None,
            }
        },
    }


class TestFormatReportSummary:
    def test_includes_header_and_column_row(self):
        summary = format_report_summary(_minimal_profile())
        assert "=== Data Profiler Report ===" in summary
        assert "Rows: 10" in summary
        assert "med=5.00 [1.00, 9.00]" in summary
        assert "(INFO) imbalance" in summary

    def test_contract_section_reflects_status(self):
        summary = format_report_summary(
            _minimal_profile(),
            contract_dict={
                "contract_name": "demo",
                "passed": False,
                "passed_rules": 1,
                "total_rules_checked": 2,
                "violations": [
                    {"column": "x", "rule": "not_null", "message": "1 null value"}
                ],
            },
        )
        assert "Contract 'demo': FAILED (1/2 rules passed)" in summary
        assert "[x] not_null: 1 null value" in summary

    def test_remediation_section_lists_actions(self):
        summary = format_report_summary(
            _minimal_profile(),
            remediation_dict={
                "actions": [
                    {
                        "priority": 1,
                        "action": "drop",
                        "target": "x",
                        "affected_rows": 3,
                        "reason": "too many nulls",
                    }
                ]
            },
        )
        assert "Remediation Plan (1 prioritized actions):" in summary
        assert "1. [drop] x (affected_rows=3): too many nulls" in summary
