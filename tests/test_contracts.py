"""Tests for declarative DataContract validation and contract inference."""

import json

import pandas as pd
import pytest

from src.contracts import ContractValidator, infer_contract
from src.models import ColumnRule, DataContract


def test_max_null_rate_reports_row_positions_with_duplicate_index_labels():
    df = pd.DataFrame(
        {"note": [None, "ok", None]},
        index=[7, 7, 7],
    )
    contract = DataContract(
        name="note_null_limit",
        rules=[ColumnRule(column="note", max_null_rate=0.5)],
    )

    report = ContractValidator().validate(df, contract)

    assert report.violating_row_indices == [7]
    assert report.violating_row_positions == [0, 2]
    assert report.violations[0].sample_indices == [7, 7]


def test_contract_validation_preserves_non_integer_index_labels():
    df = pd.DataFrame(
        {"age": [None, 42]},
        index=["customer-a", "customer-b"],
    )
    contract = DataContract(
        rules=[ColumnRule(column="age", not_null=True)],
    )

    report = ContractValidator().validate(df, contract)

    assert report.violating_row_indices == ["customer-a"]
    assert report.violating_row_positions == [0]
    assert report.violations[0].sample_indices == ["customer-a"]
    assert report.to_dict()["violating_row_indices"] == ["customer-a"]


@pytest.mark.parametrize(
    ("values", "rule", "expected_positions"),
    [
        ([1, 5, 10], ColumnRule(column="value", min_value=5), [0]),
        ([1, 5, 10], ColumnRule(column="value", max_value=5), [2]),
        (["bad", "ok", "bad"], ColumnRule(column="value", allowed_values=["ok"]), [0, 2]),
    ],
)
def test_contract_rules_report_positions_with_duplicate_index_labels(
    values, rule, expected_positions
):
    df = pd.DataFrame({"value": values}, index=[7, 7, 7])

    report = ContractValidator().validate(df, DataContract(rules=[rule]))

    assert report.violating_row_positions == expected_positions
    assert report.violating_row_indices == [7]


def test_contract_validation_rejects_duplicate_column_labels():
    df = pd.DataFrame([[1, 2]], columns=["age", "age"])
    contract = DataContract(rules=[ColumnRule(column="age", min_value=0)])

    with pytest.raises(ValueError, match="duplicate column labels"):
        ContractValidator().validate(df, contract)


def test_contract_validation_passes_on_clean_data():
    df = pd.DataFrame(
        {
            "id": ["ID-1", "ID-2", "ID-3"],
            "age": [25, 34, 48],
            "tier": ["Basic", "Pro", "Enterprise"],
            "notes": ["ok", None, "ok"],
        }
    )
    contract = DataContract(
        name="clean_schema",
        rules=[
            ColumnRule(column="id", not_null=True, unique=True),
            ColumnRule(column="age", not_null=True, min_value=18, max_value=100),
            ColumnRule(column="tier", not_null=True, allowed_values=["Basic", "Pro", "Enterprise"]),
            ColumnRule(column="notes", max_null_rate=0.5),
        ],
    )

    report = ContractValidator().validate(df, contract)
    assert report.passed is True
    assert report.failed_rules == 0
    assert report.violations == []
    assert report.violating_row_indices == []


def test_contract_validation_reports_all_rule_violations():
    df = pd.DataFrame(
        {
            "id": ["ID-1", "ID-1", None],
            "age": [15, 30, 150],
            "tier": ["Basic", "TypoTier", "Pro"],
            "nullable": [None, None, "ok"],
        }
    )
    contract = DataContract(
        name="strict_schema",
        strict_columns=True,
        rules={
            "id": ColumnRule(column="id", not_null=True, unique=True),
            "age": ColumnRule(column="age", min_value=18, max_value=100),
            "tier": ColumnRule(column="tier", allowed_values=["Basic", "Pro", "Enterprise"]),
            "nullable": ColumnRule(column="nullable", max_null_rate=0.4),
            "missing_req": ColumnRule(column="missing_req", not_null=True),
        },
    )

    report = ContractValidator().validate(df, contract)
    assert report.passed is False
    violated_rules = {(v.column, v.rule) for v in report.violations}

    assert ("id", "not_null") in violated_rules
    assert ("id", "unique") in violated_rules
    assert ("age", "min_value") in violated_rules
    assert ("age", "max_value") in violated_rules
    assert ("tier", "allowed_values") in violated_rules
    assert ("nullable", "max_null_rate") in violated_rules
    assert ("missing_req", "column_exists") in violated_rules
    assert report.violating_row_indices == [0, 1, 2]


def test_infer_contract_and_json_roundtrip(tmp_path):
    clean_df = pd.DataFrame(
        {
            "customer_id": [f"C-{i}" for i in range(1, 7)],
            "age": [22, 28, 35, 41, 52, 60],
            "tier": ["Basic", "Pro", "Basic", "Enterprise", "Pro", "Basic"],
            "active": [True, True, False, True, False, True],
        }
    )

    contract = infer_contract(clean_df, name="baseline_v1")
    validator = ContractValidator()

    assert validator.validate(clean_df, contract).passed is True

    contract_path = tmp_path / "contract.json"
    contract.to_json_file(contract_path)
    reloaded = DataContract.from_json_file(contract_path)
    assert reloaded.to_dict() == contract.to_dict()

    dirty_df = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-1", "C-3"],
            "age": [22, -5, 190],
            "tier": ["Basic", "Entrprise", None],
            "active": [True, False, True],
        }
    )
    dirty_report = validator.validate(dirty_df, reloaded)
    assert dirty_report.passed is False
    violated_pairs = {(v.column, v.rule) for v in dirty_report.violations}
    assert ("customer_id", "unique") in violated_pairs
    assert ("age", "min_value") in violated_pairs
    assert ("age", "max_value") in violated_pairs
    assert ("tier", "allowed_values") in violated_pairs
    assert ("tier", "not_null") in violated_pairs


def test_infer_contract_empty_and_rule_validation_errors():
    empty_df = pd.DataFrame({"a": pd.Series(dtype="float64")})
    empty_contract = infer_contract(empty_df)
    assert empty_contract.rules == {}
    assert ContractValidator().validate(empty_df, empty_contract).passed is True

    with pytest.raises(ValueError, match="min_value"):
        ColumnRule(column="x", min_value=10, max_value=5)
    with pytest.raises(ValueError, match="max_null_rate"):
        ColumnRule(column="x", max_null_rate=1.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_column_rule_rejects_non_finite_bounds(bad: float) -> None:
    with pytest.raises(ValueError, match="min_value must be finite"):
        ColumnRule(column="x", min_value=bad)
    with pytest.raises(ValueError, match="max_value must be finite"):
        ColumnRule(column="x", max_value=bad)


@pytest.mark.parametrize(
    "margin", [float("nan"), float("inf"), float("-inf"), -0.1, True, "0.5"]
)
def test_infer_contract_rejects_invalid_numeric_margin(margin: object) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="numeric_margin must be a finite non-negative real"):
        infer_contract(df, numeric_margin=margin)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, float("inf"), 3.0],
        [1.0, 3.0, float("-inf")],
        [float("inf"), float("-inf"), 1.0],
    ],
)
def test_infer_contract_rejects_non_finite_reference_values(values: list) -> None:
    """Non-finite reference values must not silently produce NaN bounds."""
    df = pd.DataFrame({"a": values})
    with pytest.raises(ValueError, match="column 'a' contains non-finite values"):
        infer_contract(df)


def test_infer_contract_json_contains_no_non_finite_tokens() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    payload = infer_contract(df).to_json()

    json.loads(payload, parse_constant=lambda name: pytest.fail(f"non-finite token: {name}"))
    assert not any(tok in payload for tok in ("NaN", "Infinity"))


class TestContractFileLoading:
    def test_missing_contract_file_reports_path(self, tmp_path):
        from src.models import DataContract

        with pytest.raises(FileNotFoundError, match="Contract file not found"):
            DataContract.from_json_file(tmp_path / "missing.json")

    def test_malformed_contract_json_reports_path(self, tmp_path):
        from src.models import DataContract

        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Contract file is not valid JSON"):
            DataContract.from_json_file(bad)


def test_infer_contract_rejects_non_finite_before_pandas_number_crunching() -> None:
    """Non-finite reference values must be rejected at the boundary — pandas must
    never see them (its subtract on inf emitted RuntimeWarnings during profiling)."""
    import warnings

    df = pd.DataFrame({"a": [1.0, 2.0, float("inf")]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="column 'a' contains non-finite values"):
            infer_contract(df)
