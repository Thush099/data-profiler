"""Tests for RemediationPlanner and apply_remediation."""

from pathlib import Path

import pandas as pd

from src.contracts import ContractValidator
from src.models import ColumnRule, DataContract
from src.profiler import DataProfiler
from src.remediation import RemediationPlanner, apply_remediation


def test_remediation_fixes_nulls_duplicates_constants_and_outliers():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_dirty.csv"
    df = pd.read_csv(sample_path)

    profiler = DataProfiler()
    before_profile = profiler.profile(df, target="churned")
    assert before_profile.duplicate_row_count > 0
    assert before_profile.total_missing_cells > 0
    assert "source_system" in before_profile.constant_columns
    assert before_profile.columns["age"].iqr_outlier_count > 0
    assert before_profile.columns["annual_income"].iqr_outlier_count > 0

    plan = RemediationPlanner().plan(before_profile)
    action_names = [a.action for a in plan.actions]
    assert "drop_duplicates" in action_names
    assert "drop_constant_columns" in action_names
    assert "impute_median" in action_names
    assert "impute_mode" in action_names
    assert "winsorize_iqr" in action_names

    result = apply_remediation(df, plan)
    cleaned = result.cleaned_df
    after_profile = profiler.profile(cleaned, target="churned")

    assert after_profile.duplicate_row_count == 0
    assert "source_system" not in cleaned.columns
    assert after_profile.columns["age"].null_count == 0
    assert after_profile.columns["annual_income"].null_count == 0
    assert after_profile.columns["plan_tier"].null_count == 0
    assert after_profile.columns["age"].iqr_outlier_count < before_profile.columns["age"].iqr_outlier_count
    assert (
        after_profile.columns["annual_income"].iqr_outlier_count
        < before_profile.columns["annual_income"].iqr_outlier_count
    )
    assert cleaned["age"].min() >= 0
    assert cleaned["age"].max() <= 100


def test_remediation_quarantines_contract_violations():
    df = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-2", "C-3", "C-4", "C-1"],
            "age": [25.0, 35.0, -5.0, 45.0, 25.0],
            "tier": ["Basic", "Pro", "Basic", "TypoTier", "Basic"],
            "env": ["prod"] * 5,
        }
    )
    contract = DataContract(
        name="customer_rules",
        rules=[
            ColumnRule(column="customer_id", not_null=True, unique=True),
            ColumnRule(column="age", not_null=True, min_value=18, max_value=100),
            ColumnRule(column="tier", not_null=True, allowed_values=["Basic", "Pro", "Enterprise"]),
        ],
    )

    profiler = DataProfiler()
    profile = profiler.profile(df)
    report = ContractValidator().validate(df, contract)
    assert report.passed is False

    plan = RemediationPlanner().plan(profile, contract_report=report)
    assert any(a.action == "quarantine_contract_violations" for a in plan.actions)

    result = apply_remediation(df, plan)
    assert result.rows_quarantined == 3
    assert result.rows_after == 2
    assert "env" not in result.cleaned_df.columns
    assert ContractValidator().validate(result.cleaned_df, contract).passed is True


def test_remediation_quarantines_only_violating_row_with_duplicate_index_labels():
    df = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-1", "C-2", "C-2"],
            "age": [25, 25, 35, 45],
        },
        index=[7, 7, 7, 7],
    )
    contract = DataContract(
        name="unique_customer_ids",
        rules=[ColumnRule(column="customer_id", unique=True)],
    )

    report = ContractValidator().validate(df, contract)
    result = apply_remediation(
        df,
        RemediationPlanner().plan(DataProfiler().profile(df), contract_report=report),
    )

    assert report.violating_row_positions == [1, 3]
    assert result.cleaned_df["age"].tolist() == [25, 35]
    assert result.quarantined_df["age"].tolist() == [25, 45]
    assert result.rows_quarantined == 2


def test_contract_violation_is_quarantined_before_duplicate_removal():
    df = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-1", "C-2"],
            "age": [25, 25, 35],
        },
        index=[10, 10, 12],
    )
    contract = DataContract(
        name="unique_customer_ids",
        rules=[ColumnRule(column="customer_id", unique=True)],
    )

    profile = DataProfiler().profile(df)
    report = ContractValidator().validate(df, contract)
    assert report.violating_row_positions == [1]

    result = apply_remediation(
        df,
        RemediationPlanner().plan(profile, contract_report=report),
    )

    assert result.rows_quarantined == 1
    assert len(result.quarantined_df) == 1
    assert result.quarantined_df["customer_id"].tolist() == ["C-1"]
    assert result.rows_after == 2
    assert result.cleaned_df["customer_id"].tolist() == ["C-1", "C-2"]


def test_remediation_handles_empty_and_all_null_columns():
    empty_df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    empty_profile = DataProfiler().profile(empty_df)
    empty_plan = RemediationPlanner().plan(empty_profile)
    empty_result = apply_remediation(empty_df, empty_plan)
    assert empty_result.rows_after == 0

    df_all_null = pd.DataFrame(
        {
            "dead_col": [None, None, None],
            "val": [1.0, 2.0, 3.0],
        }
    )
    null_profile = DataProfiler().profile(df_all_null)
    null_plan = RemediationPlanner().plan(null_profile)
    null_result = apply_remediation(df_all_null, null_plan)
    assert "dead_col" not in null_result.cleaned_df.columns
    assert list(null_result.cleaned_df.columns) == ["val"]


def test_quarantine_by_explicit_row_indices() -> None:
    """The row_indices quarantine path removes only valid listed rows and reports
    nothing applied when no index is valid."""
    from src.models import RemediationAction, RemediationPlan

    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    plan = RemediationPlan(actions=[RemediationAction(
        priority=1, action="quarantine_contract_violations", target="a",
        reason="contract violation", params={"row_indices": [1, 99]},
    )])
    result = apply_remediation(df, plan)
    assert list(result.cleaned_df["a"]) == [1, 3, 4], "valid violating rows are removed"
    assert list(result.quarantined_df["a"]) == [2], "the valid listed row is quarantined"
    assert any("quarantine_contract_violations" in a for a in result.applied_actions)

    empty_plan = RemediationPlan(actions=[RemediationAction(
        priority=1, action="quarantine_contract_violations", target="a",
        reason="contract violation", params={"row_indices": [99]},
    )])
    untouched = apply_remediation(df, empty_plan)
    assert list(untouched.cleaned_df["a"]) == [1, 2, 3, 4], "invalid indices quarantine nothing"
    assert not any("quarantine_contract_violations" in a for a in untouched.applied_actions)
