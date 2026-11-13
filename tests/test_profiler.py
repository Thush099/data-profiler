"""Tests for DataProfiler semantic types, statistics, target diagnostics, and edge cases."""

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.profiler import DataProfiler


def test_semantic_type_inference_all_five_types():
    df = pd.DataFrame(
        {
            "user_id": [f"USR-{i:03d}" for i in range(10)],
            "signup_date": [f"2025-01-{i + 1:02d}" for i in range(10)],
            "age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "plan": ["Basic", "Pro", "Basic", "Enterprise", "Pro", "Basic", "Pro", "Basic", "Enterprise", "Pro"],
            "is_active": [True, False, True, None, True, False, True, True, False, True],
            "subscribed_str": ["yes", "no", "yes", "yes", "no", "no", "yes", "no", "yes", "no"],
            "is_verified": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    profile = DataProfiler().profile(df)

    assert profile.columns["user_id"].semantic_type == "identifier"
    assert profile.columns["signup_date"].semantic_type == "datetime"
    assert profile.columns["age"].semantic_type == "numeric"
    assert profile.columns["plan"].semantic_type == "categorical"
    assert profile.columns["is_active"].semantic_type == "boolean"
    assert profile.columns["subscribed_str"].semantic_type == "boolean"
    assert profile.columns["is_verified"].semantic_type == "boolean"


def test_profiling_statistics_quantiles_outliers_duplicates_constants():
    df = pd.DataFrame(
        {
            "id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "A"],
            "score": [10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.5, 12.0, 11.0, 250.0, 10.0],
            "region": ["NA", "EU", "NA", None, "APAC", "EU", "NA", "NA", "EU", "APAC", "NA"],
            "env": ["prod"] * 11,
        }
    )
    profile = DataProfiler().profile(df)

    assert profile.row_count == 11
    assert profile.column_count == 4
    assert profile.duplicate_row_count == 1
    assert pytest.approx(profile.duplicate_row_rate, rel=1e-4) == 1 / 11
    assert profile.constant_columns == ["env"]

    score_prof = profile.columns["score"]
    assert score_prof.null_count == 0
    assert score_prof.q25 is not None and score_prof.median is not None and score_prof.q75 is not None
    assert score_prof.iqr == pytest.approx(score_prof.q75 - score_prof.q25)
    assert score_prof.skewness is not None and score_prof.skewness > 1.0
    assert score_prof.iqr_outlier_count >= 1
    assert score_prof.zscore_outlier_count == 1

    region_prof = profile.columns["region"]
    assert region_prof.null_count == 1
    assert region_prof.null_rate == pytest.approx(1 / 11)
    assert region_prof.cardinality == 3
    assert region_prof.mode == "NA"


def test_target_imbalance_and_leakage_warnings():
    rng = np.random.default_rng(7)
    n = 100
    target = np.array([1] * 8 + [0] * 92)
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i}" for i in range(n)],
            "safe_feature": rng.normal(0, 1, size=n),
            "leaked_numeric": target * 50.0 + rng.normal(0, 0.1, size=n),
            "leaked_status": ["churn_flag" if t == 1 else "ok_flag" for t in target],
            "churned": target,
        }
    )

    profile = DataProfiler(Config()).profile(df, target="churned")
    warning_pairs = {(w.category, w.column) for w in profile.warnings}

    assert ("class_imbalance", "churned") in warning_pairs
    assert ("target_leakage", "leaked_numeric") in warning_pairs
    assert ("target_leakage", "leaked_status") in warning_pairs
    assert ("target_leakage", "safe_feature") not in warning_pairs
    assert ("target_leakage", "customer_id") not in warning_pairs


def test_balanced_target_no_imbalance_warning():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, size=40),
            "y": [0, 1] * 20,
        }
    )
    profile = DataProfiler().profile(df, target="y")
    categories = {w.category for w in profile.warnings}
    assert "class_imbalance" not in categories
    assert "target_leakage" not in categories


def test_empty_dataframe_and_all_null_column_edge_cases():
    empty_df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
    empty_prof = DataProfiler().profile(empty_df)
    assert empty_prof.row_count == 0
    assert empty_prof.column_count == 2
    assert empty_prof.duplicate_row_count == 0
    assert empty_prof.missing_cell_rate == 0.0
    assert any(w.category == "empty_dataset" for w in empty_prof.warnings)
    assert empty_prof.columns["a"].median is None
    assert empty_prof.columns["a"].iqr_outlier_count == 0

    all_null_df = pd.DataFrame(
        {
            "empty_col": [None, None, None, None],
            "num_col": [10.0, 20.0, 30.0, 40.0],
        }
    )
    null_prof = DataProfiler().profile(all_null_df)
    assert null_prof.all_null_columns == ["empty_col"]
    assert null_prof.columns["empty_col"].is_all_null is True
    assert null_prof.columns["empty_col"].null_rate == 1.0
    assert null_prof.columns["empty_col"].cardinality == 0
    assert any(
        w.category == "all_null_column" and w.column == "empty_col"
        for w in null_prof.warnings
    )


def test_invalid_target_raises_value_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column"):
        DataProfiler().profile(df, target="missing_col")
