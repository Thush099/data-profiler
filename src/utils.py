"""I/O, JSON conversion, and terminal summary formatting helpers."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    return pd.read_csv(path)


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path]) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaNs into JSON-safe Python objects."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return obj


def save_report(report: Dict[str, Any], file_path: Union[str, Path]) -> Path:
    """Persist a profiling/validation report dictionary to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_payload = to_json_safe(report)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(clean_payload, fh, indent=2)
        fh.write("\n")
    return path


def format_report_summary(
    profile_dict: Dict[str, Any],
    contract_dict: Optional[Dict[str, Any]] = None,
    remediation_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a concise human-readable terminal summary from pipeline outputs."""
    lines = [
        "=== Data Profiler Report ===",
        (
            f"Rows: {profile_dict['row_count']} | Columns: {profile_dict['column_count']} | "
            f"Duplicates: {profile_dict['duplicate_row_count']} ({profile_dict['duplicate_row_rate']:.1%}) | "
            f"Missing cells: {profile_dict['total_missing_cells']} ({profile_dict['missing_cell_rate']:.1%})"
        ),
    ]
    if profile_dict.get("constant_columns"):
        lines.append(f"Constant columns: {', '.join(profile_dict['constant_columns'])}")
    if profile_dict.get("all_null_columns"):
        lines.append(f"All-null columns: {', '.join(profile_dict['all_null_columns'])}")

    lines.append("")
    lines.append(
        f"{'Column':<18} {'Type':<12} {'Nulls':>7} {'Card':>6} {'IQR Out':>8} {'Z Out':>6} {'Key Stats'}"
    )
    lines.append("-" * 82)
    for col_name, col in profile_dict["columns"].items():
        if col["semantic_type"] == "numeric" and col.get("median") is not None:
            stats = f"med={col['median']:.2f} [{col['min']:.2f}, {col['max']:.2f}]"
        elif col["semantic_type"] == "datetime" and col.get("min_datetime"):
            stats = f"{col['min_datetime'][:10]} .. {col['max_datetime'][:10]}"
        elif col.get("mode") is not None:
            stats = f"mode={col['mode']}"
        else:
            stats = "-"
        null_str = f"{col['null_count']}({col['null_rate']:.0%})"
        lines.append(
            f"{col_name:<18} {col['semantic_type']:<12} {null_str:>7} "
            f"{col['cardinality']:>6} {col['iqr_outlier_count']:>8} "
            f"{col['zscore_outlier_count']:>6} {stats}"
        )

    warnings = profile_dict.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            col_tag = f"[{w['column']}] " if w.get("column") else ""
            lines.append(f"  - ({w['severity'].upper()}) {w['category']}: {col_tag}{w['message']}")

    if contract_dict is not None:
        lines.append("")
        status = "PASSED" if contract_dict["passed"] else "FAILED"
        lines.append(
            f"Contract '{contract_dict['contract_name']}': {status} "
            f"({contract_dict['passed_rules']}/{contract_dict['total_rules_checked']} rules passed)"
        )
        for v in contract_dict.get("violations", []):
            lines.append(f"  - [{v['column']}] {v['rule']}: {v['message']}")

    if remediation_dict is not None:
        lines.append("")
        actions = remediation_dict.get("actions", [])
        lines.append(f"Remediation Plan ({len(actions)} prioritized actions):")
        for act in actions:
            lines.append(
                f"  {act['priority']}. [{act['action']}] {act['target']} "
                f"(affected_rows={act['affected_rows']}): {act['reason']}"
            )

    return "\n".join(lines)
