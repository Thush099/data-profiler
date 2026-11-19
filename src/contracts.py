"""Declarative data contract validation and automatic contract inference."""

from typing import List, Optional, Set

import pandas as pd

from config import Config
from src.models import (
    ColumnRule,
    ContractReport,
    ContractViolation,
    DataContract,
)
from src.profiler import DataProfiler


class ContractValidator:
    """Validates a DataFrame against declarative column rules in a DataContract."""

    def validate(self, df: pd.DataFrame, contract: DataContract) -> ContractReport:
        """Evaluate every rule in the contract against the supplied DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")

        if not df.columns.is_unique:
            duplicates = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"DataFrame contains duplicate column labels: {duplicates!r}.")

        violations: List[ContractViolation] = []
        violating_positions: Set[int] = set()
        total_rules_checked = 0
        row_count = len(df)

        if contract.strict_columns:
            expected_cols = set(contract.rules.keys())
            unexpected_cols = sorted(set(df.columns) - expected_cols)
            total_rules_checked += 1
            if unexpected_cols:
                violations.append(
                    ContractViolation(
                        column="__dataset__",
                        rule="strict_columns",
                        failed_count=len(unexpected_cols),
                        failure_rate=1.0 if row_count > 0 else 0.0,
                        message=f"Unexpected columns present: {', '.join(unexpected_cols)}.",
                        details={"unexpected_columns": unexpected_cols},
                    )
                )

        assert isinstance(contract.rules, dict)
        for col_name, rule in contract.rules.items():
            total_rules_checked += 1
            if col_name not in df.columns:
                violations.append(
                    ContractViolation(
                        column=col_name,
                        rule="column_exists",
                        failed_count=row_count,
                        failure_rate=1.0 if row_count > 0 else 0.0,
                        message=f"Required column '{col_name}' is missing from dataset.",
                    )
                )
                continue

            # Undo initial column-presence count so each individual enabled constraint is counted accurately
            total_rules_checked -= 1
            series = df[col_name]

            if rule.not_null:
                total_rules_checked += 1
                mask = series.isna()
                failed_count = int(mask.sum())
                if failed_count > 0:
                    idx = df.index[mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(mask.to_numpy()) if failed
                    )
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="not_null",
                            failed_count=failed_count,
                            failure_rate=failed_count / row_count,
                            message=f"Column '{col_name}' has {failed_count} null value(s) violating not_null.",
                            sample_indices=idx[:10],
                        )
                    )

            if rule.max_null_rate is not None:
                total_rules_checked += 1
                null_rate = float(series.isna().mean()) if row_count > 0 else 0.0
                if null_rate > rule.max_null_rate:
                    null_mask = series.isna()
                    failed_count = int(null_mask.sum())
                    idx = df.index[null_mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(null_mask.to_numpy()) if failed
                    )
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="max_null_rate",
                            failed_count=failed_count,
                            failure_rate=null_rate,
                            message=(
                                f"Column '{col_name}' null rate {null_rate:.1%} exceeds "
                                f"max_null_rate {rule.max_null_rate:.1%}."
                            ),
                            sample_indices=idx[:10],
                            details={"null_rate": round(null_rate, 4), "max_null_rate": rule.max_null_rate},
                        )
                    )

            if rule.min_value is not None:
                total_rules_checked += 1
                non_null_mask = series.notna()
                numeric_vals = pd.to_numeric(series[non_null_mask], errors="coerce")
                bad_mask = pd.Series(False, index=df.index)
                bad_mask.loc[non_null_mask] = numeric_vals.isna() | (numeric_vals < rule.min_value)
                failed_count = int(bad_mask.sum())
                if failed_count > 0:
                    idx = df.index[bad_mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(bad_mask.to_numpy()) if failed
                    )
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="min_value",
                            failed_count=failed_count,
                            failure_rate=failed_count / row_count,
                            message=(
                                f"Column '{col_name}' has {failed_count} value(s) below "
                                f"min_value={rule.min_value}."
                            ),
                            sample_indices=idx[:10],
                        )
                    )

            if rule.max_value is not None:
                total_rules_checked += 1
                non_null_mask = series.notna()
                numeric_vals = pd.to_numeric(series[non_null_mask], errors="coerce")
                bad_mask = pd.Series(False, index=df.index)
                bad_mask.loc[non_null_mask] = numeric_vals.isna() | (numeric_vals > rule.max_value)
                failed_count = int(bad_mask.sum())
                if failed_count > 0:
                    idx = df.index[bad_mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(bad_mask.to_numpy()) if failed
                    )
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="max_value",
                            failed_count=failed_count,
                            failure_rate=failed_count / row_count,
                            message=(
                                f"Column '{col_name}' has {failed_count} value(s) above "
                                f"max_value={rule.max_value}."
                            ),
                            sample_indices=idx[:10],
                        )
                    )

            if rule.allowed_values is not None:
                total_rules_checked += 1
                allowed_set = set(rule.allowed_values)
                non_null_mask = series.notna()
                bad_mask = non_null_mask & (~series.isin(allowed_set))
                failed_count = int(bad_mask.sum())
                if failed_count > 0:
                    idx = df.index[bad_mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(bad_mask.to_numpy()) if failed
                    )
                    unexpected = sorted({str(v) for v in series[bad_mask].unique()})
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="allowed_values",
                            failed_count=failed_count,
                            failure_rate=failed_count / row_count,
                            message=(
                                f"Column '{col_name}' has {failed_count} disallowed value(s): "
                                f"{', '.join(unexpected[:5])}."
                            ),
                            sample_indices=idx[:10],
                            details={"unexpected_values": unexpected[:10]},
                        )
                    )

            if rule.unique:
                total_rules_checked += 1
                non_null_mask = series.notna()
                dup_mask = pd.Series(False, index=df.index)
                dup_mask.loc[non_null_mask] = series[non_null_mask].duplicated(keep="first")
                failed_count = int(dup_mask.sum())
                if failed_count > 0:
                    idx = df.index[dup_mask].tolist()
                    violating_positions.update(
                        i for i, failed in enumerate(dup_mask.to_numpy()) if failed
                    )
                    violations.append(
                        ContractViolation(
                            column=col_name,
                            rule="unique",
                            failed_count=failed_count,
                            failure_rate=failed_count / row_count,
                            message=f"Column '{col_name}' has {failed_count} duplicate non-null value(s).",
                            sample_indices=idx[:10],
                        )
                    )

        failed_rules = len(violations)
        passed_rules = max(0, total_rules_checked - failed_rules)
        return ContractReport(
            contract_name=contract.name,
            passed=failed_rules == 0,
            total_rules_checked=total_rules_checked,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            violations=violations,
            violating_row_indices=list(
                dict.fromkeys(df.index[position] for position in sorted(violating_positions))
            ),
            violating_row_positions=sorted(violating_positions),
        )


def infer_contract(
    df: pd.DataFrame,
    name: str = "inferred_contract",
    max_categorical_values: int = 50,
    numeric_margin: float = 0.0,
    config: Optional[Config] = None,
) -> DataContract:
    """Infer a baseline DataContract from clean reference DataFrame statistics."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")
    if numeric_margin < 0:
        raise ValueError("numeric_margin must be non-negative.")

    profiler = DataProfiler(config=config)
    profile = profiler.profile(df)
    rules: dict[str, ColumnRule] = {}

    if df.empty:
        return DataContract(name=name, rules=rules)

    for col_name, col_prof in profile.columns.items():
        not_null = col_prof.null_count == 0
        max_null_rate = 0.0 if not_null else min(1.0, round(col_prof.null_rate * 1.5 + 0.01, 4))

        min_value: Optional[float] = None
        max_value: Optional[float] = None
        allowed_values: Optional[List[object]] = None
        unique = False

        if col_prof.semantic_type == "identifier" and col_prof.non_null_count > 0:
            if col_prof.uniqueness_ratio == 1.0:
                unique = True

        elif col_prof.semantic_type == "numeric" and col_prof.min is not None and col_prof.max is not None:
            span = col_prof.max - col_prof.min
            pad = span * numeric_margin
            min_value = float(col_prof.min - pad)
            max_value = float(col_prof.max + pad)

        elif col_prof.semantic_type in ("categorical", "boolean") and col_prof.non_null_count > 0:
            if 1 <= col_prof.cardinality <= max_categorical_values:
                raw_vals = df[col_name].dropna().unique().tolist()
                allowed_values = sorted(
                    [DataProfiler._to_python_scalar(v) for v in raw_vals],
                    key=str,
                )

        rules[col_name] = ColumnRule(
            column=col_name,
            not_null=not_null,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            unique=unique,
            max_null_rate=max_null_rate,
        )

    return DataContract(name=name, rules=rules)
