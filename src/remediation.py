"""Prioritized data remediation planner and deterministic cleanup execution."""

from typing import Any, Dict, List, Optional

import pandas as pd

from config import Config
from src.models import (
    ContractReport,
    DatasetProfile,
    RemediationAction,
    RemediationPlan,
    RemediationResult,
)


class RemediationPlanner:
    """Builds prioritized remediation actions from dataset profiles and contract violations."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def plan(
        self,
        profile: DatasetProfile,
        contract_report: Optional[ContractReport] = None,
    ) -> RemediationPlan:
        """Generate ordered remediation actions based on detected quality issues."""
        actions: List[RemediationAction] = []
        priority = 1

        if profile.duplicate_row_count > 0:
            actions.append(
                RemediationAction(
                    priority=priority,
                    action="drop_duplicates",
                    target="__dataset__",
                    reason=f"Remove {profile.duplicate_row_count} exact duplicate row(s).",
                    affected_rows=profile.duplicate_row_count,
                )
            )
            priority += 1

        if contract_report is not None and contract_report.violating_row_indices:
            violating_indices = list(contract_report.violating_row_indices)
            actions.append(
                RemediationAction(
                    priority=priority,
                    action="quarantine_contract_violations",
                    target="__dataset__",
                    reason=(
                        f"Quarantine {len(violating_indices)} row(s) violating "
                        f"contract '{contract_report.contract_name}'."
                    ),
                    affected_rows=len(violating_indices),
                    params={"row_indices": violating_indices},
                )
            )
            priority += 1

        constant_and_empty = [
            col
            for col in (profile.all_null_columns + profile.constant_columns)
            if col != profile.target_column
        ]
        # Preserve order while deduplicating
        cols_to_drop = list(dict.fromkeys(constant_and_empty))
        if cols_to_drop:
            actions.append(
                RemediationAction(
                    priority=priority,
                    action="drop_constant_columns",
                    target=", ".join(cols_to_drop),
                    reason=f"Drop zero-information constant/all-null column(s): {', '.join(cols_to_drop)}.",
                    affected_rows=profile.row_count,
                    params={"columns": cols_to_drop},
                )
            )
            priority += 1

        for col_name, col_prof in profile.columns.items():
            if col_name in cols_to_drop or col_name == profile.target_column:
                continue

            if 0 < col_prof.null_count < col_prof.count:
                if col_prof.semantic_type == "numeric" and col_prof.median is not None:
                    actions.append(
                        RemediationAction(
                            priority=priority,
                            action="impute_median",
                            target=col_name,
                            reason=(
                                f"Impute {col_prof.null_count} missing value(s) in numeric column "
                                f"'{col_name}' with median ({col_prof.median:.4g})."
                            ),
                            affected_rows=col_prof.null_count,
                            params={"fill_value": col_prof.median},
                        )
                    )
                    priority += 1
                elif col_prof.semantic_type in ("categorical", "boolean") and col_prof.mode is not None:
                    actions.append(
                        RemediationAction(
                            priority=priority,
                            action="impute_mode",
                            target=col_name,
                            reason=(
                                f"Impute {col_prof.null_count} missing value(s) in {col_prof.semantic_type} "
                                f"column '{col_name}' with mode ({col_prof.mode!r})."
                            ),
                            affected_rows=col_prof.null_count,
                            params={"fill_value": col_prof.mode},
                        )
                    )
                    priority += 1

            if (
                col_prof.semantic_type == "numeric"
                and col_prof.iqr_outlier_count > 0
                and col_prof.iqr is not None
                and col_prof.iqr > 0
                and col_prof.q25 is not None
                and col_prof.q75 is not None
            ):
                lower = col_prof.q25 - self.config.iqr_multiplier * col_prof.iqr
                upper = col_prof.q75 + self.config.iqr_multiplier * col_prof.iqr
                actions.append(
                    RemediationAction(
                        priority=priority,
                        action="winsorize_iqr",
                        target=col_name,
                        reason=(
                            f"Winsorize {col_prof.iqr_outlier_count} IQR outlier(s) in '{col_name}' "
                            f"to [{lower:.4g}, {upper:.4g}]."
                        ),
                        affected_rows=col_prof.iqr_outlier_count,
                        params={
                            "lower_bound": float(lower),
                            "upper_bound": float(upper),
                            "iqr_multiplier": float(self.config.iqr_multiplier),
                        },
                    )
                )
                priority += 1

        return RemediationPlan(actions=actions)


def apply_remediation(df: pd.DataFrame, plan: RemediationPlan) -> RemediationResult:
    """Execute a RemediationPlan deterministically and return cleaned + quarantined DataFrames."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")

    rows_before = int(len(df))
    working = df.copy()
    quarantined_df = df.iloc[0:0].copy()
    applied_actions: List[str] = []
    columns_dropped: List[str] = []

    for act in sorted(plan.actions, key=lambda item: item.priority):
        if act.action == "drop_duplicates":
            before = len(working)
            working = working.drop_duplicates(keep="first")
            if len(working) < before:
                applied_actions.append(f"drop_duplicates ({before - len(working)} rows)")

        elif act.action == "quarantine_contract_violations":
            raw_indices = act.params.get("row_indices", [])
            valid_idx = [idx for idx in raw_indices if idx in working.index]
            if valid_idx:
                quarantined_df = working.loc[valid_idx].copy()
                working = working.drop(index=valid_idx)
                applied_actions.append(
                    f"quarantine_contract_violations ({len(valid_idx)} rows)"
                )

        elif act.action == "drop_constant_columns":
            cols = [c for c in act.params.get("columns", []) if c in working.columns]
            if cols:
                working = working.drop(columns=cols)
                if not quarantined_df.empty:
                    quarantined_df = quarantined_df.drop(columns=[c for c in cols if c in quarantined_df.columns])
                columns_dropped.extend(cols)
                applied_actions.append(f"drop_constant_columns ({', '.join(cols)})")

        elif act.action == "impute_median":
            col = act.target
            if col in working.columns and working[col].isna().any():
                non_null = pd.to_numeric(working[col], errors="coerce").dropna()
                fill_val: Any = (
                    float(non_null.median())
                    if not non_null.empty
                    else act.params.get("fill_value")
                )
                if fill_val is not None and not pd.isna(fill_val):
                    numeric_series = pd.to_numeric(working[col], errors="coerce")
                    working[col] = numeric_series.fillna(float(fill_val))
                    applied_actions.append(f"impute_median:{col}")

        elif act.action == "impute_mode":
            col = act.target
            if col in working.columns and working[col].isna().any():
                modes = working[col].dropna().mode()
                fill_val = modes.iloc[0] if not modes.empty else act.params.get("fill_value")
                if fill_val is not None and not pd.isna(fill_val):
                    working[col] = working[col].fillna(fill_val)
                    applied_actions.append(f"impute_mode:{col}")

        elif act.action == "winsorize_iqr":
            col = act.target
            if col in working.columns:
                numeric_series = pd.to_numeric(working[col], errors="coerce").astype(float)
                non_null = numeric_series.dropna()
                if not non_null.empty:
                    q25 = float(non_null.quantile(0.25))
                    q75 = float(non_null.quantile(0.75))
                    iqr = q75 - q25
                    iqr_mult = float(act.params.get("iqr_multiplier", 1.5))
                    if iqr > 0:
                        lower = q25 - iqr_mult * iqr
                        upper = q75 + iqr_mult * iqr
                    else:
                        lower = act.params.get("lower_bound")
                        upper = act.params.get("upper_bound")
                    if lower is not None and upper is not None:
                        working[col] = numeric_series.clip(lower=float(lower), upper=float(upper))
                        applied_actions.append(f"winsorize_iqr:{col}")

    working = working.reset_index(drop=True)
    quarantined_df = quarantined_df.reset_index(drop=True)

    return RemediationResult(
        cleaned_df=working,
        quarantined_df=quarantined_df,
        applied_actions=applied_actions,
        rows_before=rows_before,
        rows_after=int(len(working)),
        rows_quarantined=int(len(quarantined_df)),
        columns_dropped=columns_dropped,
    )
