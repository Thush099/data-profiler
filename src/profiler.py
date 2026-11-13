"""Tabular data profiler with semantic type inference and quality diagnostics."""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from src.models import ColumnProfile, DatasetProfile, QualityWarning

_BOOL_STRINGS = {"true", "false", "yes", "no", "t", "f", "y", "n"}
_DATE_HINT_RE = re.compile(r"[-/:T]|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE)
_ID_SUFFIXES = ("_id", "_uuid", "_guid", "_key", "_code")
_ID_PREFIXES = ("id_", "uuid_", "guid_")


class DataProfiler:
    """Profiles DataFrames for schema types, statistics, anomalies, and target warnings."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def profile(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> DatasetProfile:
        """Compute dataset-level and column-level data quality profiles."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}.")
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        row_count = int(len(df))
        column_count = int(df.shape[1])
        duplicate_row_count = int(df.duplicated().sum()) if row_count > 0 else 0
        duplicate_row_rate = float(duplicate_row_count / row_count) if row_count > 0 else 0.0

        total_cells = row_count * column_count
        total_missing_cells = int(df.isna().sum().sum()) if total_cells > 0 else 0
        missing_cell_rate = float(total_missing_cells / total_cells) if total_cells > 0 else 0.0

        columns: Dict[str, ColumnProfile] = {}
        constant_columns: List[str] = []
        all_null_columns: List[str] = []
        warnings: List[QualityWarning] = []

        if row_count == 0:
            warnings.append(
                QualityWarning(
                    category="empty_dataset",
                    severity="warning",
                    message="Dataset contains 0 rows; summary statistics are unavailable.",
                )
            )

        if duplicate_row_count > 0:
            warnings.append(
                QualityWarning(
                    category="duplicate_rows",
                    severity="warning",
                    message=f"Found {duplicate_row_count} exact duplicate row(s) ({duplicate_row_rate:.1%}).",
                    details={"duplicate_row_count": duplicate_row_count, "rate": duplicate_row_rate},
                )
            )

        for col_name in df.columns:
            col_profile = self._profile_column(str(col_name), df[col_name])
            columns[str(col_name)] = col_profile

            if col_profile.is_all_null:
                all_null_columns.append(str(col_name))
                warnings.append(
                    QualityWarning(
                        category="all_null_column",
                        severity="high",
                        column=str(col_name),
                        message=f"Column '{col_name}' contains only null values (100% missing).",
                    )
                )
            else:
                if col_profile.is_constant:
                    constant_columns.append(str(col_name))
                    warnings.append(
                        QualityWarning(
                            category="constant_column",
                            severity="info",
                            column=str(col_name),
                            message=f"Column '{col_name}' has only 1 distinct non-null value.",
                        )
                    )
                if col_profile.null_rate > self.config.max_null_rate_warning:
                    warnings.append(
                        QualityWarning(
                            category="high_missing_rate",
                            severity="warning",
                            column=str(col_name),
                            message=(
                                f"Column '{col_name}' has {col_profile.null_count} missing values "
                                f"({col_profile.null_rate:.1%})."
                            ),
                            details={"null_count": col_profile.null_count, "null_rate": col_profile.null_rate},
                        )
                    )
                if col_profile.iqr_outlier_count > 0:
                    warnings.append(
                        QualityWarning(
                            category="outliers_detected",
                            severity="warning",
                            column=str(col_name),
                            message=(
                                f"Column '{col_name}' has {col_profile.iqr_outlier_count} IQR outlier(s) "
                                f"and {col_profile.zscore_outlier_count} z-score outlier(s)."
                            ),
                            details={
                                "iqr_outlier_count": col_profile.iqr_outlier_count,
                                "zscore_outlier_count": col_profile.zscore_outlier_count,
                            },
                        )
                    )

        if target is not None and row_count > 0:
            warnings.extend(self._check_target_diagnostics(df, target, columns))

        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=duplicate_row_count,
            duplicate_row_rate=duplicate_row_rate,
            total_missing_cells=total_missing_cells,
            missing_cell_rate=missing_cell_rate,
            constant_columns=constant_columns,
            all_null_columns=all_null_columns,
            target_column=target,
            columns=columns,
            warnings=warnings,
        )

    def infer_semantic_type(self, series: pd.Series, column_name: str = "") -> str:
        """Infer one of: numeric, categorical, boolean, datetime, identifier."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        non_null = series.dropna()
        if non_null.empty:
            if pd.api.types.is_numeric_dtype(series):
                return "numeric"
            return "categorical"

        if all(isinstance(v, (bool, np.bool_)) for v in non_null):
            return "boolean"

        col_lower = column_name.strip().lower()
        n_non_null = len(non_null)
        cardinality = int(non_null.nunique())
        uniqueness_ratio = cardinality / n_non_null

        # Check string-encoded booleans and datetimes before identifier/categorical
        if not pd.api.types.is_numeric_dtype(series):
            str_vals = non_null.astype(str).str.strip()
            lowered = set(str_vals.str.lower().unique())
            if lowered and lowered.issubset(_BOOL_STRINGS) and len(lowered) <= 2:
                return "boolean"

            if self._looks_like_datetime(str_vals):
                return "datetime"

        is_id_name = (
            col_lower == "id"
            or col_lower.endswith(_ID_SUFFIXES)
            or col_lower.startswith(_ID_PREFIXES)
        )
        is_float_dtype = pd.api.types.is_float_dtype(series)
        if not is_float_dtype:
            if is_id_name and n_non_null >= 2 and uniqueness_ratio >= 0.70:
                return "identifier"
            if (
                not pd.api.types.is_numeric_dtype(series)
                and n_non_null >= 15
                and uniqueness_ratio >= 0.98
            ):
                return "identifier"

        if pd.api.types.is_numeric_dtype(series):
            if set(non_null.unique()).issubset({0, 1}) and col_lower.startswith(
                ("is_", "has_", "can_")
            ):
                return "boolean"
            return "numeric"

        return "categorical"

    @staticmethod
    def _looks_like_datetime(str_vals: pd.Series) -> bool:
        """Return True if string series contains parseable dates without matching plain numbers."""
        sample = str_vals.head(100)
        if sample.empty:
            return False
        hint_ratio = float(sample.str.contains(_DATE_HINT_RE, regex=True).mean())
        if hint_ratio < 0.80:
            return False
        parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
        return float(parsed.notna().mean()) >= 0.90

    def _profile_column(self, name: str, series: pd.Series) -> ColumnProfile:
        count = int(len(series))
        null_count = int(series.isna().sum())
        non_null_count = count - null_count
        null_rate = float(null_count / count) if count > 0 else 0.0
        non_null = series.dropna()
        cardinality = int(non_null.nunique()) if non_null_count > 0 else 0
        uniqueness_ratio = float(cardinality / non_null_count) if non_null_count > 0 else 0.0
        is_all_null = count > 0 and non_null_count == 0
        is_constant = non_null_count > 0 and cardinality == 1

        semantic_type = self.infer_semantic_type(series, column_name=name)

        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            count=count,
            non_null_count=non_null_count,
            null_count=null_count,
            null_rate=null_rate,
            cardinality=cardinality,
            uniqueness_ratio=uniqueness_ratio,
            is_constant=is_constant,
            is_all_null=is_all_null,
        )

        if non_null_count == 0:
            return profile

        if semantic_type == "numeric":
            vals = pd.to_numeric(non_null, errors="coerce").dropna().astype(float)
            if not vals.empty:
                mean_val = float(vals.mean())
                std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                q25 = float(vals.quantile(0.25))
                median_val = float(vals.quantile(0.50))
                q75 = float(vals.quantile(0.75))
                iqr_val = float(q75 - q25)

                skew_val: Optional[float]
                if len(vals) >= 3:
                    skew_val = 0.0 if std_val == 0.0 else float(vals.skew())
                else:
                    skew_val = None

                if iqr_val > 0.0:
                    lower = q25 - self.config.iqr_multiplier * iqr_val
                    upper = q75 + self.config.iqr_multiplier * iqr_val
                    iqr_outliers = int(((vals < lower) | (vals > upper)).sum())
                else:
                    iqr_outliers = 0

                if std_val > 0.0:
                    z_scores = (vals - mean_val).abs() / std_val
                    z_outliers = int((z_scores > self.config.zscore_threshold).sum())
                else:
                    z_outliers = 0

                profile.mean = mean_val
                profile.std = std_val
                profile.min = float(vals.min())
                profile.q25 = q25
                profile.median = median_val
                profile.q75 = q75
                profile.max = float(vals.max())
                profile.iqr = iqr_val
                profile.skewness = skew_val
                profile.iqr_outlier_count = iqr_outliers
                profile.zscore_outlier_count = z_outliers

        elif semantic_type == "datetime":
            parsed_dt = pd.to_datetime(non_null, format="mixed", errors="coerce").dropna()
            if not parsed_dt.empty:
                profile.min_datetime = parsed_dt.min().isoformat()
                profile.max_datetime = parsed_dt.max().isoformat()

        if semantic_type in ("categorical", "boolean", "identifier"):
            vc = non_null.value_counts(dropna=True)
            if not vc.empty:
                profile.mode = self._to_python_scalar(vc.index[0])
                profile.top_values = [
                    {
                        "value": self._to_python_scalar(idx),
                        "count": int(cnt),
                        "rate": float(cnt / non_null_count),
                    }
                    for idx, cnt in vc.head(5).items()
                ]

        return profile

    def _check_target_diagnostics(
        self,
        df: pd.DataFrame,
        target: str,
        columns: Dict[str, ColumnProfile],
    ) -> List[QualityWarning]:
        warnings: List[QualityWarning] = []
        target_series = df[target]
        target_profile = columns[target]
        target_non_null = target_series.dropna()

        if target_non_null.empty:
            return warnings

        # 1. Class imbalance check for discrete targets
        is_discrete_target = (
            target_profile.semantic_type in ("categorical", "boolean")
            or (target_profile.semantic_type == "numeric" and target_profile.cardinality <= 20)
        )
        if is_discrete_target:
            counts = target_non_null.value_counts()
            if len(counts) == 1:
                warnings.append(
                    QualityWarning(
                        category="class_imbalance",
                        severity="high",
                        column=target,
                        message=f"Target column '{target}' contains only a single class.",
                        details={"class_counts": {str(k): int(v) for k, v in counts.items()}},
                    )
                )
            elif len(counts) >= 2:
                minority_count = int(counts.min())
                majority_count = int(counts.max())
                ratio = minority_count / majority_count
                if ratio < self.config.imbalance_ratio_threshold:
                    warnings.append(
                        QualityWarning(
                            category="class_imbalance",
                            severity="warning",
                            column=target,
                            message=(
                                f"Target '{target}' exhibits class imbalance "
                                f"(minority/majority ratio={ratio:.3f} < {self.config.imbalance_ratio_threshold:.2f})."
                            ),
                            details={
                                "minority_class": self._to_python_scalar(counts.idxmin()),
                                "majority_class": self._to_python_scalar(counts.idxmax()),
                                "minority_to_majority_ratio": round(ratio, 4),
                            },
                        )
                    )

        # 2. Target leakage check across predictor columns
        target_numeric = self._as_numeric_for_corr(target_series, target_profile.semantic_type)
        for col_name, col_prof in columns.items():
            if col_name == target or col_prof.semantic_type == "identifier" or col_prof.is_constant:
                continue

            leaked = False
            col_numeric = self._as_numeric_for_corr(df[col_name], col_prof.semantic_type)
            if target_numeric is not None and col_numeric is not None:
                valid = pd.concat([col_numeric, target_numeric], axis=1).dropna()
                if len(valid) >= 3 and valid.iloc[:, 0].std() > 0 and valid.iloc[:, 1].std() > 0:
                    corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
                    if abs(corr) >= self.config.leakage_correlation_threshold:
                        warnings.append(
                            QualityWarning(
                                category="target_leakage",
                                severity="high",
                                column=col_name,
                                message=(
                                    f"Potential target leakage: '{col_name}' has correlation "
                                    f"{corr:.3f} with target '{target}'."
                                ),
                                details={"metric": "pearson_correlation", "value": round(corr, 4)},
                            )
                        )
                        leaked = True

            if not leaked and is_discrete_target and col_prof.cardinality >= 2:
                valid_pair = df[[col_name, target]].dropna()
                n_valid = len(valid_pair)
                if n_valid >= 5 and col_prof.cardinality <= max(2, int(0.5 * n_valid)):
                    baseline_acc = float(valid_pair[target].value_counts().max() / n_valid)
                    grouped = valid_pair.groupby(col_name, observed=True)[target].value_counts()
                    # Max target count within each predictor category
                    correct_per_group = grouped.groupby(level=0, observed=True).max().sum()
                    purity = float(correct_per_group / n_valid)
                    if (purity - baseline_acc) >= 0.02 and purity >= self.config.leakage_purity_threshold:
                        warnings.append(
                            QualityWarning(
                                category="target_leakage",
                                severity="high",
                                column=col_name,
                                message=(
                                    f"Potential target leakage: '{col_name}' predicts target "
                                    f"'{target}' with {purity:.1%} conditional purity (baseline {baseline_acc:.1%})."
                                ),
                                details={
                                    "metric": "conditional_purity",
                                    "purity": round(purity, 4),
                                    "baseline": round(baseline_acc, 4),
                                },
                            )
                        )

        return warnings

    @staticmethod
    def _as_numeric_for_corr(series: pd.Series, semantic_type: str) -> Optional[pd.Series]:
        if semantic_type == "numeric":
            return pd.to_numeric(series, errors="coerce").astype(float)
        if semantic_type == "boolean":
            if pd.api.types.is_bool_dtype(series):
                return series.astype(float)
            mapping = {"true": 1.0, "yes": 1.0, "t": 1.0, "y": 1.0, "1": 1.0,
                       "false": 0.0, "no": 0.0, "f": 0.0, "n": 0.0, "0": 0.0}
            return series.map(
                lambda v: (
                    np.nan
                    if pd.isna(v)
                    else (float(v) if isinstance(v, (bool, np.bool_, int, float)) else mapping.get(str(v).strip().lower(), np.nan))
                )
            ).astype(float)
        return None

    @staticmethod
    def _to_python_scalar(val: Any) -> Any:
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        return val
