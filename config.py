"""Configuration settings for the data profiler, validator, and remediation planner."""

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Dict


@dataclass
class Config:
    """Runtime configuration thresholds and default remediation strategies."""

    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    max_null_rate_warning: float = 0.05
    imbalance_ratio_threshold: float = 0.20
    leakage_correlation_threshold: float = 0.95
    leakage_purity_threshold: float = 0.98
    default_input_path: str = "data/sample_dirty.csv"
    remediation_strategy: Dict[str, str] = field(
        default_factory=lambda: {
            "duplicates": "drop_duplicates",
            "numeric_missing": "impute_median",
            "categorical_missing": "impute_mode",
            "outliers": "winsorize_iqr",
            "constant_columns": "drop_constant_columns",
            "contract_violations": "quarantine_contract_violations",
        }
    )

    def __post_init__(self) -> None:
        for name in ("iqr_multiplier", "zscore_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive real, got {value!r}")
        if not 0.0 <= self.max_null_rate_warning <= 1.0:
            raise ValueError("max_null_rate_warning must be between 0 and 1.")
        if not 0.0 < self.imbalance_ratio_threshold <= 1.0:
            raise ValueError("imbalance_ratio_threshold must be in (0, 1].")
        if not 0.0 < self.leakage_correlation_threshold <= 1.0:
            raise ValueError("leakage_correlation_threshold must be in (0, 1].")
        if not 0.0 < self.leakage_purity_threshold <= 1.0:
            raise ValueError("leakage_purity_threshold must be in (0, 1].")
