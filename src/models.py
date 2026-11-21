"""Typed dataclasses for data profiling, schema contracts, and remediation."""

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class QualityWarning:
    """Structured warning raised during dataset profiling."""

    category: str
    severity: str
    message: str
    column: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize warning to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class ColumnProfile:
    """Comprehensive quality profile for a single DataFrame column."""

    name: str
    dtype: str
    semantic_type: str
    count: int
    non_null_count: int
    null_count: int
    null_rate: float
    cardinality: int
    uniqueness_ratio: float
    is_constant: bool
    is_all_null: bool
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    iqr_outlier_count: int = 0
    zscore_outlier_count: int = 0
    mode: Optional[Any] = None
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    min_datetime: Optional[str] = None
    max_datetime: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize column profile to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class DatasetProfile:
    """Dataset-wide profiling summary and column-level profiles."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_rate: float
    total_missing_cells: int
    missing_cell_rate: float
    constant_columns: List[str] = field(default_factory=list)
    all_null_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    warnings: List[QualityWarning] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset profile to a JSON-compatible dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_rate": self.duplicate_row_rate,
            "total_missing_cells": self.total_missing_cells,
            "missing_cell_rate": self.missing_cell_rate,
            "constant_columns": list(self.constant_columns),
            "all_null_columns": list(self.all_null_columns),
            "target_column": self.target_column,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass
class ColumnRule:
    """Declarative validation constraints for a single column."""

    column: str
    not_null: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    unique: bool = False
    max_null_rate: Optional[float] = None

    def __post_init__(self) -> None:
        for name, bound in (("min_value", self.min_value), ("max_value", self.max_value)):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, Real)
                or not math.isfinite(bound)
            ):
                raise ValueError(
                    f"Invalid rule for '{self.column}': {name} must be finite, got {bound!r}."
                )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"Invalid rule for '{self.column}': min_value ({self.min_value}) > max_value ({self.max_value})."
            )
        if self.max_null_rate is not None and not 0.0 <= self.max_null_rate <= 1.0:
            raise ValueError(
                f"Invalid max_null_rate for '{self.column}': must be in [0, 1]."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rule to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], column_name: Optional[str] = None) -> "ColumnRule":
        """Create a ColumnRule from a mapping."""
        col = data.get("column") or column_name
        if not col:
            raise ValueError("ColumnRule requires a 'column' name.")
        return cls(
            column=col,
            not_null=bool(data.get("not_null", False)),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=list(data["allowed_values"]) if data.get("allowed_values") is not None else None,
            unique=bool(data.get("unique", False)),
            max_null_rate=data.get("max_null_rate"),
        )


@dataclass
class DataContract:
    """Collection of declarative column rules defining a dataset contract."""

    name: str = "data_contract"
    rules: Union[Dict[str, ColumnRule], Sequence[ColumnRule]] = field(default_factory=dict)
    strict_columns: bool = False

    def __post_init__(self) -> None:
        normalized: Dict[str, ColumnRule] = {}
        if isinstance(self.rules, dict):
            for col_name, rule in self.rules.items():
                if isinstance(rule, ColumnRule):
                    normalized[rule.column] = rule
                elif isinstance(rule, dict):
                    parsed = ColumnRule.from_dict(rule, column_name=col_name)
                    normalized[parsed.column] = parsed
                else:
                    raise TypeError(f"Unsupported rule type for '{col_name}': {type(rule)}")
        else:
            for rule in self.rules:
                if isinstance(rule, ColumnRule):
                    normalized[rule.column] = rule
                elif isinstance(rule, dict):
                    parsed = ColumnRule.from_dict(rule)
                    normalized[parsed.column] = parsed
                else:
                    raise TypeError(f"Unsupported rule entry: {type(rule)}")
        self.rules = normalized

    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract to a JSON-compatible dictionary."""
        assert isinstance(self.rules, dict)
        return {
            "name": self.name,
            "strict_columns": self.strict_columns,
            "rules": {col: rule.to_dict() for col, rule in self.rules.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize contract to JSON text."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_json_file(self, file_path: Union[str, Path]) -> None:
        """Write contract to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataContract":
        """Construct a DataContract from a dictionary representation."""
        raw_rules = data.get("rules", {})
        parsed_rules: Dict[str, ColumnRule] = {}
        if isinstance(raw_rules, list):
            for entry in raw_rules:
                rule = ColumnRule.from_dict(entry)
                parsed_rules[rule.column] = rule
        elif isinstance(raw_rules, dict):
            for col_name, entry in raw_rules.items():
                rule = entry if isinstance(entry, ColumnRule) else ColumnRule.from_dict(entry, column_name=col_name)
                parsed_rules[rule.column] = rule
        else:
            raise ValueError("'rules' in contract payload must be a dict or list.")
        return cls(
            name=str(data.get("name", "data_contract")),
            rules=parsed_rules,
            strict_columns=bool(data.get("strict_columns", False)),
        )

    @classmethod
    def from_json(cls, payload: str) -> "DataContract":
        """Construct a DataContract from a JSON string."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> "DataContract":
        """Load a DataContract from a JSON file."""
        path = Path(file_path)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract file not found: {path}") from None
        try:
            return cls.from_json(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Contract file is not valid JSON: {path} ({exc})") from exc


@dataclass
class ContractViolation:
    """Details about a single failed contract constraint."""

    column: str
    rule: str
    failed_count: int
    failure_rate: float
    message: str
    sample_indices: List[Any] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize violation to a dictionary."""
        return asdict(self)


@dataclass
class ContractReport:
    """Aggregated outcome from validating a DataFrame against a DataContract."""

    contract_name: str
    passed: bool
    total_rules_checked: int
    passed_rules: int
    failed_rules: int
    violations: List[ContractViolation] = field(default_factory=list)
    violating_row_indices: List[Any] = field(default_factory=list)
    violating_row_positions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract validation report to a dictionary."""
        return {
            "contract_name": self.contract_name,
            "passed": self.passed,
            "total_rules_checked": self.total_rules_checked,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "violations": [v.to_dict() for v in self.violations],
            "violating_row_indices": list(self.violating_row_indices),
            "violating_row_positions": list(self.violating_row_positions),
        }


@dataclass
class RemediationAction:
    """Prioritized remediation step generated from profiling/contract findings."""

    priority: int
    action: str
    target: str
    reason: str
    affected_rows: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize remediation action to a dictionary."""
        return asdict(self)


@dataclass
class RemediationPlan:
    """Ordered collection of remediation actions for a dataset."""

    actions: List[RemediationAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize remediation plan to a dictionary."""
        return {
            "total_actions": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class RemediationResult:
    """Outcome of executing a RemediationPlan against a DataFrame."""

    cleaned_df: Any
    quarantined_df: Any
    applied_actions: List[str]
    rows_before: int
    rows_after: int
    rows_quarantined: int
    columns_dropped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary metadata (excluding raw DataFrames) to dictionary."""
        return {
            "applied_actions": list(self.applied_actions),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_quarantined": self.rows_quarantined,
            "columns_dropped": list(self.columns_dropped),
        }
