"""End-to-end orchestration pipeline combining profiling, contracts, and remediation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from config import Config
from src.contracts import ContractValidator
from src.models import (
    ContractReport,
    DataContract,
    DatasetProfile,
    RemediationPlan,
    RemediationResult,
)
from src.profiler import DataProfiler
from src.remediation import RemediationPlanner, apply_remediation


@dataclass
class PipelineOutput:
    """Structured result from executing the DataQualityPipeline."""

    profile: DatasetProfile
    contract_report: Optional[ContractReport]
    remediation_plan: RemediationPlan
    remediation_result: Optional[RemediationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline results to a JSON-serializable report dictionary."""
        payload: Dict[str, Any] = {
            "profile": self.profile.to_dict(),
            "contract_validation": (
                self.contract_report.to_dict() if self.contract_report is not None else None
            ),
            "remediation_plan": self.remediation_plan.to_dict(),
        }
        if self.remediation_result is not None:
            payload["remediation_execution"] = self.remediation_result.to_dict()
        return payload


class DataQualityPipeline:
    """Runs profiling, optional contract validation, and remediation planning."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.profiler = DataProfiler(config=self.config)
        self.validator = ContractValidator()
        self.planner = RemediationPlanner(config=self.config)

    def run(
        self,
        df: pd.DataFrame,
        contract: Optional[DataContract] = None,
        target: Optional[str] = None,
        execute_remediation: bool = False,
    ) -> PipelineOutput:
        """Execute profiling, contract validation, and optional remediation."""
        profile = self.profiler.profile(df, target=target)
        contract_report = (
            self.validator.validate(df, contract) if contract is not None else None
        )
        remediation_plan = self.planner.plan(profile, contract_report=contract_report)
        remediation_result = (
            apply_remediation(df, remediation_plan) if execute_remediation else None
        )
        return PipelineOutput(
            profile=profile,
            contract_report=contract_report,
            remediation_plan=remediation_plan,
            remediation_result=remediation_result,
        )
