
from dataclasses import dataclass
from typing import Dict

@dataclass
class Config:
    data_quality_threshold: float = 0.5
    remediation_strategy: Dict = {'handle_missing_values': 'impute_with_mean', 'handle_invalid_values': 'remove'}
