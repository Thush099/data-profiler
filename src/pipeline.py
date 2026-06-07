
import logging
from src.models import DataQualityModel
from src.utils import calculate_data_quality_metrics
from typing import Dict

class DataProfiler:
    def __init__(self, config: Config):
        self.config = config
        self.model = DataQualityModel()

    def profile(self, data: Dict) -> Dict:
        logging.info('Profiling data...')
        metrics = calculate_data_quality_metrics(data)
        predictions = self.model.predict(metrics)

        report = {
            'data_quality_metrics': metrics,
            'remediation_suggestions': predictions
        }

        return report
