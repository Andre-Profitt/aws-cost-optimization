"""Analysis module for resource usage patterns and optimization opportunities"""

from .pattern_detector import (
    PatternDetector,
    ResourcePattern,
    WorkloadCharacteristics,
    WorkloadType,
    UsagePhase
)
from .s3_access_analyzer import (
    S3AccessAnalyzer,
    integrate_with_s3_optimizer
)
from .periodic_detector import (
    PeriodicResourceDetector,
    PeriodType,
    PeriodicPattern,
    ResourcePeriodicity
)
from .cost_anomaly_detector import (
    CostAnomalyDetector,
    CostAnomaly,
    CostTrend
)

__all__ = [
    'PatternDetector',
    'ResourcePattern',
    'WorkloadCharacteristics',
    'WorkloadType',
    'UsagePhase',
    'S3AccessAnalyzer',
    'integrate_with_s3_optimizer',
    'PeriodicResourceDetector',
    'PeriodType',
    'PeriodicPattern',
    'ResourcePeriodicity',
    'CostAnomalyDetector',
    'CostAnomaly',
    'CostTrend'
]