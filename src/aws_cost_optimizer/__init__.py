__version__ = "1.0.0"

# Core components
from .optimization.ec2_optimizer import EC2Optimizer
from .optimization.network_optimizer import NetworkOptimizer
from .optimization.reserved_instance_analyzer import ReservedInstanceAnalyzer
from .optimization.auto_remediation_engine import (
    AutoRemediationEngine,
    RemediationPolicy,
    RemediationAction,
    RemediationStatus
)

from .analysis.pattern_detector import PatternDetector
from .analysis.cost_anomaly_detector import CostAnomalyDetector

from .discovery.multi_account import MultiAccountDiscovery

__all__ = [
    'EC2Optimizer',
    'NetworkOptimizer',
    'ReservedInstanceAnalyzer',
    'AutoRemediationEngine',
    'RemediationPolicy',
    'RemediationAction',
    'RemediationStatus',
    'PatternDetector',
    'CostAnomalyDetector',
    'MultiAccountDiscovery'
]