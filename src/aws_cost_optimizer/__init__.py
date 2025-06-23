__version__ = "1.0.0"

# Core components
from .optimization.ec2_optimizer import EC2Optimizer, EC2OptimizationRecommendation
from .optimization.network_optimizer import NetworkOptimizer, NetworkOptimizationRecommendation
from .optimization.reserved_instance_analyzer import ReservedInstanceAnalyzer, RIRecommendation, SavingsPlanRecommendation
from .optimization.rds_optimizer import RDSOptimizer, RDSRecommendation, RDSOptimizationAction
from .optimization.s3_optimizer import S3Optimizer, S3OptimizationRecommendation
from .optimization.auto_remediation_engine import (
    AutoRemediationEngine,
    RemediationPolicy,
    RemediationAction,
    RemediationStatus,
    RemediationTask
)
from .optimization.safety_checks import SafetyChecker, SafetyOrchestrator

from .analysis.pattern_detector import PatternDetector, ResourcePattern, WorkloadCharacteristics, WorkloadType, UsagePhase
from .analysis.cost_anomaly_detector import CostAnomalyDetector, CostAnomaly, CostTrend

from .discovery.multi_account import MultiAccountInventory, AWSAccount

from .orchestrator import CostOptimizationOrchestrator, OptimizationResult

__all__ = [
    # Version
    '__version__',
    
    # Optimizers
    'EC2Optimizer',
    'EC2OptimizationRecommendation',
    'NetworkOptimizer',
    'NetworkOptimizationRecommendation',
    'RDSOptimizer',
    'RDSRecommendation',
    'RDSOptimizationAction',
    'S3Optimizer',
    'S3OptimizationRecommendation',
    'ReservedInstanceAnalyzer',
    'RIRecommendation',
    'SavingsPlanRecommendation',
    
    # Auto-remediation
    'AutoRemediationEngine',
    'RemediationPolicy',
    'RemediationAction',
    'RemediationStatus',
    'RemediationTask',
    
    # Safety
    'SafetyChecker',
    'SafetyOrchestrator',
    
    # Analysis
    'PatternDetector',
    'ResourcePattern',
    'WorkloadCharacteristics',
    'WorkloadType',
    'UsagePhase',
    'CostAnomalyDetector',
    'CostAnomaly',
    'CostTrend',
    
    # Discovery
    'MultiAccountInventory',
    'AWSAccount',
    
    # Orchestration
    'CostOptimizationOrchestrator',
    'OptimizationResult'
]