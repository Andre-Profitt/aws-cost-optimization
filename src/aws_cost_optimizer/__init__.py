__version__ = "2.0.0"

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
from .analysis.s3_access_analyzer import S3AccessAnalyzer, integrate_with_s3_optimizer
from .analysis.periodic_detector import PeriodicResourceDetector, PeriodType, PeriodicPattern, ResourcePeriodicity

from .ml.cost_predictor import CostPredictor, CostPrediction, PredictedAnomaly, ModelPerformance, PredictionType, AnomalyType

from .realtime.cost_controller import RealtimeCostController, CostThreshold, CircuitBreaker, CostEvent, ControlAction, ThresholdType

from .tagging.intelligent_tagger import IntelligentTagger, TagSuggestion, TaggingRule, TagComplianceResult, TagCategory

from .tracking.savings_tracker import SavingsTracker, OptimizationRecord, SavingsSummary, OptimizationType, SavingsStatus

from .discovery.multi_account import MultiAccountInventory, AWSAccount

from .multi_account import MultiAccountInventory as MAInventory, EmergencyCostReducer

from .compliance import (
    ComplianceManager,
    AuditTrail,
    ComplianceStatus,
    AuditEventType,
    ComplianceRule,
    ComplianceViolation,
    AuditEvent
)

from .enterprise import (
    EnterpriseConfig,
    EnterpriseOptimizer,
    EnhancedSafetyChecker
)

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
    'S3AccessAnalyzer',
    'integrate_with_s3_optimizer',
    'PeriodicResourceDetector',
    'PeriodType',
    'PeriodicPattern',
    'ResourcePeriodicity',
    
    # ML
    'CostPredictor',
    'CostPrediction',
    'PredictedAnomaly',
    'ModelPerformance',
    'PredictionType',
    'AnomalyType',
    
    # Real-time
    'RealtimeCostController',
    'CostThreshold',
    'CircuitBreaker',
    'CostEvent',
    'ControlAction',
    'ThresholdType',
    
    # Tagging
    'IntelligentTagger',
    'TagSuggestion',
    'TaggingRule',
    'TagComplianceResult',
    'TagCategory',
    
    # Tracking
    'SavingsTracker',
    'OptimizationRecord',
    'SavingsSummary',
    'OptimizationType',
    'SavingsStatus',
    
    # Discovery
    'MultiAccountInventory',
    'AWSAccount',
    
    # Multi-Account
    'MAInventory',
    'EmergencyCostReducer',
    
    # Compliance
    'ComplianceManager',
    'AuditTrail',
    'ComplianceStatus',
    'AuditEventType',
    'ComplianceRule',
    'ComplianceViolation',
    'AuditEvent',
    
    # Enterprise
    'EnterpriseConfig',
    'EnterpriseOptimizer',
    'EnhancedSafetyChecker',
    
    # Orchestration
    'CostOptimizationOrchestrator',
    'OptimizationResult'
]