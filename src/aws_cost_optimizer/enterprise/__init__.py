"""
Enterprise features for AWS Cost Optimizer

This module provides enterprise-grade enhancements including:
- Dependency mapping for safe change execution
- Change management with approval workflows
- Monitoring integration for impact assessment
- Enhanced compliance and audit capabilities
"""

from .integration import (
    EnterpriseConfig,
    EnterpriseOptimizer,
    EnhancedSafetyChecker,
    run_enterprise_optimization_example
)

__all__ = [
    'EnterpriseConfig',
    'EnterpriseOptimizer',
    'EnhancedSafetyChecker',
    'run_enterprise_optimization_example'
]