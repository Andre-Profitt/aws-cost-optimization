"""
AWS Cost Optimizer - Safely reduce AWS costs without breaking production

A comprehensive tool for discovering, analyzing, and optimizing AWS resources
with focus on safety and production stability.
"""

__version__ = "1.0.0"
__author__ = "AWS Cost Optimizer Team"
__email__ = "aws-cost-optimizer@your-org.com"

from .discovery.multi_account import MultiAccountInventory, AWSAccount
from .analysis.pattern_detector import PatternDetector
from .optimization.safety_checks import SafetyOrchestrator
from .optimization.ec2_optimizer import EC2Optimizer

__all__ = [
    "MultiAccountInventory",
    "AWSAccount", 
    "PatternDetector",
    "SafetyOrchestrator",
    "EC2Optimizer"
]