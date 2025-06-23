"""Intelligent tagging module for automated resource classification"""

from .intelligent_tagger import (
    IntelligentTagger,
    TagSuggestion,
    TaggingRule,
    TagComplianceResult,
    TagCategory
)

__all__ = [
    'IntelligentTagger',
    'TagSuggestion',
    'TaggingRule',
    'TagComplianceResult',
    'TagCategory'
]