from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OptimizationAction(Enum):
    STOP = "stop"
    RIGHTSIZE = "rightsize"
    DELETE = "delete"
    SCHEDULE = "schedule"
    MOVE_TO_SPOT = "move_to_spot"
    PURCHASE_RI = "purchase_ri"
    CHANGE_STORAGE_CLASS = "change_storage_class"


@dataclass
class EC2OptimizationRecommendation:
    instance_id: str
    instance_type: str
    region: str
    action: str
    reason: str
    monthly_savings: float
    annual_savings: float
    risk_level: str
    implementation_steps: List[str]
    tags: Dict[str, str]
    metrics: Dict[str, float]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class EBSOptimizationRecommendation:
    volume_id: str
    volume_type: str
    size_gb: int
    region: str
    action: str
    reason: str
    monthly_savings: float
    annual_savings: float
    risk_level: str
    implementation_steps: List[str]
    attached_instance_id: Optional[str] = None
    tags: Dict[str, str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = {}


@dataclass
class S3OptimizationRecommendation:
    bucket_name: str
    region: str
    action: str
    reason: str
    monthly_savings: float
    annual_savings: float
    risk_level: str
    implementation_steps: List[str]
    current_storage_class: str
    recommended_storage_class: Optional[str] = None
    size_gb: float = 0.0
    object_count: int = 0
    last_modified: Optional[datetime] = None
    tags: Dict[str, str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = {}