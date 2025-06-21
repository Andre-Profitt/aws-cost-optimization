from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class EC2OptimizationRecommendation:
    instance_id: str
    action: str
    estimated_monthly_savings: float
    reason: str

class EC2Optimizer:
    def __init__(self, cpu_threshold=10.0, network_io_threshold=5.0):
        self.cpu_threshold = cpu_threshold
        self.network_io_threshold = network_io_threshold
    
    def analyze_instance(self, instance: Dict[str, Any]) -> EC2OptimizationRecommendation:
        """Analyze single EC2 instance for optimization"""
        instance_id = instance['resource_id']
        cpu_avg = instance['usage_metrics'].get('cpu_utilization_avg', 100)
        network_mb = instance['usage_metrics'].get('network_in_total', 0) / (1024**2)
        
        # Check if idle (using guide thresholds)
        if cpu_avg < self.cpu_threshold and network_mb < self.network_io_threshold:
            monthly_cost = self._calculate_monthly_cost(instance)
            return EC2OptimizationRecommendation(
                instance_id=instance_id,
                action='stop',
                estimated_monthly_savings=monthly_cost * 0.9,
                reason=f"Idle: CPU {cpu_avg:.1f}%, Network {network_mb:.1f}MB"
            )
        
        return EC2OptimizationRecommendation(
            instance_id=instance_id,
            action='none',
            estimated_monthly_savings=0,
            reason="Instance properly utilized"
        )
    
    def _calculate_monthly_cost(self, instance: Dict[str, Any]) -> float:
        """Calculate monthly cost for instance"""
        # Simplified pricing
        instance_costs = {
            't2.micro': 8.5,
            't2.small': 17,
            't2.medium': 34,
            't3.large': 62,
            'm5.large': 70,
            'm5.xlarge': 140
        }
        instance_type = instance['cost_info']['instance_type']
        return instance_costs.get(instance_type, 100) # Default $100