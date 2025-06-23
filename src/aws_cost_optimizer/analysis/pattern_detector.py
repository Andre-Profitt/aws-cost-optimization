"""
Complete Pattern Detection System - Analyzes usage patterns and workload characteristics
"""
import boto3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    """Types of workload patterns"""
    STEADY_STATE = "steady_state"
    BATCH_PROCESSING = "batch_processing"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SEASONAL = "seasonal"
    SPIKY = "spiky"
    UNKNOWN = "unknown"

class UsagePhase(Enum):
    """Phases of resource usage lifecycle"""
    STARTUP = "startup"
    GROWTH = "growth"
    MATURE = "mature"
    DECLINING = "declining"

@dataclass
class ResourcePattern:
    """Pattern analysis for a resource"""
    resource_id: str
    resource_type: str  # 'ec2', 'rds', 'ebs', etc.
    region: str
    
    # Pattern classification
    workload_type: WorkloadType
    usage_phase: UsagePhase
    
    # Time-based patterns
    daily_pattern: Dict[int, float]  # Hour -> avg utilization
    weekly_pattern: Dict[int, float]  # Day of week -> avg utilization
    monthly_pattern: Dict[int, float]  # Day of month -> avg utilization
    
    # Statistical analysis
    utilization_stats: Dict[str, float]
    seasonality_score: float
    variability_score: float
    predictability_score: float
    
    # Business context inference
    business_criticality: str  # 'low', 'medium', 'high', 'unknown'
    environment_type: str     # 'dev', 'test', 'staging', 'prod', 'unknown'
    cost_sensitivity: str     # 'low', 'medium', 'high'
    
    # Optimization recommendations
    optimization_opportunities: List[str]
    scheduling_potential: bool
    rightsizing_potential: bool
    ri_suitability: float  # 0-1 score
    
    # Metadata
    analysis_period_days: int
    confidence_score: float
    last_analyzed: datetime

@dataclass
class WorkloadCharacteristics:
    """Detailed workload characteristics"""
    resource_group: str  # Application or service name
    workload_patterns: List[ResourcePattern]
    
    # Aggregate patterns
    overall_workload_type: WorkloadType
    peak_hours: List[int]
    low_usage_hours: List[int]
    weekend_usage_ratio: float
    
    # Cost analysis
    total_monthly_cost: float
    optimization_potential_pct: float
    estimated_monthly_savings: float
    
    # Recommendations
    architectural_recommendations: List[str]
    operational_recommendations: List[str]
    cost_optimization_score: float

class PatternDetector:
    """Complete pattern detection and workload characterization system"""
    
    def __init__(self,
                 analysis_period_days: int = 90,
                 min_data_points: int = 100,
                 seasonality_threshold: float = 0.3,
                 predictability_threshold: float = 0.7):
        """
        Initialize pattern detector
        
        Args:
            analysis_period_days: Days of historical data to analyze
            min_data_points: Minimum data points required for analysis
            seasonality_threshold: Threshold for detecting seasonal patterns
            predictability_threshold: Threshold for predictable workloads
        """
        self.analysis_period_days = analysis_period_days
        self.min_data_points = min_data_points
        self.seasonality_threshold = seasonality_threshold
        self.predictability_threshold = predictability_threshold
        
        # Pattern matching rules
        self.workload_rules = self._load_workload_classification_rules()
        
    def analyze_all_resources(self, region_name: str = None) -> List[ResourcePattern]:
        """Analyze patterns for all resources across services"""
        patterns = []
        
        if region_name:
            regions = [region_name]
        else:
            ec2 = boto3.client('ec2')
            regions = [region['RegionName'] for region in ec2.describe_regions()['Regions']]
        
        for region in regions:
            try:
                logger.info(f"Analyzing resource patterns in region: {region}")
                
                # Analyze EC2 instances
                ec2_patterns = self._analyze_ec2_patterns(region)
                patterns.extend(ec2_patterns)
                
                # Analyze RDS instances
                rds_patterns = self._analyze_rds_patterns(region)
                patterns.extend(rds_patterns)
                
                # Analyze EBS volumes
                ebs_patterns = self._analyze_ebs_patterns(region)
                patterns.extend(ebs_patterns)
                
            except Exception as e:
                logger.error(f"Error analyzing patterns in region {region}: {e}")
                continue
        
        logger.info(f"Analyzed patterns for {len(patterns)} resources")
        return patterns
    
    def characterize_workloads(self, patterns: List[ResourcePattern]) -> List[WorkloadCharacteristics]:
        """Group resources and characterize workloads"""
        
        # Group patterns by application/service
        workload_groups = self._group_patterns_by_workload(patterns)
        
        workload_characteristics = []
        
        for group_name, group_patterns in workload_groups.items():
            try:
                characteristics = self._analyze_workload_group(group_name, group_patterns)
                workload_characteristics.append(characteristics)
            except Exception as e:
                logger.error(f"Error characterizing workload {group_name}: {e}")
        
        return workload_characteristics
    
    def _analyze_ec2_patterns(self, region: str) -> List[ResourcePattern]:
        """Analyze EC2 instance usage patterns"""
        patterns = []
        
        try:
            ec2 = boto3.client('ec2', region_name=region)
            cloudwatch = boto3.client('cloudwatch', region_name=region)
            
            # Get all instances
            instances = self._get_ec2_instances(ec2)
            
            # Analyze each instance
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_instance = {
                    executor.submit(self._analyze_single_ec2_instance, instance, cloudwatch, region): instance
                    for instance in instances
                }
                
                for future in as_completed(future_to_instance):
                    instance = future_to_instance[future]
                    try:
                        pattern = future.result()
                        if pattern:
                            patterns.append(pattern)
                    except Exception as e:
                        logger.error(f"Error analyzing EC2 instance {instance.get('InstanceId', 'unknown')}: {e}")
            
        except Exception as e:
            logger.error(f"Error analyzing EC2 patterns in {region}: {e}")
        
        return patterns
    
    def _analyze_single_ec2_instance(self, instance: Dict, cloudwatch, region: str) -> Optional[ResourcePattern]:
        """Analyze pattern for a single EC2 instance"""
        
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        state = instance['State']['Name']
        
        # Skip terminated instances
        if state == 'terminated':
            return None
        
        # Get instance tags for context
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        try:
            # Get comprehensive metrics
            metrics = self._get_ec2_comprehensive_metrics(instance_id, cloudwatch)
            
            if not metrics or len(metrics['timestamps']) < self.min_data_points:
                return None
            
            # Analyze patterns
            daily_pattern = self._extract_daily_pattern(metrics)
            weekly_pattern = self._extract_weekly_pattern(metrics)
            monthly_pattern = self._extract_monthly_pattern(metrics)
            
            # Calculate statistics
            utilization_stats = self._calculate_utilization_stats(metrics)
            seasonality_score = self._calculate_seasonality_score(metrics)
            variability_score = self._calculate_variability_score(metrics)
            predictability_score = self._calculate_predictability_score(metrics)
            
            # Classify workload
            workload_type = self._classify_workload_type(
                daily_pattern, weekly_pattern, utilization_stats, tags
            )
            
            # Determine usage phase
            usage_phase = self._determine_usage_phase(metrics, tags)
            
            # Infer business context
            business_criticality = self._infer_business_criticality(tags, workload_type)
            environment_type = self._infer_environment_type(tags)
            cost_sensitivity = self._infer_cost_sensitivity(tags, environment_type)
            
            # Generate optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                workload_type, utilization_stats, daily_pattern, weekly_pattern
            )
            
            # Assess optimization potential
            scheduling_potential = self._assess_scheduling_potential(
                daily_pattern, weekly_pattern, environment_type
            )
            rightsizing_potential = self._assess_rightsizing_potential(utilization_stats)
            ri_suitability = self._calculate_ri_suitability(
                workload_type, predictability_score, utilization_stats
            )
            
            # Calculate confidence
            confidence_score = self._calculate_analysis_confidence(
                len(metrics['timestamps']), variability_score, predictability_score
            )
            
            return ResourcePattern(
                resource_id=instance_id,
                resource_type='ec2',
                region=region,
                workload_type=workload_type,
                usage_phase=usage_phase,
                daily_pattern=daily_pattern,
                weekly_pattern=weekly_pattern,
                monthly_pattern=monthly_pattern,
                utilization_stats=utilization_stats,
                seasonality_score=seasonality_score,
                variability_score=variability_score,
                predictability_score=predictability_score,
                business_criticality=business_criticality,
                environment_type=environment_type,
                cost_sensitivity=cost_sensitivity,
                optimization_opportunities=optimization_opportunities,
                scheduling_potential=scheduling_potential,
                rightsizing_potential=rightsizing_potential,
                ri_suitability=ri_suitability,
                analysis_period_days=self.analysis_period_days,
                confidence_score=confidence_score,
                last_analyzed=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pattern for {instance_id}: {e}")
            return None
    
    def _get_ec2_comprehensive_metrics(self, instance_id: str, cloudwatch) -> Dict[str, List]:
        """Get comprehensive metrics for EC2 instance"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.analysis_period_days)
        
        metrics_data = {
            'timestamps': [],
            'cpu_utilization': [],
            'network_in': [],
            'network_out': [],
            'disk_read_ops': [],
            'disk_write_ops': []
        }
        
        # Define metrics to collect
        metric_queries = [
            ('AWS/EC2', 'CPUUtilization', 'cpu_utilization'),
            ('AWS/EC2', 'NetworkIn', 'network_in'),
            ('AWS/EC2', 'NetworkOut', 'network_out'),
            ('AWS/EC2', 'DiskReadOps', 'disk_read_ops'),
            ('AWS/EC2', 'DiskWriteOps', 'disk_write_ops')
        ]
        
        try:
            # Get CPU utilization as the primary metric
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour intervals
                Statistics=['Average']
            )
            
            if not response['Datapoints']:
                return {}
            
            # Sort by timestamp
            datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
            
            for dp in datapoints:
                metrics_data['timestamps'].append(dp['Timestamp'])
                metrics_data['cpu_utilization'].append(dp['Average'])
            
            # Get other metrics for the same time points
            for namespace, metric_name, key in metric_queries[1:]:  # Skip CPU as we already have it
                try:
                    response = cloudwatch.get_metric_statistics(
                        Namespace=namespace,
                        MetricName=metric_name,
                        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=3600,
                        Statistics=['Average']
                    )
                    
                    # Align with timestamps
                    metric_values = []
                    for timestamp in metrics_data['timestamps']:
                        value = 0
                        for dp in response['Datapoints']:
                            if abs((dp['Timestamp'] - timestamp).total_seconds()) < 1800:  # 30 min tolerance
                                value = dp['Average']
                                break
                        metric_values.append(value)
                    
                    metrics_data[key] = metric_values
                    
                except Exception as e:
                    logger.warning(f"Could not get {metric_name} for {instance_id}: {e}")
                    metrics_data[key] = [0] * len(metrics_data['timestamps'])
            
        except Exception as e:
            logger.error(f"Error getting metrics for {instance_id}: {e}")
            return {}
        
        return metrics_data
    
    def _extract_daily_pattern(self, metrics: Dict[str, List]) -> Dict[int, float]:
        """Extract daily usage pattern (24-hour cycle)"""
        daily_pattern = defaultdict(list)
        
        for i, timestamp in enumerate(metrics['timestamps']):
            hour = timestamp.hour
            cpu_utilization = metrics['cpu_utilization'][i]
            daily_pattern[hour].append(cpu_utilization)
        
        # Calculate average for each hour
        return {
            hour: statistics.mean(values) 
            for hour, values in daily_pattern.items()
        }
    
    def _extract_weekly_pattern(self, metrics: Dict[str, List]) -> Dict[int, float]:
        """Extract weekly usage pattern (7-day cycle)"""
        weekly_pattern = defaultdict(list)
        
        for i, timestamp in enumerate(metrics['timestamps']):
            day_of_week = timestamp.weekday()  # 0 = Monday
            cpu_utilization = metrics['cpu_utilization'][i]
            weekly_pattern[day_of_week].append(cpu_utilization)
        
        return {
            day: statistics.mean(values) 
            for day, values in weekly_pattern.items()
        }
    
    def _extract_monthly_pattern(self, metrics: Dict[str, List]) -> Dict[int, float]:
        """Extract monthly usage pattern"""
        monthly_pattern = defaultdict(list)
        
        for i, timestamp in enumerate(metrics['timestamps']):
            day_of_month = timestamp.day
            cpu_utilization = metrics['cpu_utilization'][i]
            monthly_pattern[day_of_month].append(cpu_utilization)
        
        return {
            day: statistics.mean(values) 
            for day, values in monthly_pattern.items()
        }
    
    def _calculate_utilization_stats(self, metrics: Dict[str, List]) -> Dict[str, float]:
        """Calculate comprehensive utilization statistics"""
        cpu_values = metrics['cpu_utilization']
        
        if not cpu_values:
            return {}
        
        return {
            'mean': statistics.mean(cpu_values),
            'median': statistics.median(cpu_values),
            'min': min(cpu_values),
            'max': max(cpu_values),
            'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
            'percentile_95': sorted(cpu_values)[int(len(cpu_values) * 0.95)],
            'percentile_75': sorted(cpu_values)[int(len(cpu_values) * 0.75)],
            'percentile_25': sorted(cpu_values)[int(len(cpu_values) * 0.25)],
            'zero_utilization_pct': (cpu_values.count(0) / len(cpu_values)) * 100,
            'low_utilization_pct': (len([x for x in cpu_values if x < 10]) / len(cpu_values)) * 100
        }
    
    def _calculate_seasonality_score(self, metrics: Dict[str, List]) -> float:
        """Calculate seasonality score using autocorrelation"""
        cpu_values = metrics['cpu_utilization']
        
        if len(cpu_values) < 48:  # Need at least 2 days of data
            return 0.0
        
        try:
            # Simple autocorrelation for daily pattern (24-hour lag)
            if len(cpu_values) >= 24:
                daily_correlation = np.corrcoef(cpu_values[:-24], cpu_values[24:])[0, 1]
                daily_correlation = max(0, daily_correlation)  # Only positive correlation
            else:
                daily_correlation = 0
            
            # Weekly pattern (168-hour lag) if we have enough data
            if len(cpu_values) >= 168:
                weekly_correlation = np.corrcoef(cpu_values[:-168], cpu_values[168:])[0, 1]
                weekly_correlation = max(0, weekly_correlation)
            else:
                weekly_correlation = 0
            
            # Combine daily and weekly seasonality
            seasonality_score = max(daily_correlation, weekly_correlation)
            
            return min(1.0, seasonality_score)
            
        except Exception as e:
            logger.warning(f"Error calculating seasonality: {e}")
            return 0.0
    
    def _calculate_variability_score(self, metrics: Dict[str, List]) -> float:
        """Calculate variability score (coefficient of variation)"""
        cpu_values = metrics['cpu_utilization']
        
        if not cpu_values:
            return 1.0
        
        mean_value = statistics.mean(cpu_values)
        if mean_value == 0:
            return 1.0
        
        std_dev = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
        coefficient_of_variation = std_dev / mean_value
        
        # Normalize to 0-1 scale
        return min(1.0, coefficient_of_variation)
    
    def _calculate_predictability_score(self, metrics: Dict[str, List]) -> float:
        """Calculate predictability score based on pattern regularity"""
        
        # High seasonality + low variability = high predictability
        seasonality = self._calculate_seasonality_score(metrics)
        variability = self._calculate_variability_score(metrics)
        
        predictability = seasonality * (1 - variability)
        
        return max(0.0, min(1.0, predictability))
    
    def _classify_workload_type(self, daily_pattern: Dict[int, float], 
                              weekly_pattern: Dict[int, float],
                              utilization_stats: Dict[str, float], 
                              tags: Dict[str, str]) -> WorkloadType:
        """Classify workload type based on patterns and context"""
        
        if not daily_pattern or not utilization_stats:
            return WorkloadType.UNKNOWN
        
        # Check tags for hints
        environment = tags.get('Environment', '').lower()
        workload_hint = tags.get('WorkloadType', '').lower()
        
        if environment in ['dev', 'development']:
            return WorkloadType.DEVELOPMENT
        elif environment in ['test', 'testing']:
            return WorkloadType.TESTING
        elif workload_hint in ['batch', 'processing']:
            return WorkloadType.BATCH_PROCESSING
        
        # Analyze patterns
        mean_utilization = utilization_stats.get('mean', 0)
        variability = utilization_stats.get('std_dev', 0) / mean_utilization if mean_utilization > 0 else 0
        
        # Get peak and off-peak hours
        if len(daily_pattern) >= 12:  # Have enough hourly data
            max_hour = max(daily_pattern, key=daily_pattern.get)
            min_hour = min(daily_pattern, key=daily_pattern.get)
            peak_valley_ratio = daily_pattern[max_hour] / max(1, daily_pattern[min_hour])
        else:
            peak_valley_ratio = 1
        
        # Classification logic
        if variability < 0.2 and mean_utilization > 10:
            return WorkloadType.STEADY_STATE
        elif peak_valley_ratio > 3 and len([h for h, u in daily_pattern.items() if u > mean_utilization * 1.5]) < 6:
            return WorkloadType.BATCH_PROCESSING
        elif variability > 0.8:
            return WorkloadType.SPIKY
        elif self._has_seasonal_pattern(weekly_pattern, daily_pattern):
            return WorkloadType.SEASONAL
        else:
            return WorkloadType.UNKNOWN
    
    def _has_seasonal_pattern(self, weekly_pattern: Dict[int, float], 
                            daily_pattern: Dict[int, float]) -> bool:
        """Check if workload has seasonal patterns"""
        
        # Check for weekend vs weekday differences
        if len(weekly_pattern) >= 7:
            weekday_avg = statistics.mean([weekly_pattern.get(i, 0) for i in range(5)])  # Mon-Fri
            weekend_avg = statistics.mean([weekly_pattern.get(i, 0) for i in range(5, 7)])  # Sat-Sun
            
            if weekday_avg > 0 and abs(weekday_avg - weekend_avg) / weekday_avg > 0.3:
                return True
        
        # Check for business hours pattern
        if len(daily_pattern) >= 24:
            business_hours = [daily_pattern.get(i, 0) for i in range(9, 17)]  # 9 AM - 5 PM
            off_hours = [daily_pattern.get(i, 0) for i in list(range(0, 9)) + list(range(17, 24))]
            
            if business_hours and off_hours:
                business_avg = statistics.mean(business_hours)
                off_avg = statistics.mean(off_hours)
                
                if business_avg > 0 and (business_avg - off_avg) / business_avg > 0.4:
                    return True
        
        return False
    
    def _determine_usage_phase(self, metrics: Dict[str, List], tags: Dict[str, str]) -> UsagePhase:
        """Determine the usage phase of the resource"""
        
        cpu_values = metrics['cpu_utilization']
        
        if len(cpu_values) < 30:  # Need at least 30 data points
            return UsagePhase.STARTUP
        
        # Split data into periods to analyze trend
        periods = 3
        period_size = len(cpu_values) // periods
        period_averages = []
        
        for i in range(periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size
            period_avg = statistics.mean(cpu_values[start_idx:end_idx])
            period_averages.append(period_avg)
        
        # Analyze trend
        if len(period_averages) >= 2:
            first_period = period_averages[0]
            last_period = period_averages[-1]
            
            if first_period > 0:
                growth_rate = (last_period - first_period) / first_period
                
                if growth_rate > 0.2:  # 20% growth
                    return UsagePhase.GROWTH
                elif growth_rate < -0.2:  # 20% decline
                    return UsagePhase.DECLINING
                else:
                    # Check if it's mature (stable) or startup (low usage)
                    avg_utilization = statistics.mean(cpu_values)
                    if avg_utilization > 5:
                        return UsagePhase.MATURE
                    else:
                        return UsagePhase.STARTUP
        
        return UsagePhase.MATURE
    
    def _infer_business_criticality(self, tags: Dict[str, str], workload_type: WorkloadType) -> str:
        """Infer business criticality from tags and patterns"""
        
        # Check explicit tags
        criticality = tags.get('Criticality', '').lower()
        if criticality in ['high', 'critical']:
            return 'high'
        elif criticality in ['medium', 'moderate']:
            return 'medium'
        elif criticality in ['low']:
            return 'low'
        
        # Infer from environment
        environment = tags.get('Environment', '').lower()
        if environment in ['prod', 'production']:
            return 'high'
        elif environment in ['staging', 'stage']:
            return 'medium'
        elif environment in ['dev', 'development', 'test', 'testing']:
            return 'low'
        
        # Infer from workload type
        if workload_type == WorkloadType.STEADY_STATE:
            return 'high'
        elif workload_type in [WorkloadType.DEVELOPMENT, WorkloadType.TESTING]:
            return 'low'
        else:
            return 'unknown'
    
    def _infer_environment_type(self, tags: Dict[str, str]) -> str:
        """Infer environment type from tags"""
        
        environment = tags.get('Environment', '').lower()
        env_keywords = {
            'prod': ['prod', 'production', 'prd'],
            'staging': ['staging', 'stage', 'stg'],
            'test': ['test', 'testing', 'tst'],
            'dev': ['dev', 'development', 'develop']
        }
        
        for env_type, keywords in env_keywords.items():
            if any(keyword in environment for keyword in keywords):
                return env_type
        
        # Check other tag values
        for tag_value in tags.values():
            tag_value_lower = tag_value.lower()
            for env_type, keywords in env_keywords.items():
                if any(keyword in tag_value_lower for keyword in keywords):
                    return env_type
        
        return 'unknown'
    
    def _infer_cost_sensitivity(self, tags: Dict[str, str], environment_type: str) -> str:
        """Infer cost sensitivity"""
        
        # Development and testing are typically more cost-sensitive
        if environment_type in ['dev', 'test']:
            return 'high'
        elif environment_type == 'staging':
            return 'medium'
        elif environment_type == 'prod':
            return 'low'  # Production typically prioritizes availability over cost
        
        return 'medium'
    
    def _identify_optimization_opportunities(self, workload_type: WorkloadType,
                                           utilization_stats: Dict[str, float],
                                           daily_pattern: Dict[int, float],
                                           weekly_pattern: Dict[int, float]) -> List[str]:
        """Identify optimization opportunities based on patterns"""
        
        opportunities = []
        
        if not utilization_stats:
            return opportunities
        
        mean_utilization = utilization_stats.get('mean', 0)
        max_utilization = utilization_stats.get('max', 0)
        low_utilization_pct = utilization_stats.get('low_utilization_pct', 0)
        
        # Low utilization opportunities
        if mean_utilization < 10:
            opportunities.append("Consider stopping or terminating - very low utilization")
        elif mean_utilization < 25:
            opportunities.append("Rightsizing opportunity - consistently low utilization")
        
        # Scheduling opportunities
        if workload_type in [WorkloadType.DEVELOPMENT, WorkloadType.TESTING]:
            opportunities.append("Scheduling opportunity - dev/test workload")
        
        if self._has_seasonal_pattern(weekly_pattern, daily_pattern):
            opportunities.append("Scheduling opportunity - clear business hours pattern")
        
        # RI opportunities
        if workload_type == WorkloadType.STEADY_STATE and mean_utilization > 10:
            opportunities.append("Reserved Instance opportunity - steady workload")
        
        # Spot instance opportunities
        if workload_type in [WorkloadType.BATCH_PROCESSING, WorkloadType.DEVELOPMENT, WorkloadType.TESTING]:
            opportunities.append("Spot instance opportunity - fault-tolerant workload")
        
        # Storage optimization
        if low_utilization_pct > 70:
            opportunities.append("Storage optimization - resource rarely used")
        
        return opportunities
    
    def _assess_scheduling_potential(self, daily_pattern: Dict[int, float],
                                   weekly_pattern: Dict[int, float],
                                   environment_type: str) -> bool:
        """Assess if resource is suitable for scheduling"""
        
        # Development and testing environments are good candidates
        if environment_type in ['dev', 'test']:
            return True
        
        # Check for clear business hours pattern
        if self._has_seasonal_pattern(weekly_pattern, daily_pattern):
            return True
        
        # Check if there are long periods of low usage
        if len(daily_pattern) >= 24:
            low_usage_hours = len([u for u in daily_pattern.values() if u < 5])
            if low_usage_hours >= 8:  # 8+ hours of low usage
                return True
        
        return False
    
    def _assess_rightsizing_potential(self, utilization_stats: Dict[str, float]) -> bool:
        """Assess if resource needs rightsizing"""
        
        if not utilization_stats:
            return False
        
        mean_utilization = utilization_stats.get('mean', 0)
        percentile_95 = utilization_stats.get('percentile_95', 0)
        
        # High potential if consistently underutilized
        if mean_utilization < 20 and percentile_95 < 50:
            return True
        
        return False
    
    def _calculate_ri_suitability(self, workload_type: WorkloadType,
                                 predictability_score: float,
                                 utilization_stats: Dict[str, float]) -> float:
        """Calculate Reserved Instance suitability score (0-1)"""
        
        score = 0.0
        
        # Base score from workload type
        workload_scores = {
            WorkloadType.STEADY_STATE: 0.8,
            WorkloadType.SEASONAL: 0.6,
            WorkloadType.SPIKY: 0.2,
            WorkloadType.BATCH_PROCESSING: 0.3,
            WorkloadType.DEVELOPMENT: 0.1,
            WorkloadType.TESTING: 0.1,
            WorkloadType.UNKNOWN: 0.3
        }
        
        score += workload_scores.get(workload_type, 0.3)
        
        # Adjust for predictability
        score += predictability_score * 0.2
        
        # Adjust for utilization
        mean_utilization = utilization_stats.get('mean', 0)
        if mean_utilization > 10:
            score += 0.2
        elif mean_utilization > 5:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_analysis_confidence(self, data_points: int, variability: float, predictability: float) -> float:
        """Calculate confidence in the pattern analysis"""
        
        confidence = 0.5  # Base confidence
        
        # More data points = higher confidence
        if data_points >= 500:
            confidence += 0.3
        elif data_points >= 200:
            confidence += 0.2
        elif data_points >= 100:
            confidence += 0.1
        
        # Lower variability = higher confidence
        if variability < 0.3:
            confidence += 0.2
        elif variability < 0.5:
            confidence += 0.1
        
        # Higher predictability = higher confidence
        confidence += predictability * 0.2
        
        return min(1.0, confidence)
    
    def _analyze_rds_patterns(self, region: str) -> List[ResourcePattern]:
        """Analyze RDS instance patterns (simplified)"""
        patterns = []
        
        try:
            rds = boto3.client('rds', region_name=region)
            
            # Get RDS instances
            paginator = rds.get_paginator('describe_db_instances')
            
            for page in paginator.paginate():
                for instance in page['DBInstances']:
                    if instance['DBInstanceStatus'] == 'available':
                        
                        # RDS typically has steady patterns
                        pattern = ResourcePattern(
                            resource_id=instance['DBInstanceIdentifier'],
                            resource_type='rds',
                            region=region,
                            workload_type=WorkloadType.STEADY_STATE,
                            usage_phase=UsagePhase.MATURE,
                            daily_pattern={i: 50.0 for i in range(24)},  # Assume steady 50%
                            weekly_pattern={i: 50.0 for i in range(7)},
                            monthly_pattern={i: 50.0 for i in range(31)},
                            utilization_stats={'mean': 50.0, 'std_dev': 5.0},
                            seasonality_score=0.1,
                            variability_score=0.1,
                            predictability_score=0.9,
                            business_criticality='high',
                            environment_type='prod',
                            cost_sensitivity='low',
                            optimization_opportunities=['Reserved Instance opportunity'],
                            scheduling_potential=False,
                            rightsizing_potential=False,
                            ri_suitability=0.9,
                            analysis_period_days=self.analysis_period_days,
                            confidence_score=0.8,
                            last_analyzed=datetime.utcnow()
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.error(f"Error analyzing RDS patterns in {region}: {e}")
        
        return patterns
    
    def _analyze_ebs_patterns(self, region: str) -> List[ResourcePattern]:
        """Analyze EBS volume patterns (simplified)"""
        patterns = []
        
        try:
            ec2 = boto3.client('ec2', region_name=region)
            
            # Get EBS volumes
            paginator = ec2.get_paginator('describe_volumes')
            
            for page in paginator.paginate():
                for volume in page['Volumes']:
                    
                    # Simple pattern for EBS volumes
                    if volume['State'] == 'available':  # Unattached
                        workload_type = WorkloadType.UNKNOWN
                        optimization_opportunities = ['Delete unattached volume']
                        scheduling_potential = False
                    else:
                        workload_type = WorkloadType.STEADY_STATE
                        optimization_opportunities = []
                        scheduling_potential = False
                    
                    pattern = ResourcePattern(
                        resource_id=volume['VolumeId'],
                        resource_type='ebs',
                        region=region,
                        workload_type=workload_type,
                        usage_phase=UsagePhase.MATURE,
                        daily_pattern={},
                        weekly_pattern={},
                        monthly_pattern={},
                        utilization_stats={},
                        seasonality_score=0.0,
                        variability_score=0.1,
                        predictability_score=0.8,
                        business_criticality='medium',
                        environment_type='unknown',
                        cost_sensitivity='medium',
                        optimization_opportunities=optimization_opportunities,
                        scheduling_potential=scheduling_potential,
                        rightsizing_potential=False,
                        ri_suitability=0.0,
                        analysis_period_days=self.analysis_period_days,
                        confidence_score=0.6,
                        last_analyzed=datetime.utcnow()
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.error(f"Error analyzing EBS patterns in {region}: {e}")
        
        return patterns
    
    def _get_ec2_instances(self, ec2_client) -> List[Dict]:
        """Get all EC2 instances"""
        instances = []
        
        try:
            paginator = ec2_client.get_paginator('describe_instances')
            
            for page in paginator.paginate():
                for reservation in page['Reservations']:
                    instances.extend(reservation['Instances'])
                    
        except Exception as e:
            logger.error(f"Error getting EC2 instances: {e}")
        
        return instances
    
    def _group_patterns_by_workload(self, patterns: List[ResourcePattern]) -> Dict[str, List[ResourcePattern]]:
        """Group resource patterns by workload/application"""
        
        workload_groups = defaultdict(list)
        
        for pattern in patterns:
            # Try to infer workload grouping from resource naming and tags
            group_name = self._infer_workload_group(pattern)
            workload_groups[group_name].append(pattern)
        
        return dict(workload_groups)
    
    def _infer_workload_group(self, pattern: ResourcePattern) -> str:
        """Infer workload group from resource naming patterns"""
        
        resource_id = pattern.resource_id.lower()
        
        # Common naming patterns
        if 'web' in resource_id:
            return 'web-application'
        elif 'db' in resource_id or pattern.resource_type == 'rds':
            return 'database'
        elif 'api' in resource_id:
            return 'api-services'
        elif 'batch' in resource_id:
            return 'batch-processing'
        elif pattern.environment_type == 'dev':
            return 'development'
        elif pattern.environment_type == 'test':
            return 'testing'
        else:
            return f"{pattern.resource_type}-{pattern.environment_type}"
    
    def _analyze_workload_group(self, group_name: str, patterns: List[ResourcePattern]) -> WorkloadCharacteristics:
        """Analyze characteristics of a workload group"""
        
        if not patterns:
            return WorkloadCharacteristics(
                resource_group=group_name,
                workload_patterns=patterns,
                overall_workload_type=WorkloadType.UNKNOWN,
                peak_hours=[],
                low_usage_hours=[],
                weekend_usage_ratio=1.0,
                total_monthly_cost=0.0,
                optimization_potential_pct=0.0,
                estimated_monthly_savings=0.0,
                architectural_recommendations=[],
                operational_recommendations=[],
                cost_optimization_score=0.0
            )
        
        # Determine overall workload type (most common)
        workload_types = [p.workload_type for p in patterns]
        overall_workload_type = max(set(workload_types), key=workload_types.count)
        
        # Aggregate daily patterns
        aggregated_daily = defaultdict(list)
        for pattern in patterns:
            for hour, utilization in pattern.daily_pattern.items():
                aggregated_daily[hour].append(utilization)
        
        avg_daily_pattern = {
            hour: statistics.mean(values) 
            for hour, values in aggregated_daily.items()
        }
        
        # Find peak and low usage hours
        if avg_daily_pattern:
            sorted_hours = sorted(avg_daily_pattern.items(), key=lambda x: x[1], reverse=True)
            peak_hours = [hour for hour, _ in sorted_hours[:4]]  # Top 4 hours
            low_usage_hours = [hour for hour, _ in sorted_hours[-4:]]  # Bottom 4 hours
        else:
            peak_hours = []
            low_usage_hours = []
        
        # Calculate weekend usage ratio
        weekend_usage_ratio = self._calculate_weekend_usage_ratio(patterns)
        
        # Estimate costs and savings (simplified)
        total_monthly_cost = len(patterns) * 100  # Simplified estimate
        optimization_potential = sum(1 for p in patterns if p.optimization_opportunities)
        optimization_potential_pct = (optimization_potential / len(patterns)) * 100
        estimated_monthly_savings = total_monthly_cost * (optimization_potential_pct / 100) * 0.3
        
        # Generate recommendations
        architectural_recommendations = self._generate_architectural_recommendations(
            overall_workload_type, patterns
        )
        operational_recommendations = self._generate_operational_recommendations(
            overall_workload_type, patterns, avg_daily_pattern
        )
        
        # Calculate cost optimization score
        cost_optimization_score = self._calculate_cost_optimization_score(patterns)
        
        return WorkloadCharacteristics(
            resource_group=group_name,
            workload_patterns=patterns,
            overall_workload_type=overall_workload_type,
            peak_hours=peak_hours,
            low_usage_hours=low_usage_hours,
            weekend_usage_ratio=weekend_usage_ratio,
            total_monthly_cost=total_monthly_cost,
            optimization_potential_pct=optimization_potential_pct,
            estimated_monthly_savings=estimated_monthly_savings,
            architectural_recommendations=architectural_recommendations,
            operational_recommendations=operational_recommendations,
            cost_optimization_score=cost_optimization_score
        )
    
    def _calculate_weekend_usage_ratio(self, patterns: List[ResourcePattern]) -> float:
        """Calculate weekend vs weekday usage ratio"""
        
        weekend_usage = []
        weekday_usage = []
        
        for pattern in patterns:
            if len(pattern.weekly_pattern) >= 7:
                weekday_avg = statistics.mean([pattern.weekly_pattern.get(i, 0) for i in range(5)])
                weekend_avg = statistics.mean([pattern.weekly_pattern.get(i, 0) for i in range(5, 7)])
                
                if weekday_avg > 0:
                    weekend_usage.append(weekend_avg)
                    weekday_usage.append(weekday_avg)
        
        if weekend_usage and weekday_usage:
            avg_weekend = statistics.mean(weekend_usage)
            avg_weekday = statistics.mean(weekday_usage)
            return avg_weekend / avg_weekday if avg_weekday > 0 else 1.0
        
        return 1.0
    
    def _generate_architectural_recommendations(self, workload_type: WorkloadType, 
                                              patterns: List[ResourcePattern]) -> List[str]:
        """Generate architectural recommendations"""
        
        recommendations = []
        
        if workload_type == WorkloadType.BATCH_PROCESSING:
            recommendations.extend([
                "Consider using spot instances for cost savings",
                "Implement auto-scaling based on queue depth",
                "Use containerized workloads for better resource utilization"
            ])
        
        elif workload_type == WorkloadType.DEVELOPMENT:
            recommendations.extend([
                "Implement auto-shutdown for development resources",
                "Use smaller instance types for development",
                "Consider shared development environments"
            ])
        
        elif workload_type == WorkloadType.STEADY_STATE:
            recommendations.extend([
                "Purchase Reserved Instances for predictable savings",
                "Consider migrating to Graviton instances",
                "Implement proper auto-scaling policies"
            ])
        
        # Add recommendations based on resource mix
        ec2_count = len([p for p in patterns if p.resource_type == 'ec2'])
        rds_count = len([p for p in patterns if p.resource_type == 'rds'])
        
        if ec2_count > 10:
            recommendations.append("Consider implementing a service mesh for microservices")
        
        if rds_count > 1:
            recommendations.append("Evaluate database consolidation opportunities")
        
        return recommendations
    
    def _generate_operational_recommendations(self, workload_type: WorkloadType,
                                            patterns: List[ResourcePattern],
                                            daily_pattern: Dict[int, float]) -> List[str]:
        """Generate operational recommendations"""
        
        recommendations = []
        
        # Scheduling recommendations
        if daily_pattern and len(daily_pattern) >= 12:
            max_hour = max(daily_pattern, key=daily_pattern.get)
            min_hour = min(daily_pattern, key=daily_pattern.get)
            
            if daily_pattern[max_hour] / max(1, daily_pattern[min_hour]) > 2:
                recommendations.append(f"Schedule scaling events around peak hour {max_hour}:00")
                recommendations.append(f"Consider scheduled shutdown during low usage hour {min_hour}:00")
        
        # Environment-specific recommendations
        dev_test_count = len([p for p in patterns if p.environment_type in ['dev', 'test']])
        if dev_test_count > 0:
            recommendations.extend([
                "Implement automated start/stop scheduling for dev/test environments",
                "Use snapshots instead of keeping dev/test databases running 24/7"
            ])
        
        # Cost monitoring recommendations
        high_cost_resources = len([p for p in patterns if len(p.optimization_opportunities) > 2])
        if high_cost_resources > 0:
            recommendations.extend([
                "Set up cost alerts for high-spend resources",
                "Implement regular cost optimization reviews"
            ])
        
        return recommendations
    
    def _calculate_cost_optimization_score(self, patterns: List[ResourcePattern]) -> float:
        """Calculate overall cost optimization score (0-100)"""
        
        if not patterns:
            return 0.0
        
        # Factors that contribute to optimization score
        high_utilization = len([p for p in patterns if p.utilization_stats.get('mean', 0) > 50])
        has_ri_opportunity = len([p for p in patterns if p.ri_suitability > 0.7])
        has_scheduling_potential = len([p for p in patterns if p.scheduling_potential])
        has_rightsizing_potential = len([p for p in patterns if p.rightsizing_potential])
        
        total_resources = len(patterns)
        
        # Calculate score components
        utilization_score = (high_utilization / total_resources) * 30
        ri_score = (has_ri_opportunity / total_resources) * 25
        scheduling_score = (has_scheduling_potential / total_resources) * 25
        rightsizing_score = (1 - has_rightsizing_potential / total_resources) * 20  # Inverse for rightsizing
        
        total_score = utilization_score + ri_score + scheduling_score + rightsizing_score
        
        return min(100.0, total_score)
    
    def _load_workload_classification_rules(self) -> Dict:
        """Load workload classification rules"""
        
        return {
            'steady_state': {
                'cpu_variability': 0.3,
                'min_utilization': 10,
                'pattern_consistency': 0.7
            },
            'batch_processing': {
                'peak_duration_hours': 6,
                'peak_valley_ratio': 3,
                'off_period_utilization': 5
            },
            'development': {
                'business_hours_only': True,
                'weekend_usage_ratio': 0.2,
                'avg_utilization': 20
            }
        }
    
    def export_pattern_analysis(self, patterns: List[ResourcePattern], 
                               workload_characteristics: List[WorkloadCharacteristics],
                               output_file: str):
        """Export pattern analysis to Excel file"""
        
        if not patterns:
            logger.warning("No patterns to export")
            return
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Resource patterns sheet
            pattern_data = []
            for pattern in patterns:
                pattern_data.append({
                    'Resource ID': pattern.resource_id,
                    'Resource Type': pattern.resource_type,
                    'Region': pattern.region,
                    'Workload Type': pattern.workload_type.value,
                    'Usage Phase': pattern.usage_phase.value,
                    'Environment': pattern.environment_type,
                    'Business Criticality': pattern.business_criticality,
                    'Mean Utilization %': f"{pattern.utilization_stats.get('mean', 0):.1f}%",
                    'Seasonality Score': f"{pattern.seasonality_score:.2f}",
                    'Variability Score': f"{pattern.variability_score:.2f}",
                    'Predictability Score': f"{pattern.predictability_score:.2f}",
                    'RI Suitability': f"{pattern.ri_suitability:.2f}",
                    'Scheduling Potential': pattern.scheduling_potential,
                    'Rightsizing Potential': pattern.rightsizing_potential,
                    'Optimization Opportunities': '; '.join(pattern.optimization_opportunities),
                    'Confidence Score': f"{pattern.confidence_score:.2f}",
                    'Analysis Date': pattern.last_analyzed.strftime('%Y-%m-%d')
                })
            
            pd.DataFrame(pattern_data).to_excel(writer, sheet_name='Resource Patterns', index=False)
            
            # Workload characteristics sheet
            if workload_characteristics:
                workload_data = []
                for wc in workload_characteristics:
                    workload_data.append({
                        'Workload Group': wc.resource_group,
                        'Overall Type': wc.overall_workload_type.value,
                        'Resource Count': len(wc.workload_patterns),
                        'Peak Hours': ', '.join(map(str, wc.peak_hours)),
                        'Low Usage Hours': ', '.join(map(str, wc.low_usage_hours)),
                        'Weekend Usage Ratio': f"{wc.weekend_usage_ratio:.2f}",
                        'Estimated Monthly Cost': f"${wc.total_monthly_cost:.2f}",
                        'Optimization Potential %': f"{wc.optimization_potential_pct:.1f}%",
                        'Estimated Savings': f"${wc.estimated_monthly_savings:.2f}",
                        'Cost Optimization Score': f"{wc.cost_optimization_score:.1f}",
                        'Architectural Recommendations': '; '.join(wc.architectural_recommendations),
                        'Operational Recommendations': '; '.join(wc.operational_recommendations)
                    })
                
                pd.DataFrame(workload_data).to_excel(writer, sheet_name='Workload Analysis', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Resources Analyzed',
                    'Workload Groups Identified',
                    'High Confidence Patterns',
                    'RI Suitable Resources',
                    'Scheduling Candidates',
                    'Rightsizing Candidates',
                    'Development Resources',
                    'Production Resources'
                ],
                'Value': [
                    len(patterns),
                    len(workload_characteristics),
                    len([p for p in patterns if p.confidence_score > 0.8]),
                    len([p for p in patterns if p.ri_suitability > 0.7]),
                    len([p for p in patterns if p.scheduling_potential]),
                    len([p for p in patterns if p.rightsizing_potential]),
                    len([p for p in patterns if p.environment_type == 'dev']),
                    len([p for p in patterns if p.environment_type == 'prod'])
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Exported pattern analysis for {len(patterns)} resources to {output_file}")
