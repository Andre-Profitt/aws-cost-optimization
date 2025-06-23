"""
Complete RDS Optimizer - Analyzes RDS instances and databases for cost optimization
"""
import boto3
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class RDSOptimizationAction(Enum):
    """Types of RDS optimization actions"""
    STOP_INSTANCE = "stop_instance"
    DOWNSIZE_INSTANCE = "downsize_instance"
    CHANGE_INSTANCE_TYPE = "change_instance_type"
    CONVERT_TO_SERVERLESS = "convert_to_serverless"
    PURCHASE_RESERVED_INSTANCE = "purchase_reserved_instance"
    ENABLE_AURORA_SERVERLESS = "enable_aurora_serverless"
    OPTIMIZE_STORAGE = "optimize_storage"
    DELETE_UNUSED_SNAPSHOTS = "delete_unused_snapshots"
    SCHEDULE_INSTANCE = "schedule_instance"
    MIGRATE_TO_AURORA = "migrate_to_aurora"
    ENABLE_PERFORMANCE_INSIGHTS = "enable_performance_insights"
    OPTIMIZE_BACKUP_RETENTION = "optimize_backup_retention"
    NO_ACTION = "no_action"

@dataclass
class RDSRecommendation:
    """RDS optimization recommendation"""
    instance_identifier: str
    engine: str
    instance_class: str
    region: str
    action: RDSOptimizationAction
    current_monthly_cost: float
    estimated_monthly_savings: float
    annual_savings: float
    confidence_score: float
    reason: str
    details: Dict[str, Any]
    risk_level: str
    prerequisites: List[str]
    implementation_effort: str
    rollback_plan: str
    
    # Metrics that led to recommendation
    cpu_avg: float
    cpu_max: float
    memory_avg: Optional[float] = None
    connections_avg: Optional[float] = None
    iops_avg: Optional[float] = None
    database_connections: Optional[int] = None
    
    # Instance details
    multi_az: bool = False
    backup_retention_period: int = 7
    storage_type: str = "gp2"
    allocated_storage: int = 0
    storage_encrypted: bool = False
    
    # Recommendation-specific data
    recommended_instance_class: Optional[str] = None
    recommended_storage_type: Optional[str] = None
    serverless_configuration: Optional[Dict[str, Any]] = None

class RDSOptimizer:
    """Complete RDS optimization analyzer"""
    
    def __init__(self,
                 cpu_threshold: float = 20.0,
                 memory_threshold: float = 30.0,
                 connections_threshold: float = 10.0,
                 lookback_days: int = 14,
                 confidence_threshold: float = 0.7,
                 min_savings_threshold: float = 10.0):
        """
        Initialize RDS optimizer
        
        Args:
            cpu_threshold: CPU utilization threshold for rightsizing (%)
            memory_threshold: Memory utilization threshold (%)
            connections_threshold: Database connections threshold (%)
            lookback_days: Days of metrics to analyze
            confidence_threshold: Minimum confidence for recommendations
            min_savings_threshold: Minimum monthly savings to recommend
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.connections_threshold = connections_threshold
        self.lookback_days = lookback_days
        self.confidence_threshold = confidence_threshold
        self.min_savings_threshold = min_savings_threshold
        
        # Instance class families for rightsizing
        self.instance_families = self._load_instance_families()
        self.pricing_data = {}
    
    def _load_instance_families(self) -> Dict[str, List[str]]:
        """Load RDS instance class families for rightsizing"""
        return {
            'db.t3': ['db.t3.micro', 'db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge', 'db.t3.2xlarge'],
            'db.t4g': ['db.t4g.micro', 'db.t4g.small', 'db.t4g.medium', 'db.t4g.large', 'db.t4g.xlarge', 'db.t4g.2xlarge'],
            'db.m6i': ['db.m6i.large', 'db.m6i.xlarge', 'db.m6i.2xlarge', 'db.m6i.4xlarge', 'db.m6i.8xlarge', 'db.m6i.12xlarge', 'db.m6i.16xlarge', 'db.m6i.24xlarge', 'db.m6i.32xlarge'],
            'db.m5': ['db.m5.large', 'db.m5.xlarge', 'db.m5.2xlarge', 'db.m5.4xlarge', 'db.m5.8xlarge', 'db.m5.12xlarge', 'db.m5.16xlarge', 'db.m5.24xlarge'],
            'db.r6i': ['db.r6i.large', 'db.r6i.xlarge', 'db.r6i.2xlarge', 'db.r6i.4xlarge', 'db.r6i.8xlarge', 'db.r6i.12xlarge', 'db.r6i.16xlarge', 'db.r6i.24xlarge', 'db.r6i.32xlarge'],
            'db.r5': ['db.r5.large', 'db.r5.xlarge', 'db.r5.2xlarge', 'db.r5.4xlarge', 'db.r5.8xlarge', 'db.r5.12xlarge', 'db.r5.16xlarge', 'db.r5.24xlarge']
        }
    
    def analyze_all_databases(self, region_name: str = None) -> List[RDSRecommendation]:
        """Analyze all RDS instances across regions"""
        recommendations = []
        
        if region_name:
            regions = [region_name]
        else:
            ec2 = boto3.client('ec2')
            regions = [region['RegionName'] for region in ec2.describe_regions()['Regions']]
        
        for region in regions:
            try:
                logger.info(f"Analyzing RDS instances in region: {region}")
                regional_recommendations = self._analyze_region_databases(region)
                recommendations.extend(regional_recommendations)
            except Exception as e:
                logger.error(f"Error analyzing region {region}: {e}")
                continue
        
        # Sort by potential savings
        recommendations.sort(key=lambda x: x.estimated_monthly_savings, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} RDS recommendations")
        return recommendations
    
    def _analyze_region_databases(self, region: str) -> List[RDSRecommendation]:
        """Analyze RDS instances in a specific region"""
        recommendations = []
        
        try:
            rds = boto3.client('rds', region_name=region)
            cloudwatch = boto3.client('cloudwatch', region_name=region)
            
            # Get all RDS instances
            paginator = rds.get_paginator('describe_db_instances')
            instances = []
            
            for page in paginator.paginate():
                instances.extend(page['DBInstances'])
            
            if not instances:
                return recommendations
            
            # Load pricing data for the region
            self._load_pricing_data(region)
            
            # Analyze each instance
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_instance = {
                    executor.submit(self._analyze_single_database, instance, cloudwatch, region): instance
                    for instance in instances
                }
                
                for future in as_completed(future_to_instance):
                    instance = future_to_instance[future]
                    try:
                        db_recommendations = future.result()
                        if db_recommendations:
                            recommendations.extend(db_recommendations)
                    except Exception as e:
                        logger.error(f"Error analyzing database {instance.get('DBInstanceIdentifier', 'unknown')}: {e}")
            
            # Also analyze snapshots for cleanup opportunities
            snapshot_recommendations = self._analyze_snapshots(rds, region)
            recommendations.extend(snapshot_recommendations)
            
        except Exception as e:
            logger.error(f"Error in region {region}: {e}")
        
        return recommendations
    
    def _analyze_single_database(self, instance: Dict, cloudwatch, region: str) -> List[RDSRecommendation]:
        """Analyze a single RDS instance"""
        recommendations = []
        
        instance_id = instance['DBInstanceIdentifier']
        instance_class = instance['DBInstanceClass']
        engine = instance['Engine']
        status = instance['DBInstanceStatus']
        
        # Skip if not available
        if status != 'available':
            return recommendations
        
        # Get instance tags
        tags = {}
        try:
            rds = boto3.client('rds', region_name=region)
            response = rds.list_tags_for_resource(
                ResourceName=instance['DBInstanceArn']
            )
            tags = {tag['Key']: tag['Value'] for tag in response['TagList']}
        except Exception as e:
            logger.warning(f"Could not get tags for {instance_id}: {e}")
        
        # Skip if tagged as do not optimize
        if tags.get('DoNotOptimize', '').lower() == 'true':
            return recommendations
        
        try:
            # Get metrics
            metrics = self._get_database_metrics(instance_id, cloudwatch)
            
            # Calculate current cost
            current_monthly_cost = self._calculate_monthly_cost(instance, region)
            
            # Generate recommendations
            instance_recommendations = self._generate_database_recommendations(
                instance, metrics, current_monthly_cost, region, tags
            )
            
            recommendations.extend(instance_recommendations)
            
        except Exception as e:
            logger.error(f"Error analyzing database {instance_id}: {e}")
        
        return recommendations
    
    def _get_database_metrics(self, instance_id: str, cloudwatch) -> Dict[str, float]:
        """Get CloudWatch metrics for a database instance"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.lookback_days)
        
        metrics = {}
        
        # Define metrics to collect
        metric_queries = [
            ('AWS/RDS', 'CPUUtilization', 'cpu'),
            ('AWS/RDS', 'DatabaseConnections', 'connections'),
            ('AWS/RDS', 'FreeableMemory', 'freeable_memory'),
            ('AWS/RDS', 'ReadIOPS', 'read_iops'),
            ('AWS/RDS', 'WriteIOPS', 'write_iops'),
            ('AWS/RDS', 'ReadLatency', 'read_latency'),
            ('AWS/RDS', 'WriteLatency', 'write_latency'),
            ('AWS/RDS', 'FreeStorageSpace', 'free_storage')
        ]
        
        for namespace, metric_name, key in metric_queries:
            try:
                response = cloudwatch.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,  # 1 hour intervals
                    Statistics=['Average', 'Maximum']
                )
                
                if response['Datapoints']:
                    averages = [dp['Average'] for dp in response['Datapoints']]
                    maximums = [dp['Maximum'] for dp in response['Datapoints']]
                    
                    metrics[f'{key}_avg'] = statistics.mean(averages) if averages else 0
                    metrics[f'{key}_max'] = max(maximums) if maximums else 0
                    metrics[f'{key}_datapoints'] = len(averages)
                else:
                    metrics[f'{key}_avg'] = 0
                    metrics[f'{key}_max'] = 0
                    metrics[f'{key}_datapoints'] = 0
                    
            except Exception as e:
                logger.warning(f"Could not get {metric_name} for {instance_id}: {e}")
                metrics[f'{key}_avg'] = 0
                metrics[f'{key}_max'] = 0
                metrics[f'{key}_datapoints'] = 0
        
        # Calculate derived metrics
        total_iops_avg = metrics.get('read_iops_avg', 0) + metrics.get('write_iops_avg', 0)
        metrics['total_iops_avg'] = total_iops_avg
        
        return metrics
    
    def _generate_database_recommendations(self, instance: Dict, metrics: Dict[str, float],
                                         current_cost: float, region: str, tags: Dict[str, str]) -> List[RDSRecommendation]:
        """Generate optimization recommendations for a database instance"""
        recommendations = []
        
        instance_id = instance['DBInstanceIdentifier']
        instance_class = instance['DBInstanceClass']
        engine = instance['Engine']
        multi_az = instance.get('MultiAZ', False)
        allocated_storage = instance.get('AllocatedStorage', 0)
        storage_type = instance.get('StorageType', 'gp2')
        backup_retention = instance.get('BackupRetentionPeriod', 7)
        
        cpu_avg = metrics.get('cpu_avg', 0)
        cpu_max = metrics.get('cpu_max', 0)
        connections_avg = metrics.get('connections_avg', 0)
        memory_avg = metrics.get('freeable_memory_avg', 0)
        iops_avg = metrics.get('total_iops_avg', 0)
        
        # 1. Check for rightsizing opportunities
        downsize_rec = self._check_instance_rightsizing(
            instance, metrics, current_cost, region, tags
        )
        if downsize_rec:
            recommendations.append(downsize_rec)
        
        # 2. Check for Reserved Instance opportunities
        ri_rec = self._check_reserved_instance_opportunity(
            instance, metrics, current_cost, region, tags
        )
        if ri_rec:
            recommendations.append(ri_rec)
        
        # 3. Check for Aurora migration opportunities
        aurora_rec = self._check_aurora_migration(
            instance, metrics, current_cost, region, tags
        )
        if aurora_rec:
            recommendations.append(aurora_rec)
        
        # 4. Check for storage optimization
        storage_rec = self._check_storage_optimization(
            instance, metrics, current_cost, region, tags
        )
        if storage_rec:
            recommendations.append(storage_rec)
        
        # 5. Check for scheduling opportunities (dev/test environments)
        schedule_rec = self._check_scheduling_opportunity(
            instance, metrics, current_cost, region, tags
        )
        if schedule_rec:
            recommendations.append(schedule_rec)
        
        # 6. Check for backup optimization
        backup_rec = self._check_backup_optimization(
            instance, metrics, current_cost, region, tags
        )
        if backup_rec:
            recommendations.append(backup_rec)
        
        # 7. Check for Graviton migration (ARM instances)
        graviton_rec = self._check_graviton_migration(
            instance, metrics, current_cost, region, tags
        )
        if graviton_rec:
            recommendations.append(graviton_rec)
        
        return recommendations
    
    def _check_instance_rightsizing(self, instance: Dict, metrics: Dict[str, float],
                                  current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for instance rightsizing opportunities"""
        
        instance_class = instance['DBInstanceClass']
        cpu_avg = metrics.get('cpu_avg', 0)
        cpu_max = metrics.get('cpu_max', 0)
        connections_avg = metrics.get('connections_avg', 0)
        
        # Check if instance is underutilized
        if cpu_avg < self.cpu_threshold and cpu_max < 50:
            smaller_class, savings = self._find_smaller_instance_class(instance_class, region)
            
            if smaller_class and savings >= self.min_savings_threshold:
                confidence = min(0.9, (self.cpu_threshold - cpu_avg) / self.cpu_threshold)
                
                # Lower confidence if connections are high (might need current capacity)
                if connections_avg > 100:
                    confidence *= 0.8
                
                if confidence >= self.confidence_threshold:
                    return RDSRecommendation(
                        instance_identifier=instance['DBInstanceIdentifier'],
                        engine=instance['Engine'],
                        instance_class=instance_class,
                        region=region,
                        action=RDSOptimizationAction.DOWNSIZE_INSTANCE,
                        current_monthly_cost=current_cost,
                        estimated_monthly_savings=savings,
                        annual_savings=savings * 12,
                        confidence_score=confidence,
                        reason=f"Low CPU utilization (avg: {cpu_avg:.1f}%) - can downsize",
                        details={
                            'current_class': instance_class,
                            'recommended_class': smaller_class,
                            'cpu_avg': cpu_avg,
                            'cpu_max': cpu_max,
                            'connections_avg': connections_avg
                        },
                        risk_level=self._assess_risk_level(instance, tags, RDSOptimizationAction.DOWNSIZE_INSTANCE),
                        prerequisites=[
                            "Verify application performance requirements",
                            "Test with smaller instance during low-traffic period",
                            "Monitor performance after change"
                        ],
                        implementation_effort='medium',
                        rollback_plan=f"Scale back up to {instance_class}",
                        cpu_avg=cpu_avg,
                        cpu_max=cpu_max,
                        connections_avg=connections_avg,
                        multi_az=instance.get('MultiAZ', False),
                        backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                        storage_type=instance.get('StorageType', 'gp2'),
                        allocated_storage=instance.get('AllocatedStorage', 0),
                        recommended_instance_class=smaller_class
                    )
        
        return None
    
    def _check_reserved_instance_opportunity(self, instance: Dict, metrics: Dict[str, float],
                                           current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for Reserved Instance purchase opportunities"""
        
        # Check if this is a stable production workload
        environment = tags.get('Environment', '').lower()
        cpu_avg = metrics.get('cpu_avg', 0)
        connections_avg = metrics.get('connections_avg', 0)
        
        # Good RI candidates: production workloads with consistent usage
        if (environment in ['prod', 'production'] or 
            cpu_avg > 5 or  # Some consistent usage
            connections_avg > 10):  # Regular database activity
            
            # Estimate RI savings (typically 30-60% for 1-year term)
            ri_savings = current_cost * 0.35  # Conservative 35% savings
            
            if ri_savings >= self.min_savings_threshold:
                return RDSRecommendation(
                    instance_identifier=instance['DBInstanceIdentifier'],
                    engine=instance['Engine'],
                    instance_class=instance['DBInstanceClass'],
                    region=region,
                    action=RDSOptimizationAction.PURCHASE_RESERVED_INSTANCE,
                    current_monthly_cost=current_cost,
                    estimated_monthly_savings=ri_savings,
                    annual_savings=ri_savings * 12,
                    confidence_score=0.8,
                    reason="Stable workload suitable for Reserved Instance purchase",
                    details={
                        'environment': environment,
                        'recommended_term': '1-year',
                        'payment_option': 'partial_upfront',
                        'estimated_savings_percentage': '35%'
                    },
                    risk_level='low',
                    prerequisites=[
                        "Confirm 1+ year commitment",
                        "Verify instance will remain in same AZ",
                        "Check budget approval for upfront payment"
                    ],
                    implementation_effort='low',
                    rollback_plan="Continue with on-demand pricing after RI term",
                    cpu_avg=cpu_avg,
                    cpu_max=metrics.get('cpu_max', 0),
                    connections_avg=connections_avg,
                    multi_az=instance.get('MultiAZ', False),
                    backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                    storage_type=instance.get('StorageType', 'gp2'),
                    allocated_storage=instance.get('AllocatedStorage', 0)
                )
        
        return None
    
    def _check_aurora_migration(self, instance: Dict, metrics: Dict[str, float],
                              current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for Aurora migration opportunities"""
        
        engine = instance['Engine']
        
        # Aurora is available for MySQL and PostgreSQL
        if engine not in ['mysql', 'postgres']:
            return None
        
        # Check if workload benefits from Aurora features
        connections_avg = metrics.get('connections_avg', 0)
        iops_avg = metrics.get('total_iops_avg', 0)
        
        # Aurora benefits: high availability, better performance, automatic scaling
        if (instance.get('MultiAZ', False) or  # Already paying for HA
            connections_avg > 50 or  # High connection workload
            iops_avg > 1000):  # High I/O workload
            
            # Estimate Aurora costs (can be 10-20% more expensive but with better features)
            aurora_cost = current_cost * 1.1  # 10% increase
            potential_performance_savings = current_cost * 0.15  # 15% from better performance
            
            net_savings = potential_performance_savings - (aurora_cost - current_cost)
            
            if net_savings >= self.min_savings_threshold:
                return RDSRecommendation(
                    instance_identifier=instance['DBInstanceIdentifier'],
                    engine=engine,
                    instance_class=instance['DBInstanceClass'],
                    region=region,
                    action=RDSOptimizationAction.MIGRATE_TO_AURORA,
                    current_monthly_cost=current_cost,
                    estimated_monthly_savings=net_savings,
                    annual_savings=net_savings * 12,
                    confidence_score=0.6,  # Lower confidence for migration
                    reason="Workload would benefit from Aurora's performance and availability features",
                    details={
                        'current_engine': engine,
                        'target_aurora_engine': f'aurora-{engine}',
                        'benefits': ['Better performance', 'Automatic failover', 'Read replicas', 'Serverless option'],
                        'estimated_performance_improvement': '15%'
                    },
                    risk_level='medium',
                    prerequisites=[
                        "Test application compatibility with Aurora",
                        "Plan migration strategy",
                        "Verify all features are supported",
                        "Schedule maintenance window"
                    ],
                    implementation_effort='high',
                    rollback_plan="Migrate back to standard RDS (requires downtime)",
                    cpu_avg=metrics.get('cpu_avg', 0),
                    cpu_max=metrics.get('cpu_max', 0),
                    connections_avg=connections_avg,
                    multi_az=instance.get('MultiAZ', False),
                    backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                    storage_type=instance.get('StorageType', 'gp2'),
                    allocated_storage=instance.get('AllocatedStorage', 0)
                )
        
        return None
    
    def _check_storage_optimization(self, instance: Dict, metrics: Dict[str, float],
                                  current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for storage optimization opportunities"""
        
        storage_type = instance.get('StorageType', 'gp2')
        allocated_storage = instance.get('AllocatedStorage', 0)
        iops_avg = metrics.get('total_iops_avg', 0)
        
        # Check if using expensive io1/io2 with low IOPS usage
        if storage_type in ['io1', 'io2'] and iops_avg < 1000:
            # Recommend switching to gp3
            storage_savings = current_cost * 0.2  # Estimate 20% storage savings
            
            if storage_savings >= self.min_savings_threshold:
                return RDSRecommendation(
                    instance_identifier=instance['DBInstanceIdentifier'],
                    engine=instance['Engine'],
                    instance_class=instance['DBInstanceClass'],
                    region=region,
                    action=RDSOptimizationAction.OPTIMIZE_STORAGE,
                    current_monthly_cost=current_cost,
                    estimated_monthly_savings=storage_savings,
                    annual_savings=storage_savings * 12,
                    confidence_score=0.8,
                    reason=f"Using {storage_type} storage with low IOPS utilization",
                    details={
                        'current_storage_type': storage_type,
                        'recommended_storage_type': 'gp3',
                        'current_iops_usage': iops_avg,
                        'allocated_storage_gb': allocated_storage
                    },
                    risk_level='low',
                    prerequisites=[
                        "Verify gp3 performance meets requirements",
                        "Test during maintenance window"
                    ],
                    implementation_effort='low',
                    rollback_plan=f"Switch back to {storage_type}",
                    cpu_avg=metrics.get('cpu_avg', 0),
                    cpu_max=metrics.get('cpu_max', 0),
                    connections_avg=metrics.get('connections_avg', 0),
                    iops_avg=iops_avg,
                    multi_az=instance.get('MultiAZ', False),
                    backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                    storage_type=storage_type,
                    allocated_storage=allocated_storage,
                    recommended_storage_type='gp3'
                )
        
        return None
    
    def _check_scheduling_opportunity(self, instance: Dict, metrics: Dict[str, float],
                                    current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for database scheduling opportunities"""
        
        environment = tags.get('Environment', '').lower()
        workload_type = tags.get('WorkloadType', '').lower()
        
        # Development and testing databases can often be scheduled
        if environment in ['dev', 'development', 'test', 'testing', 'staging']:
            
            # Estimate downtime savings (stop during nights/weekends)
            # Assume 60% uptime for dev/test (12 hours weekdays + weekends off)
            uptime_percentage = 0.6
            potential_savings = current_cost * (1 - uptime_percentage)
            
            if potential_savings >= self.min_savings_threshold:
                return RDSRecommendation(
                    instance_identifier=instance['DBInstanceIdentifier'],
                    engine=instance['Engine'],
                    instance_class=instance['DBInstanceClass'],
                    region=region,
                    action=RDSOptimizationAction.SCHEDULE_INSTANCE,
                    current_monthly_cost=current_cost,
                    estimated_monthly_savings=potential_savings,
                    annual_savings=potential_savings * 12,
                    confidence_score=0.7,
                    reason=f"Development/test database suitable for scheduling",
                    details={
                        'environment': environment,
                        'suggested_schedule': {
                            'weekdays': '8:00 AM - 6:00 PM',
                            'weekends': 'Stopped',
                            'estimated_uptime': '60%'
                        }
                    },
                    risk_level='low',
                    prerequisites=[
                        "Confirm with development team",
                        "Verify no automated processes require 24/7 access",
                        "Set up automated start/stop schedules"
                    ],
                    implementation_effort='medium',
                    rollback_plan="Remove scheduling and run 24/7",
                    cpu_avg=metrics.get('cpu_avg', 0),
                    cpu_max=metrics.get('cpu_max', 0),
                    connections_avg=metrics.get('connections_avg', 0),
                    multi_az=instance.get('MultiAZ', False),
                    backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                    storage_type=instance.get('StorageType', 'gp2'),
                    allocated_storage=instance.get('AllocatedStorage', 0)
                )
        
        return None
    
    def _check_backup_optimization(self, instance: Dict, metrics: Dict[str, float],
                                 current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for backup retention optimization"""
        
        backup_retention = instance.get('BackupRetentionPeriod', 7)
        environment = tags.get('Environment', '').lower()
        
        # Development environments often don't need long backup retention
        if (environment in ['dev', 'development', 'test', 'testing'] and 
            backup_retention > 7):
            
            # Estimate backup storage savings
            backup_savings = current_cost * 0.1 * (backup_retention - 7) / 7  # Rough estimate
            
            if backup_savings >= self.min_savings_threshold:
                return RDSRecommendation(
                    instance_identifier=instance['DBInstanceIdentifier'],
                    engine=instance['Engine'],
                    instance_class=instance['DBInstanceClass'],
                    region=region,
                    action=RDSOptimizationAction.OPTIMIZE_BACKUP_RETENTION,
                    current_monthly_cost=current_cost,
                    estimated_monthly_savings=backup_savings,
                    annual_savings=backup_savings * 12,
                    confidence_score=0.6,
                    reason=f"Development environment with long backup retention ({backup_retention} days)",
                    details={
                        'current_retention_days': backup_retention,
                        'recommended_retention_days': 7,
                        'environment': environment
                    },
                    risk_level='medium',
                    prerequisites=[
                        "Verify backup requirements with team",
                        "Check compliance requirements",
                        "Confirm 7 days is sufficient"
                    ],
                    implementation_effort='low',
                    rollback_plan=f"Restore {backup_retention}-day retention",
                    cpu_avg=metrics.get('cpu_avg', 0),
                    cpu_max=metrics.get('cpu_max', 0),
                    connections_avg=metrics.get('connections_avg', 0),
                    multi_az=instance.get('MultiAZ', False),
                    backup_retention_period=backup_retention,
                    storage_type=instance.get('StorageType', 'gp2'),
                    allocated_storage=instance.get('AllocatedStorage', 0)
                )
        
        return None
    
    def _check_graviton_migration(self, instance: Dict, metrics: Dict[str, float],
                                current_cost: float, region: str, tags: Dict[str, str]) -> Optional[RDSRecommendation]:
        """Check for Graviton (ARM) instance migration opportunities"""
        
        instance_class = instance['DBInstanceClass']
        engine = instance['Engine']
        
        # Graviton instances are available for certain engines
        if engine not in ['mysql', 'postgres']:
            return None
        
        # Map Intel instances to Graviton equivalents
        graviton_mapping = {
            'db.m5.large': 'db.m6g.large',
            'db.m5.xlarge': 'db.m6g.xlarge',
            'db.m5.2xlarge': 'db.m6g.2xlarge',
            'db.m5.4xlarge': 'db.m6g.4xlarge',
            'db.r5.large': 'db.r6g.large',
            'db.r5.xlarge': 'db.r6g.xlarge',
            'db.r5.2xlarge': 'db.r6g.2xlarge',
            'db.r5.4xlarge': 'db.r6g.4xlarge'
        }
        
        graviton_class = graviton_mapping.get(instance_class)
        if not graviton_class:
            return None
        
        # Graviton typically provides 20% better price/performance
        graviton_savings = current_cost * 0.2
        
        if graviton_savings >= self.min_savings_threshold:
            return RDSRecommendation(
                instance_identifier=instance['DBInstanceIdentifier'],
                engine=engine,
                instance_class=instance_class,
                region=region,
                action=RDSOptimizationAction.CHANGE_INSTANCE_TYPE,
                current_monthly_cost=current_cost,
                estimated_monthly_savings=graviton_savings,
                annual_savings=graviton_savings * 12,
                confidence_score=0.7,
                reason="Can migrate to Graviton for better price/performance",
                details={
                    'current_class': instance_class,
                    'recommended_class': graviton_class,
                    'architecture': 'ARM64 (Graviton)',
                    'expected_performance_improvement': '20%'
                },
                risk_level='medium',
                prerequisites=[
                    "Test application compatibility with ARM64",
                    "Verify all drivers support ARM architecture",
                    "Plan migration during maintenance window"
                ],
                implementation_effort='medium',
                rollback_plan=f"Migrate back to {instance_class}",
                cpu_avg=metrics.get('cpu_avg', 0),
                cpu_max=metrics.get('cpu_max', 0),
                connections_avg=metrics.get('connections_avg', 0),
                multi_az=instance.get('MultiAZ', False),
                backup_retention_period=instance.get('BackupRetentionPeriod', 7),
                storage_type=instance.get('StorageType', 'gp2'),
                allocated_storage=instance.get('AllocatedStorage', 0),
                recommended_instance_class=graviton_class
            )
        
        return None
    
    def _analyze_snapshots(self, rds_client, region: str) -> List[RDSRecommendation]:
        """Analyze RDS snapshots for cleanup opportunities"""
        recommendations = []
        
        try:
            # Get manual snapshots
            paginator = rds_client.get_paginator('describe_db_snapshots')
            snapshots = []
            
            for page in paginator.paginate(SnapshotType='manual'):
                snapshots.extend(page['DBSnapshots'])
            
            # Find old snapshots
            cutoff_date = datetime.utcnow() - timedelta(days=90)  # Older than 90 days
            old_snapshots = []
            
            for snapshot in snapshots:
                if snapshot['SnapshotCreateTime'].replace(tzinfo=None) < cutoff_date:
                    old_snapshots.append(snapshot)
            
            if old_snapshots:
                # Estimate storage savings from snapshot cleanup
                estimated_savings = len(old_snapshots) * 5  # Estimate $5 per old snapshot per month
                
                if estimated_savings >= self.min_savings_threshold:
                    recommendations.append(RDSRecommendation(
                        instance_identifier='multiple-snapshots',
                        engine='various',
                        instance_class='N/A',
                        region=region,
                        action=RDSOptimizationAction.DELETE_UNUSED_SNAPSHOTS,
                        current_monthly_cost=estimated_savings,
                        estimated_monthly_savings=estimated_savings * 0.9,
                        annual_savings=estimated_savings * 0.9 * 12,
                        confidence_score=0.8,
                        reason=f"Found {len(old_snapshots)} snapshots older than 90 days",
                        details={
                            'old_snapshot_count': len(old_snapshots),
                            'oldest_snapshot_date': min(s['SnapshotCreateTime'] for s in old_snapshots).strftime('%Y-%m-%d'),
                            'snapshots_to_review': [s['DBSnapshotIdentifier'] for s in old_snapshots[:10]]  # Show first 10
                        },
                        risk_level='medium',
                        prerequisites=[
                            "Review snapshot retention requirements",
                            "Verify snapshots are not needed for compliance",
                            "Identify snapshots safe to delete"
                        ],
                        implementation_effort='low',
                        rollback_plan="Snapshots cannot be restored once deleted",
                        cpu_avg=0,
                        cpu_max=0,
                        multi_az=False,
                        backup_retention_period=0,
                        storage_type='snapshot',
                        allocated_storage=0
                    ))
        
        except Exception as e:
            logger.error(f"Error analyzing snapshots in {region}: {e}")
        
        return recommendations
    
    def _find_smaller_instance_class(self, current_class: str, region: str) -> Tuple[Optional[str], float]:
        """Find a smaller instance class in the same family"""
        # Extract family (e.g., 'db.m5' from 'db.m5.large')
        class_parts = current_class.split('.')
        if len(class_parts) < 3:
            return None, 0
        
        family = f"{class_parts[0]}.{class_parts[1]}"
        
        if family not in self.instance_families:
            return None, 0
        
        family_classes = self.instance_families[family]
        current_index = family_classes.index(current_class) if current_class in family_classes else -1
        
        if current_index <= 0:  # Already smallest or not found
            return None, 0
        
        # Suggest one size smaller
        smaller_class = family_classes[current_index - 1]
        
        # Calculate savings
        current_price = self._get_instance_price(current_class, region)
        smaller_price = self._get_instance_price(smaller_class, region)
        
        monthly_savings = (current_price - smaller_price) * 24 * 30
        
        return smaller_class, monthly_savings
    
    def _assess_risk_level(self, instance: Dict, tags: Dict[str, str], action: RDSOptimizationAction) -> str:
        """Assess the risk level of the recommended action"""
        environment = tags.get('Environment', '').lower()
        criticality = tags.get('Criticality', '').lower()
        
        # High risk conditions
        if environment in ['prod', 'production'] and action in [
            RDSOptimizationAction.DOWNSIZE_INSTANCE,
            RDSOptimizationAction.MIGRATE_TO_AURORA
        ]:
            return 'high'
        
        if criticality in ['critical', 'high'] and action != RDSOptimizationAction.PURCHASE_RESERVED_INSTANCE:
            return 'high'
        
        # Medium risk conditions
        if action in [
            RDSOptimizationAction.CHANGE_INSTANCE_TYPE,
            RDSOptimizationAction.OPTIMIZE_STORAGE,
            RDSOptimizationAction.OPTIMIZE_BACKUP_RETENTION
        ]:
            return 'medium'
        
        return 'low'
    
    def _calculate_monthly_cost(self, instance: Dict, region: str) -> float:
        """Calculate monthly cost for a database instance"""
        instance_class = instance['DBInstanceClass']
        multi_az = instance.get('MultiAZ', False)
        allocated_storage = instance.get('AllocatedStorage', 0)
        storage_type = instance.get('StorageType', 'gp2')
        
        # Instance cost
        hourly_rate = self._get_instance_price(instance_class, region)
        if multi_az:
            hourly_rate *= 2  # Multi-AZ doubles the cost
        
        monthly_instance_cost = hourly_rate * 24 * 30
        
        # Storage cost (simplified)
        storage_cost_per_gb = {
            'gp2': 0.10,
            'gp3': 0.08,
            'io1': 0.125,
            'io2': 0.125,
            'magnetic': 0.10
        }
        
        monthly_storage_cost = allocated_storage * storage_cost_per_gb.get(storage_type, 0.10)
        
        return monthly_instance_cost + monthly_storage_cost
    
    def _get_instance_price(self, instance_class: str, region: str) -> float:
        """Get hourly price for an instance class"""
        if region not in self.pricing_data:
            self._load_pricing_data(region)
        
        return self.pricing_data.get(region, {}).get(instance_class, 0.20)  # Default price
    
    def _load_pricing_data(self, region: str):
        """Load pricing data for a region"""
        # Simplified pricing data - in production, use AWS Pricing API
        base_prices = {
            'db.t3.micro': 0.017,
            'db.t3.small': 0.034,
            'db.t3.medium': 0.068,
            'db.t3.large': 0.136,
            'db.t3.xlarge': 0.272,
            'db.t3.2xlarge': 0.544,
            'db.m5.large': 0.192,
            'db.m5.xlarge': 0.384,
            'db.m5.2xlarge': 0.768,
            'db.m5.4xlarge': 1.536,
            'db.r5.large': 0.240,
            'db.r5.xlarge': 0.480,
            'db.r5.2xlarge': 0.960,
        }
        
        # Regional pricing multipliers
        multipliers = {
            'us-east-1': 1.0,
            'us-west-2': 1.0,
            'eu-west-1': 1.1,
            'ap-southeast-1': 1.15,
        }
        
        multiplier = multipliers.get(region, 1.1)
        self.pricing_data[region] = {
            instance_class: price * multiplier 
            for instance_class, price in base_prices.items()
        }
    
    def export_recommendations(self, recommendations: List[RDSRecommendation], output_file: str):
        """Export recommendations to Excel file"""
        if not recommendations:
            logger.warning("No recommendations to export")
            return
        
        # Convert to DataFrame
        data = []
        for rec in recommendations:
            data.append({
                'Instance ID': rec.instance_identifier,
                'Engine': rec.engine,
                'Instance Class': rec.instance_class,
                'Region': rec.region,
                'Action': rec.action.value,
                'Current Monthly Cost': f"${rec.current_monthly_cost:.2f}",
                'Monthly Savings': f"${rec.estimated_monthly_savings:.2f}",
                'Annual Savings': f"${rec.annual_savings:.2f}",
                'Confidence': f"{rec.confidence_score:.1%}",
                'Risk Level': rec.risk_level,
                'Implementation Effort': rec.implementation_effort,
                'Reason': rec.reason,
                'CPU Avg %': f"{rec.cpu_avg:.1f}%",
                'CPU Max %': f"{rec.cpu_max:.1f}%",
                'Connections Avg': rec.connections_avg or 0,
                'Multi-AZ': rec.multi_az,
                'Storage Type': rec.storage_type,
                'Allocated Storage GB': rec.allocated_storage,
                'Backup Retention Days': rec.backup_retention_period,
                'Recommended Class': rec.recommended_instance_class or 'N/A',
                'Prerequisites': '; '.join(rec.prerequisites),
                'Rollback Plan': rec.rollback_plan
            })
        
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Recommendations',
                    'Total Monthly Savings',
                    'Total Annual Savings',
                    'High-Confidence Recommendations',
                    'Low-Risk Recommendations'
                ],
                'Value': [
                    len(recommendations),
                    f"${sum(r.estimated_monthly_savings for r in recommendations):,.2f}",
                    f"${sum(r.annual_savings for r in recommendations):,.2f}",
                    len([r for r in recommendations if r.confidence_score > 0.8]),
                    len([r for r in recommendations if r.risk_level == 'low'])
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Group by action type
            action_summary = df.groupby('Action').agg({
                'Monthly Savings': lambda x: len(x),
                'Annual Savings': lambda x: x.str.replace('$', '').str.replace(',', '').astype(float).sum()
            }).rename(columns={'Monthly Savings': 'Count', 'Annual Savings': 'Total Annual Savings'})
            
            action_summary['Total Annual Savings'] = action_summary['Total Annual Savings'].apply(lambda x: f"${x:,.2f}")
            action_summary.to_excel(writer, sheet_name='By Action Type')
        
        logger.info(f"Exported {len(recommendations)} RDS recommendations to {output_file}")
    
    def generate_cli_commands(self, recommendations: List[RDSRecommendation], output_file: str):
        """Generate CLI commands for implementing recommendations"""
        if not recommendations:
            return
        
        commands = []
        commands.append("#!/bin/bash")
        commands.append("# AWS RDS Optimization Commands")
        commands.append(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        commands.append("")
        
        for rec in recommendations:
            commands.append(f"# {rec.reason}")
            commands.append(f"# Savings: ${rec.estimated_monthly_savings:.2f}/month")
            commands.append(f"# Risk: {rec.risk_level}, Confidence: {rec.confidence_score:.1%}")
            
            if rec.action == RDSOptimizationAction.DOWNSIZE_INSTANCE:
                commands.append(f"aws rds modify-db-instance --region {rec.region} --db-instance-identifier {rec.instance_identifier} --db-instance-class {rec.recommended_instance_class} --apply-immediately")
            
            elif rec.action == RDSOptimizationAction.OPTIMIZE_STORAGE:
                commands.append(f"aws rds modify-db-instance --region {rec.region} --db-instance-identifier {rec.instance_identifier} --storage-type {rec.recommended_storage_type}")
            
            elif rec.action == RDSOptimizationAction.SCHEDULE_INSTANCE:
                commands.append(f"# Set up scheduling for {rec.instance_identifier}")
                commands.append(f"# Create Lambda function or use AWS Systems Manager for automated start/stop")
            
            elif rec.action == RDSOptimizationAction.DELETE_UNUSED_SNAPSHOTS:
                commands.append(f"# Review and delete old snapshots")
                commands.append(f"aws rds describe-db-snapshots --region {rec.region} --snapshot-type manual")
                commands.append(f"# aws rds delete-db-snapshot --region {rec.region} --db-snapshot-identifier SNAPSHOT_ID")
            
            commands.append("")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(commands))
        
        logger.info(f"Generated RDS CLI commands in {output_file}")
