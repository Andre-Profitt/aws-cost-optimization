"""
Enhanced EC2 Optimizer module with comprehensive optimization strategies
Includes: Instance rightsizing, EBS optimization, scheduling, and more
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..analysis.pattern_detector import PatternDetector
from .safety_checks import SafetyChecker

logger = logging.getLogger(__name__)

@dataclass
class EC2OptimizationRecommendation:
    """Represents an EC2 optimization recommendation"""
    instance_id: str
    instance_type: str
    region: str
    action: str  # 'stop', 'terminate', 'rightsize', 'schedule', 'migrate_to_spot'
    current_monthly_cost: float
    recommended_monthly_cost: float
    monthly_savings: float
    annual_savings: float
    confidence: float
    reason: str
    risk_level: str
    implementation_steps: List[str]
    rollback_plan: str
    tags: Dict[str, str]

@dataclass
class EBSOptimizationRecommendation:
    """Represents an EBS volume optimization recommendation"""
    volume_id: str
    volume_type: str
    size_gb: int
    action: str  # 'delete', 'snapshot_and_delete', 'change_type', 'reduce_size'
    monthly_savings: float
    reason: str
    risk_level: str

class EC2Optimizer:
    """Comprehensive EC2 optimization engine"""
    
    # Instance pricing per hour (simplified - should use Pricing API)
    INSTANCE_PRICING = {
        't2.micro': 0.0116,
        't2.small': 0.023,
        't2.medium': 0.0464,
        't2.large': 0.0928,
        't3.micro': 0.0104,
        't3.small': 0.0208,
        't3.medium': 0.0416,
        't3.large': 0.0832,
        'm5.large': 0.096,
        'm5.xlarge': 0.192,
        'm5.2xlarge': 0.384,
        'c5.large': 0.085,
        'c5.xlarge': 0.17,
        'r5.large': 0.126,
        'r5.xlarge': 0.252
    }
    
    def __init__(self,
                 cpu_threshold: float = 10.0,
                 memory_threshold: float = 20.0,
                 network_threshold: float = 5.0,
                 observation_days: int = 14,
                 session: Optional[boto3.Session] = None):
        """
        Initialize EC2 Optimizer
        
        Args:
            cpu_threshold: CPU % threshold for idle detection
            memory_threshold: Memory % threshold for rightsizing
            network_threshold: Network MB threshold for idle detection
            observation_days: Days of metrics to analyze
            session: Boto3 session
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.network_threshold = network_threshold
        self.observation_days = observation_days
        self.session = session or boto3.Session()
        self.ec2 = self.session.client('ec2')
        self.cloudwatch = self.session.client('cloudwatch')
        self.ce = self.session.client('ce')
        self.pattern_detector = PatternDetector(lookback_days=observation_days)
        self.safety_checker = SafetyChecker(self.session)
        
    def analyze_all_instances(self, regions: List[str] = None) -> List[EC2OptimizationRecommendation]:
        """
        Analyze all EC2 instances across regions
        
        Args:
            regions: List of regions to analyze (None = all regions)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not regions:
            regions = self._get_all_regions()
            
        logger.info(f"Analyzing EC2 instances across {len(regions)} regions")
        
        # Analyze each region in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_region = {
                executor.submit(self._analyze_region_instances, region): region 
                for region in regions
            }
            
            for future in as_completed(future_to_region):
                region = future_to_region[future]
                try:
                    region_recommendations = future.result()
                    recommendations.extend(region_recommendations)
                    logger.info(f"Found {len(region_recommendations)} recommendations in {region}")
                except Exception as e:
                    logger.error(f"Failed to analyze {region}: {e}")
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.monthly_savings, reverse=True)
        
        return recommendations
    
    def _get_all_regions(self) -> List[str]:
        """Get all enabled regions"""
        try:
            response = self.ec2.describe_regions()
            return [r['RegionName'] for r in response['Regions']]
        except Exception as e:
            logger.error(f"Failed to get regions: {e}")
            return ['us-east-1', 'us-west-2']  # Fallback
    
    def _analyze_region_instances(self, region: str) -> List[EC2OptimizationRecommendation]:
        """Analyze instances in a specific region"""
        recommendations = []
        
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            # Get all instances
            paginator = ec2_regional.get_paginator('describe_instances')
            
            for page in paginator.paginate():
                for reservation in page['Reservations']:
                    for instance in reservation['Instances']:
                        if instance['State']['Name'] == 'running':
                            # Analyze individual instance
                            instance_recs = self._analyze_instance(instance, region)
                            recommendations.extend(instance_recs)
            
            # Analyze EBS volumes
            ebs_recommendations = self._analyze_ebs_volumes(region)
            # Convert EBS recommendations to EC2 format for consistency
            for ebs_rec in ebs_recommendations:
                recommendations.append(EC2OptimizationRecommendation(
                    instance_id=ebs_rec.volume_id,
                    instance_type='EBS Volume',
                    region=region,
                    action=ebs_rec.action,
                    current_monthly_cost=ebs_rec.size_gb * 0.10,  # $0.10/GB/month
                    recommended_monthly_cost=0,
                    monthly_savings=ebs_rec.monthly_savings,
                    annual_savings=ebs_rec.monthly_savings * 12,
                    confidence=0.9,
                    reason=ebs_rec.reason,
                    risk_level=ebs_rec.risk_level,
                    implementation_steps=[f"Action: {ebs_rec.action}"],
                    rollback_plan="Restore from snapshot if needed",
                    tags={}
                ))
                
        except Exception as e:
            logger.error(f"Failed to analyze instances in {region}: {e}")
            
        return recommendations
    
    def _analyze_instance(self, instance: Dict[str, Any], region: str) -> List[EC2OptimizationRecommendation]:
        """Analyze a single instance for optimization opportunities"""
        recommendations = []
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        
        # Get instance tags
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        # Skip if tagged to not optimize
        if tags.get('DoNotOptimize', '').lower() in ['true', 'yes', '1']:
            return recommendations
        
        # Perform safety checks
        safety_result = self.safety_checker.check_instance_safety(instance_id)
        if not safety_result['safe_to_modify']:
            logger.info(f"Skipping {instance_id} due to safety concerns: {safety_result['blockers']}")
            return recommendations
        
        # Get CloudWatch metrics
        metrics = self._get_instance_metrics(instance_id, region)
        
        # Get pattern analysis
        patterns = self.pattern_detector.analyze_ec2_patterns(instance_id)
        
        # 1. Check for idle instances
        idle_rec = self._check_idle_instance(instance, metrics, patterns, region)
        if idle_rec:
            recommendations.append(idle_rec)
        
        # 2. Check for rightsizing opportunities
        rightsize_rec = self._check_rightsizing(instance, metrics, tags, region)
        if rightsize_rec:
            recommendations.append(rightsize_rec)
        
        # 3. Check for scheduling opportunities
        schedule_rec = self._check_scheduling_opportunity(instance, patterns, tags, region)
        if schedule_rec:
            recommendations.append(schedule_rec)
        
        # 4. Check for Spot migration opportunities
        spot_rec = self._check_spot_migration(instance, tags, region)
        if spot_rec:
            recommendations.append(spot_rec)
        
        # 5. Check for Savings Plans opportunities
        sp_rec = self._check_savings_plan_candidate(instance, metrics, region)
        if sp_rec:
            recommendations.append(sp_rec)
        
        return recommendations
    
    def _get_instance_metrics(self, instance_id: str, region: str) -> Dict[str, Any]:
        """Get CloudWatch metrics for an instance"""
        metrics = {}
        
        try:
            cw_regional = self.session.client('cloudwatch', region_name=region)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.observation_days)
            
            # CPU Utilization
            cpu_response = cw_regional.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            
            if cpu_response['Datapoints']:
                cpu_data = pd.DataFrame(cpu_response['Datapoints'])
                metrics['cpu_avg'] = cpu_data['Average'].mean()
                metrics['cpu_max'] = cpu_data['Maximum'].max()
                metrics['cpu_p95'] = cpu_data['Average'].quantile(0.95) if len(cpu_data) > 10 else metrics['cpu_max']
            else:
                metrics['cpu_avg'] = 0
                metrics['cpu_max'] = 0
                metrics['cpu_p95'] = 0
            
            # Network In/Out
            for metric_name, key in [('NetworkIn', 'network_in'), ('NetworkOut', 'network_out')]:
                response = cw_regional.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # Daily
                    Statistics=['Sum']
                )
                
                if response['Datapoints']:
                    total_bytes = sum(dp['Sum'] for dp in response['Datapoints'])
                    metrics[f'{key}_total_mb'] = total_bytes / (1024**2)
                    metrics[f'{key}_daily_avg_mb'] = metrics[f'{key}_total_mb'] / self.observation_days
                else:
                    metrics[f'{key}_total_mb'] = 0
                    metrics[f'{key}_daily_avg_mb'] = 0
            
            # EBS metrics if available
            # Memory metrics if CloudWatch agent is installed
            
        except Exception as e:
            logger.error(f"Failed to get metrics for {instance_id}: {e}")
            
        return metrics
    
    def _check_idle_instance(self, instance: Dict[str, Any], 
                           metrics: Dict[str, Any], 
                           patterns: Dict[str, Any],
                           region: str) -> Optional[EC2OptimizationRecommendation]:
        """Check if instance is idle and can be stopped"""
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        # Check if CPU and network are below thresholds
        is_idle = (
            metrics.get('cpu_avg', 100) < self.cpu_threshold and
            metrics.get('network_in_daily_avg_mb', 100) < self.network_threshold and
            metrics.get('network_out_daily_avg_mb', 100) < self.network_threshold
        )
        
        if is_idle:
            # Check if it's a periodic workload
            if patterns.get('patterns', {}).get('cpu_pattern') == 'periodic':
                return None  # Don't stop periodic workloads
            
            current_cost = self._calculate_instance_cost(instance_type)
            
            # Determine action based on environment
            env = tags.get('Environment', '').lower()
            if env in ['dev', 'development', 'test']:
                action = 'stop'
                savings = current_cost * 0.9  # Save 90% by stopping
                risk = 'low'
            elif env == 'staging':
                action = 'schedule'  # Suggest scheduling instead
                savings = current_cost * 0.65  # Save 65% with scheduling
                risk = 'medium'
            else:
                action = 'rightsize'  # Production - suggest rightsizing only
                return None  # Handle in rightsizing check
            
            return EC2OptimizationRecommendation(
                instance_id=instance_id,
                instance_type=instance_type,
                region=region,
                action=action,
                current_monthly_cost=current_cost,
                recommended_monthly_cost=current_cost - savings,
                monthly_savings=savings,
                annual_savings=savings * 12,
                confidence=0.9,
                reason=f"Instance is idle: CPU {metrics['cpu_avg']:.1f}%, Network {metrics['network_in_daily_avg_mb']:.1f}MB/day",
                risk_level=risk,
                implementation_steps=[
                    f"1. Verify instance is not needed: {instance_id}",
                    f"2. Create snapshot of attached volumes",
                    f"3. Stop instance: aws ec2 stop-instances --instance-ids {instance_id}",
                    "4. Set reminder to check in 7 days",
                    "5. Terminate if not needed after 30 days"
                ],
                rollback_plan=f"aws ec2 start-instances --instance-ids {instance_id}",
                tags=tags
            )
        
        return None
    
    def _check_rightsizing(self, instance: Dict[str, Any],
                          metrics: Dict[str, Any],
                          tags: Dict[str, str],
                          region: str) -> Optional[EC2OptimizationRecommendation]:
        """Check if instance can be rightsized"""
        instance_id = instance['InstanceId']
        current_type = instance['InstanceType']
        
        # Skip if recently launched
        launch_time = instance['LaunchTime'].replace(tzinfo=None)
        if (datetime.utcnow() - launch_time).days < 7:
            return None
        
        # Check CPU utilization for rightsizing
        cpu_avg = metrics.get('cpu_avg', 100)
        cpu_p95 = metrics.get('cpu_p95', 100)
        
        if cpu_avg < 40 and cpu_p95 < 60:
            # Recommend smaller instance
            recommended_type = self._get_recommended_instance_type(
                current_type, cpu_avg, cpu_p95
            )
            
            if recommended_type and recommended_type != current_type:
                current_cost = self._calculate_instance_cost(current_type)
                new_cost = self._calculate_instance_cost(recommended_type)
                savings = current_cost - new_cost
                
                if savings > 20:  # Only recommend if saving > $20/month
                    return EC2OptimizationRecommendation(
                        instance_id=instance_id,
                        instance_type=current_type,
                        region=region,
                        action='rightsize',
                        current_monthly_cost=current_cost,
                        recommended_monthly_cost=new_cost,
                        monthly_savings=savings,
                        annual_savings=savings * 12,
                        confidence=0.8,
                        reason=f"Low CPU utilization: avg {cpu_avg:.1f}%, p95 {cpu_p95:.1f}%",
                        risk_level='medium',
                        implementation_steps=[
                            f"1. Schedule maintenance window",
                            f"2. Stop instance: aws ec2 stop-instances --instance-ids {instance_id}",
                            f"3. Modify type: aws ec2 modify-instance-attribute --instance-id {instance_id} --instance-type {recommended_type}",
                            f"4. Start instance: aws ec2 start-instances --instance-ids {instance_id}",
                            "5. Monitor performance for 24 hours"
                        ],
                        rollback_plan=f"Repeat steps 2-4 with original type: {current_type}",
                        tags=tags
                    )
        
        return None
    
    def _check_scheduling_opportunity(self, instance: Dict[str, Any],
                                    patterns: Dict[str, Any],
                                    tags: Dict[str, str],
                                    region: str) -> Optional[EC2OptimizationRecommendation]:
        """Check if instance can benefit from scheduling"""
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        env = tags.get('Environment', '').lower()
        
        # Only suggest scheduling for non-production
        if env not in ['dev', 'development', 'test', 'staging']:
            return None
        
        # Check if already has a schedule
        if tags.get('Schedule'):
            return None
        
        # Analyze usage patterns
        if patterns.get('patterns', {}).get('cpu_pattern') in ['business_hours', 'periodic']:
            current_cost = self._calculate_instance_cost(instance_type)
            
            # Assume 65% savings with business hours scheduling
            savings = current_cost * 0.65
            
            return EC2OptimizationRecommendation(
                instance_id=instance_id,
                instance_type=instance_type,
                region=region,
                action='schedule',
                current_monthly_cost=current_cost,
                recommended_monthly_cost=current_cost - savings,
                monthly_savings=savings,
                annual_savings=savings * 12,
                confidence=0.85,
                reason=f"Instance shows business hours usage pattern in {env} environment",
                risk_level='low',
                implementation_steps=[
                    "1. Implement AWS Instance Scheduler",
                    "2. Tag instance with Schedule=business-hours",
                    "3. Configure schedule: Mon-Fri 7AM-7PM",
                    "4. Enable CloudWatch monitoring",
                    "5. Set up on-demand start if needed"
                ],
                rollback_plan="Remove Schedule tag to disable scheduling",
                tags=tags
            )
        
        return None
    
    def _check_spot_migration(self, instance: Dict[str, Any],
                            tags: Dict[str, str],
                            region: str) -> Optional[EC2OptimizationRecommendation]:
        """Check if instance can be migrated to Spot"""
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        env = tags.get('Environment', '').lower()
        
        # Only for non-production stateless workloads
        if env not in ['dev', 'test'] or tags.get('Stateful', '').lower() == 'true':
            return None
        
        # Check if it's already spot
        if instance.get('InstanceLifecycle') == 'spot':
            return None
        
        current_cost = self._calculate_instance_cost(instance_type)
        
        # Assume 70% savings with Spot
        spot_discount = 0.70
        savings = current_cost * spot_discount
        
        return EC2OptimizationRecommendation(
            instance_id=instance_id,
            instance_type=instance_type,
            region=region,
            action='migrate_to_spot',
            current_monthly_cost=current_cost,
            recommended_monthly_cost=current_cost - savings,
            monthly_savings=savings,
            annual_savings=savings * 12,
            confidence=0.7,
            reason=f"Stateless {env} workload suitable for Spot instances",
            risk_level='medium',
            implementation_steps=[
                "1. Create AMI from instance",
                "2. Create Spot Fleet or ASG with Spot",
                "3. Configure instance recovery",
                "4. Test application resilience",
                "5. Terminate on-demand instance"
            ],
            rollback_plan="Launch on-demand instance from AMI",
            tags=tags
        )
    
    def _check_savings_plan_candidate(self, instance: Dict[str, Any],
                                    metrics: Dict[str, Any],
                                    region: str) -> Optional[EC2OptimizationRecommendation]:
        """Check if instance is a good Savings Plan candidate"""
        # This is handled by Reserved Instance Analyzer
        # Just return None here
        return None
    
    def _get_recommended_instance_type(self, current_type: str, 
                                     cpu_avg: float, 
                                     cpu_p95: float) -> Optional[str]:
        """Get recommended instance type based on usage"""
        # Parse current instance family and size
        parts = current_type.split('.')
        if len(parts) != 2:
            return None
        
        family = parts[0]
        size = parts[1]
        
        # Define size hierarchy
        sizes = ['nano', 'micro', 'small', 'medium', 'large', 'xlarge', '2xlarge', '4xlarge']
        
        current_idx = sizes.index(size) if size in sizes else -1
        if current_idx == -1:
            return None
        
        # Determine target size based on CPU
        if cpu_avg < 10 and cpu_p95 < 20:
            target_idx = max(0, current_idx - 2)
        elif cpu_avg < 20 and cpu_p95 < 40:
            target_idx = max(0, current_idx - 1)
        else:
            return None  # No downsize needed
        
        # Check if target size exists for this family
        target_size = sizes[target_idx]
        target_type = f"{family}.{target_size}"
        
        # Validate it's a real instance type
        if target_type in self.INSTANCE_PRICING:
            return target_type
        
        # Try t3 family if current family doesn't have small sizes
        if family not in ['t2', 't3'] and target_idx < 3:
            t3_type = f"t3.{target_size}"
            if t3_type in self.INSTANCE_PRICING:
                return t3_type
        
        return None
    
    def _calculate_instance_cost(self, instance_type: str) -> float:
        """Calculate monthly cost for instance type"""
        hourly_price = self.INSTANCE_PRICING.get(instance_type, 0.1)
        return hourly_price * 24 * 30
    
    def _analyze_ebs_volumes(self, region: str) -> List[EBSOptimizationRecommendation]:
        """Analyze EBS volumes for optimization"""
        recommendations = []
        
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            # Get all volumes
            paginator = ec2_regional.get_paginator('describe_volumes')
            
            for page in paginator.paginate():
                for volume in page['Volumes']:
                    # Check unattached volumes
                    if volume['State'] == 'available':
                        recommendations.append(EBSOptimizationRecommendation(
                            volume_id=volume['VolumeId'],
                            volume_type=volume['VolumeType'],
                            size_gb=volume['Size'],
                            action='delete' if self._is_old_volume(volume) else 'snapshot_and_delete',
                            monthly_savings=volume['Size'] * 0.10,  # $0.10/GB/month
                            reason=f"Unattached volume for {self._get_volume_age_days(volume)} days",
                            risk_level='low' if self._has_recent_snapshot(volume, ec2_regional) else 'medium'
                        ))
                    
                    # Check for gp2 to gp3 migration
                    elif volume['VolumeType'] == 'gp2' and volume['Size'] > 100:
                        gp2_cost = volume['Size'] * 0.10
                        gp3_cost = volume['Size'] * 0.08  # 20% cheaper
                        
                        recommendations.append(EBSOptimizationRecommendation(
                            volume_id=volume['VolumeId'],
                            volume_type=volume['VolumeType'],
                            size_gb=volume['Size'],
                            action='change_type',
                            monthly_savings=gp2_cost - gp3_cost,
                            reason="gp3 is 20% cheaper than gp2 with better performance",
                            risk_level='low'
                        ))
                        
        except Exception as e:
            logger.error(f"Failed to analyze EBS volumes in {region}: {e}")
            
        return recommendations
    
    def _is_old_volume(self, volume: Dict[str, Any]) -> bool:
        """Check if volume is old (>30 days)"""
        create_time = volume['CreateTime'].replace(tzinfo=None)
        return (datetime.utcnow() - create_time).days > 30
    
    def _get_volume_age_days(self, volume: Dict[str, Any]) -> int:
        """Get volume age in days"""
        create_time = volume['CreateTime'].replace(tzinfo=None)
        return (datetime.utcnow() - create_time).days
    
    def _has_recent_snapshot(self, volume: Dict[str, Any], ec2_client) -> bool:
        """Check if volume has recent snapshot"""
        try:
            response = ec2_client.describe_snapshots(
                Filters=[
                    {'Name': 'volume-id', 'Values': [volume['VolumeId']]},
                    {'Name': 'status', 'Values': ['completed']}
                ]
            )
            
            if response['Snapshots']:
                latest = max(response['Snapshots'], key=lambda s: s['StartTime'])
                age_days = (datetime.utcnow() - latest['StartTime'].replace(tzinfo=None)).days
                return age_days < 7
                
        except Exception:
            pass
            
        return False
    
    def generate_optimization_report(self, recommendations: List[EC2OptimizationRecommendation]) -> pd.DataFrame:
        """Generate optimization report DataFrame"""
        data = []
        
        for rec in recommendations:
            data.append({
                'Instance ID': rec.instance_id,
                'Type': rec.instance_type,
                'Region': rec.region,
                'Action': rec.action,
                'Monthly Savings': f"${rec.monthly_savings:.2f}",
                'Annual Savings': f"${rec.annual_savings:.2f}",
                'Confidence': f"{rec.confidence:.0%}",
                'Risk': rec.risk_level,
                'Reason': rec.reason
            })
        
        return pd.DataFrame(data)
    
    def export_recommendations(self, recommendations: List[EC2OptimizationRecommendation],
                             output_file: str = 'ec2_optimization_report.xlsx'):
        """Export recommendations to Excel"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary
            summary_df = self.generate_optimization_report(recommendations)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed implementation
            implementation_data = []
            for rec in recommendations:
                implementation_data.append({
                    'Instance': rec.instance_id,
                    'Action': rec.action,
                    'Savings': rec.monthly_savings,
                    'Steps': '\n'.join(rec.implementation_steps),
                    'Rollback': rec.rollback_plan
                })
            
            impl_df = pd.DataFrame(implementation_data)
            impl_df.to_excel(writer, sheet_name='Implementation Guide', index=False)
            
        logger.info(f"Exported EC2 optimization report to {output_file}")