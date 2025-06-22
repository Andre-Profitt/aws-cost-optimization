import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from botocore.exceptions import ClientError

from .models import EC2OptimizationRecommendation, EBSOptimizationRecommendation


logger = logging.getLogger(__name__)


class EC2Optimizer:
    def __init__(
        self,
        cpu_threshold: float = 10.0,
        memory_threshold: float = 20.0,
        network_threshold: float = 5.0,
        observation_days: int = 14
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.network_threshold = network_threshold
        self.observation_days = observation_days
        self.pattern_detector = None
        self.safety_checker = None
        
        # Instance pricing (simplified - in production, fetch from AWS Pricing API)
        self.instance_prices = {
            't3.nano': 0.0052,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            't3.2xlarge': 0.3328,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'm5.4xlarge': 0.768,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34,
        }
        
        # Instance family sizing
        self.instance_sizes = {
            'nano': 0.25,
            'micro': 0.5,
            'small': 1,
            'medium': 2,
            'large': 4,
            'xlarge': 8,
            '2xlarge': 16,
            '4xlarge': 32,
            '8xlarge': 64,
            '12xlarge': 96,
            '16xlarge': 128,
            '24xlarge': 192,
        }
    
    def set_pattern_detector(self, detector):
        self.pattern_detector = detector
    
    def set_safety_checker(self, checker):
        self.safety_checker = checker
    
    def analyze(self, regions: List[str] = None) -> Dict[str, List[EC2OptimizationRecommendation]]:
        if regions is None:
            regions = self._get_all_regions()
        
        all_recommendations = {}
        
        for region in regions:
            logger.info(f"Analyzing EC2 instances in region: {region}")
            recommendations = self._analyze_region(region)
            if recommendations:
                all_recommendations[region] = recommendations
        
        return all_recommendations
    
    def _get_all_regions(self) -> List[str]:
        ec2 = boto3.client('ec2')
        response = ec2.describe_regions()
        return [region['RegionName'] for region in response['Regions']]
    
    def _analyze_region(self, region: str) -> List[EC2OptimizationRecommendation]:
        ec2 = boto3.client('ec2', region_name=region)
        recommendations = []
        
        try:
            # Get all running instances
            response = ec2.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_recommendations = self._analyze_instance(instance, region)
                    recommendations.extend(instance_recommendations)
            
            # Analyze EBS volumes
            ebs_recommendations = self._analyze_ebs_volumes(region)
            
        except ClientError as e:
            logger.error(f"Error analyzing region {region}: {e}")
        
        return recommendations
    
    def _analyze_instance(self, instance: Dict[str, Any], region: str) -> List[EC2OptimizationRecommendation]:
        recommendations = []
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        
        # Extract tags
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        # Safety check
        if self.safety_checker:
            safety_result = self.safety_checker.check_instance_safety(instance, tags)
            if not safety_result.get('safe_to_modify', True):
                logger.info(f"Skipping instance {instance_id} due to safety checks: {safety_result.get('blockers')}")
                return recommendations
        
        # Get instance metrics
        metrics = self._get_instance_metrics(instance_id, region)
        if not metrics:
            return recommendations
        
        # Pattern detection
        patterns = {}
        if self.pattern_detector:
            patterns = self.pattern_detector.analyze_ec2_patterns(instance_id, region)
        
        # Check for idle instance
        if self._is_idle_instance(metrics, patterns):
            rec = self._create_idle_recommendation(instance, metrics, tags, region)
            if rec:
                recommendations.append(rec)
        
        # Check for rightsizing opportunities
        elif self._needs_rightsizing(metrics):
            rec = self._check_rightsizing(instance, metrics, tags, region)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _get_instance_metrics(self, instance_id: str, region: str) -> Dict[str, float]:
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.observation_days)
        
        metrics = {}
        
        # Get CPU utilization
        try:
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                metrics['cpu_avg'] = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])
                metrics['cpu_max'] = max(dp['Maximum'] for dp in response['Datapoints'])
                
                # Calculate 95th percentile
                values = sorted([dp['Average'] for dp in response['Datapoints']])
                p95_index = int(len(values) * 0.95)
                metrics['cpu_p95'] = values[p95_index] if values else 0
            
            # Get network metrics
            for metric_name, metric_key in [
                ('NetworkIn', 'network_in'),
                ('NetworkOut', 'network_out')
            ]:
                response = cloudwatch.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName=metric_name,
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=3600,
                    Statistics=['Sum']
                )
                
                if response['Datapoints']:
                    total_bytes = sum(dp['Sum'] for dp in response['Datapoints'])
                    metrics[f'{metric_key}_total_mb'] = total_bytes / (1024 * 1024)
                    metrics[f'{metric_key}_daily_avg_mb'] = (total_bytes / (1024 * 1024)) / self.observation_days
        
        except ClientError as e:
            logger.error(f"Error getting metrics for instance {instance_id}: {e}")
        
        return metrics
    
    def _is_idle_instance(self, metrics: Dict[str, float], patterns: Dict[str, Any]) -> bool:
        if not metrics:
            return False
        
        # Check if it's a periodic workload
        if patterns.get('patterns', {}).get('cpu_pattern') == 'periodic':
            return False
        
        # Check CPU utilization
        cpu_idle = metrics.get('cpu_avg', 100) < self.cpu_threshold
        
        # Check network activity
        network_idle = (
            metrics.get('network_in_daily_avg_mb', float('inf')) < self.network_threshold and
            metrics.get('network_out_daily_avg_mb', float('inf')) < self.network_threshold
        )
        
        return cpu_idle and network_idle
    
    def _needs_rightsizing(self, metrics: Dict[str, float]) -> bool:
        if not metrics:
            return False
        
        # Check if CPU is consistently low but not idle
        cpu_avg = metrics.get('cpu_avg', 0)
        cpu_p95 = metrics.get('cpu_p95', 0)
        
        return (
            self.cpu_threshold < cpu_avg < 40 and  # Not idle but underutilized
            cpu_p95 < 60  # Peak usage is still low
        )
    
    def _create_idle_recommendation(
        self,
        instance: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Dict[str, str],
        region: str
    ) -> EC2OptimizationRecommendation:
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        
        monthly_cost = self._calculate_instance_cost(instance_type)
        
        return EC2OptimizationRecommendation(
            instance_id=instance_id,
            instance_type=instance_type,
            region=region,
            action='stop',
            reason=f"Instance is idle (CPU avg: {metrics.get('cpu_avg', 0):.1f}%, "
                   f"Network I/O: {metrics.get('network_in_daily_avg_mb', 0):.1f} MB/day)",
            monthly_savings=monthly_cost,
            annual_savings=monthly_cost * 12,
            risk_level='low' if tags.get('Environment', '').lower() in ['dev', 'test'] else 'medium',
            implementation_steps=[
                f"1. Verify instance {instance_id} is not needed",
                "2. Create an AMI backup if needed",
                f"3. Stop instance using: aws ec2 stop-instances --instance-ids {instance_id}",
                "4. Monitor for any issues",
                "5. Terminate after verification period"
            ],
            tags=tags,
            metrics=metrics
        )
    
    def _check_rightsizing(
        self,
        instance: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Dict[str, str],
        region: str
    ) -> Optional[EC2OptimizationRecommendation]:
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        
        recommended_type = self._get_recommended_instance_type(
            instance_type,
            metrics.get('cpu_avg', 0),
            metrics.get('cpu_p95', 0)
        )
        
        if not recommended_type or recommended_type == instance_type:
            return None
        
        current_cost = self._calculate_instance_cost(instance_type)
        new_cost = self._calculate_instance_cost(recommended_type)
        monthly_savings = current_cost - new_cost
        
        return EC2OptimizationRecommendation(
            instance_id=instance_id,
            instance_type=instance_type,
            region=region,
            action='rightsize',
            reason=f"Instance can be downsized (CPU avg: {metrics.get('cpu_avg', 0):.1f}%, "
                   f"P95: {metrics.get('cpu_p95', 0):.1f}%)",
            monthly_savings=monthly_savings,
            annual_savings=monthly_savings * 12,
            risk_level='low' if tags.get('Environment', '').lower() in ['dev', 'test'] else 'medium',
            implementation_steps=[
                f"1. Review application requirements for {instance_id}",
                "2. Create an AMI backup",
                f"3. Stop instance and change type to {recommended_type}",
                "4. Start instance and monitor performance",
                "5. Revert if performance issues occur"
            ],
            tags=tags,
            metrics=metrics
        )
    
    def _get_recommended_instance_type(
        self,
        current_type: str,
        cpu_avg: float,
        cpu_p95: float
    ) -> Optional[str]:
        # Parse instance type
        parts = current_type.split('.')
        if len(parts) != 2:
            return None
        
        family, size = parts
        
        # Get current size value
        current_size_value = self.instance_sizes.get(size, 0)
        if current_size_value == 0:
            return None
        
        # Calculate recommended size based on CPU usage
        if cpu_p95 < 20:
            size_factor = 0.25  # Can go down 4x
        elif cpu_p95 < 40:
            size_factor = 0.5   # Can go down 2x
        elif cpu_p95 < 60:
            size_factor = 0.75  # Can go down slightly
        else:
            return None  # No downsizing recommended
        
        recommended_size_value = current_size_value * size_factor
        
        # Find the closest available size
        available_sizes = sorted(self.instance_sizes.items(), key=lambda x: x[1])
        for size_name, size_value in available_sizes:
            if size_value >= recommended_size_value:
                if size_value < current_size_value:
                    return f"{family}.{size_name}"
                break
        
        return None
    
    def _calculate_instance_cost(self, instance_type: str) -> float:
        hourly_price = self.instance_prices.get(instance_type, 0.1)  # Default to $0.10/hour
        return hourly_price * 24 * 30  # Monthly cost
    
    def _analyze_ebs_volumes(self, region: str) -> List[EBSOptimizationRecommendation]:
        ec2 = boto3.client('ec2', region_name=region)
        recommendations = []
        
        try:
            response = ec2.describe_volumes()
            
            for volume in response['Volumes']:
                if volume['State'] == 'available':  # Unattached volume
                    volume_id = volume['VolumeId']
                    
                    # Check if volume is old enough
                    if self._is_old_volume(volume):
                        # Check for recent snapshots
                        if not self._has_recent_snapshot(volume_id, region):
                            rec = EBSOptimizationRecommendation(
                                volume_id=volume_id,
                                volume_type=volume['VolumeType'],
                                size_gb=volume['Size'],
                                region=region,
                                action='delete',
                                reason="Unattached volume with no recent snapshots",
                                monthly_savings=volume['Size'] * 0.10,  # $0.10 per GB for gp2
                                annual_savings=volume['Size'] * 0.10 * 12,
                                risk_level='low',
                                implementation_steps=[
                                    f"1. Verify volume {volume_id} is not needed",
                                    "2. Create a snapshot if data needs to be preserved",
                                    f"3. Delete volume using: aws ec2 delete-volume --volume-id {volume_id}",
                                ],
                                tags={tag['Key']: tag['Value'] for tag in volume.get('Tags', [])}
                            )
                            recommendations.append(rec)
        
        except ClientError as e:
            logger.error(f"Error analyzing EBS volumes in {region}: {e}")
        
        return recommendations
    
    def _is_old_volume(self, volume: Dict[str, Any]) -> bool:
        create_time = volume['CreateTime']
        if hasattr(create_time, 'tzinfo'):
            create_time = create_time.replace(tzinfo=None)
        age = datetime.utcnow() - create_time
        return age.days > 30
    
    def _has_recent_snapshot(self, volume_id: str, region: str) -> bool:
        ec2 = boto3.client('ec2', region_name=region)
        
        try:
            response = ec2.describe_snapshots(
                Filters=[
                    {'Name': 'volume-id', 'Values': [volume_id]},
                    {'Name': 'status', 'Values': ['completed']}
                ],
                OwnerIds=['self']
            )
            
            if response['Snapshots']:
                latest_snapshot = max(response['Snapshots'], key=lambda x: x['StartTime'])
                snapshot_time = latest_snapshot['StartTime']
                if hasattr(snapshot_time, 'tzinfo'):
                    snapshot_time = snapshot_time.replace(tzinfo=None)
                age = datetime.utcnow() - snapshot_time
                return age.days < 30
        
        except ClientError as e:
            logger.error(f"Error checking snapshots for volume {volume_id}: {e}")
        
        return False