import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class SafetyChecker:
    def __init__(self):
        self.critical_tags = ['DoNotOptimize', 'Production', 'Critical']
        self.safe_environments = ['dev', 'development', 'test', 'testing', 'qa', 'staging']
        self.protected_instances = set()
        
        # Patterns that indicate critical instances
        self.critical_name_patterns = [
            'prod', 'production', 'master', 'main',
            'database', 'db', 'critical', 'core'
        ]
    
    def check_instance_safety(
        self,
        instance: Dict[str, Any],
        tags: Dict[str, str]
    ) -> Dict[str, Any]:
        instance_id = instance['InstanceId']
        warnings = []
        blockers = []
        
        # Check for critical tags
        for tag_key, tag_value in tags.items():
            if tag_key in self.critical_tags:
                if tag_value.lower() in ['true', 'yes', '1']:
                    blockers.append(f"Instance has {tag_key} tag set to {tag_value}")
            
            # Check environment tag
            if tag_key.lower() == 'environment':
                if tag_value.lower() not in self.safe_environments:
                    warnings.append(f"Instance is in {tag_value} environment")
        
        # Check instance name
        instance_name = tags.get('Name', '').lower()
        for pattern in self.critical_name_patterns:
            if pattern in instance_name:
                warnings.append(f"Instance name contains '{pattern}'")
        
        # Check for attached resources
        safety_checks = {
            'has_elastic_ip': self._check_elastic_ip(instance),
            'is_in_asg': self._check_auto_scaling_group(instance_id, instance.get('Placement', {}).get('Region')),
            'has_load_balancer': self._check_load_balancer_attachment(instance_id, instance.get('Placement', {}).get('Region')),
            'is_nat_instance': self._check_nat_instance(instance, tags),
            'has_iam_role': bool(instance.get('IamInstanceProfile')),
            'is_domain_controller': self._check_domain_controller(tags),
            'has_recent_activity': self._check_recent_activity(instance_id, instance.get('Placement', {}).get('Region'))
        }
        
        # Add warnings for certain conditions
        if safety_checks['has_elastic_ip']:
            warnings.append("Instance has Elastic IP attached")
        
        if safety_checks['is_in_asg']:
            blockers.append("Instance is part of an Auto Scaling Group")
        
        if safety_checks['has_load_balancer']:
            warnings.append("Instance is attached to a Load Balancer")
        
        if safety_checks['is_nat_instance']:
            blockers.append("Instance appears to be a NAT instance")
        
        if safety_checks['is_domain_controller']:
            blockers.append("Instance appears to be a domain controller")
        
        # Check instance launch time
        launch_time = instance.get('LaunchTime')
        if launch_time:
            if hasattr(launch_time, 'tzinfo'):
                launch_time = launch_time.replace(tzinfo=None)
            age = datetime.utcnow() - launch_time
            if age.days < 7:
                warnings.append(f"Instance is only {age.days} days old")
        
        # Determine if safe to modify
        safe_to_modify = len(blockers) == 0
        
        return {
            'safe_to_modify': safe_to_modify,
            'warnings': warnings,
            'blockers': blockers,
            'safety_checks': safety_checks,
            'risk_score': self._calculate_risk_score(warnings, blockers, safety_checks)
        }
    
    def _check_elastic_ip(self, instance: Dict[str, Any]) -> bool:
        return bool(instance.get('PublicIpAddress')) and instance.get('PublicIpAddress') != instance.get('PrivateIpAddress')
    
    def _check_auto_scaling_group(self, instance_id: str, region: str) -> bool:
        if not region:
            return False
        
        try:
            autoscaling = boto3.client('autoscaling', region_name=region)
            response = autoscaling.describe_auto_scaling_instances(
                InstanceIds=[instance_id]
            )
            return len(response['AutoScalingInstances']) > 0
        except ClientError:
            return False
    
    def _check_load_balancer_attachment(self, instance_id: str, region: str) -> bool:
        if not region:
            return False
        
        try:
            # Check ELB Classic
            elb = boto3.client('elb', region_name=region)
            response = elb.describe_load_balancers()
            
            for lb in response['LoadBalancerDescriptions']:
                if instance_id in [i['InstanceId'] for i in lb.get('Instances', [])]:
                    return True
            
            # Check ALB/NLB
            elbv2 = boto3.client('elbv2', region_name=region)
            response = elbv2.describe_target_health()
            
            # This would need proper implementation with target group ARNs
            # For now, we'll skip the detailed check
            
        except ClientError:
            pass
        
        return False
    
    def _check_nat_instance(self, instance: Dict[str, Any], tags: Dict[str, str]) -> bool:
        # Check common NAT instance indicators
        instance_name = tags.get('Name', '').lower()
        nat_indicators = ['nat', 'nat-instance', 'nat-gateway']
        
        for indicator in nat_indicators:
            if indicator in instance_name:
                return True
        
        # Check if source/dest check is disabled (common for NAT instances)
        if instance.get('SourceDestCheck') is False:
            return True
        
        return False
    
    def _check_domain_controller(self, tags: Dict[str, str]) -> bool:
        instance_name = tags.get('Name', '').lower()
        dc_indicators = ['domain-controller', 'dc', 'active-directory', 'ad-']
        
        for indicator in dc_indicators:
            if indicator in instance_name:
                return True
        
        # Check for specific DC-related tags
        for tag_key, tag_value in tags.items():
            if 'domain' in tag_key.lower() or 'directory' in tag_key.lower():
                return True
        
        return False
    
    def _check_recent_activity(self, instance_id: str, region: str) -> bool:
        if not region:
            return False
        
        try:
            cloudwatch = boto3.client('cloudwatch', region_name=region)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Check recent CPU activity
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Maximum']
            )
            
            if response['Datapoints']:
                recent_max = max(dp['Maximum'] for dp in response['Datapoints'])
                return recent_max > 50  # Active if CPU > 50% in last hour
            
        except ClientError:
            pass
        
        return False
    
    def _calculate_risk_score(
        self,
        warnings: List[str],
        blockers: List[str],
        safety_checks: Dict[str, bool]
    ) -> int:
        score = 0
        
        # Blockers are high risk
        score += len(blockers) * 10
        
        # Warnings are medium risk
        score += len(warnings) * 5
        
        # Specific safety check failures
        if safety_checks.get('has_elastic_ip'):
            score += 3
        if safety_checks.get('has_load_balancer'):
            score += 5
        if safety_checks.get('has_iam_role'):
            score += 2
        if safety_checks.get('has_recent_activity'):
            score += 4
        
        return min(score, 100)  # Cap at 100
    
    def check_volume_safety(self, volume: Dict[str, Any]) -> Dict[str, Any]:
        volume_id = volume['VolumeId']
        warnings = []
        blockers = []
        
        # Check volume tags
        tags = {tag['Key']: tag['Value'] for tag in volume.get('Tags', [])}
        
        # Check for critical tags
        for tag_key, tag_value in tags.items():
            if tag_key in self.critical_tags:
                if tag_value.lower() in ['true', 'yes', '1']:
                    blockers.append(f"Volume has {tag_key} tag set to {tag_value}")
        
        # Check if volume is encrypted
        if volume.get('Encrypted'):
            warnings.append("Volume is encrypted")
        
        # Check volume type
        if volume.get('VolumeType') in ['io1', 'io2']:
            warnings.append(f"Volume is high-performance type: {volume['VolumeType']}")
        
        # Check size
        if volume.get('Size', 0) > 1000:  # 1TB
            warnings.append(f"Large volume: {volume['Size']} GB")
        
        safe_to_modify = len(blockers) == 0
        
        return {
            'safe_to_modify': safe_to_modify,
            'warnings': warnings,
            'blockers': blockers
        }
    
    def check_s3_bucket_safety(self, bucket_name: str, region: str = None) -> Dict[str, Any]:
        s3 = boto3.client('s3', region_name=region)
        warnings = []
        blockers = []
        
        try:
            # Check bucket versioning
            versioning = s3.get_bucket_versioning(Bucket=bucket_name)
            if versioning.get('Status') == 'Enabled':
                warnings.append("Bucket has versioning enabled")
            
            # Check for lifecycle policies
            try:
                lifecycle = s3.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                if lifecycle.get('Rules'):
                    warnings.append(f"Bucket has {len(lifecycle['Rules'])} lifecycle rules")
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchLifecycleConfiguration':
                    raise
            
            # Check for replication
            try:
                replication = s3.get_bucket_replication(Bucket=bucket_name)
                if replication.get('ReplicationConfiguration'):
                    blockers.append("Bucket has replication configured")
            except ClientError as e:
                if e.response['Error']['Code'] != 'ReplicationConfigurationNotFoundError':
                    raise
            
            # Check for public access
            try:
                public_block = s3.get_public_access_block(Bucket=bucket_name)
                config = public_block['PublicAccessBlockConfiguration']
                if not all([
                    config.get('BlockPublicAcls', False),
                    config.get('IgnorePublicAcls', False),
                    config.get('BlockPublicPolicy', False),
                    config.get('RestrictPublicBuckets', False)
                ]):
                    warnings.append("Bucket may have public access")
            except ClientError:
                warnings.append("Could not verify public access settings")
            
            # Check tags
            try:
                tagging = s3.get_bucket_tagging(Bucket=bucket_name)
                tags = {tag['Key']: tag['Value'] for tag in tagging.get('TagSet', [])}
                
                for tag_key, tag_value in tags.items():
                    if tag_key in self.critical_tags:
                        if tag_value.lower() in ['true', 'yes', '1']:
                            blockers.append(f"Bucket has {tag_key} tag set to {tag_value}")
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchTagSet':
                    raise
        
        except ClientError as e:
            logger.error(f"Error checking S3 bucket safety for {bucket_name}: {e}")
            blockers.append(f"Error accessing bucket: {str(e)}")
        
        safe_to_modify = len(blockers) == 0
        
        return {
            'safe_to_modify': safe_to_modify,
            'warnings': warnings,
            'blockers': blockers
        }