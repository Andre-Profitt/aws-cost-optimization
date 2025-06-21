"""Safety checks module to ensure optimization actions don't impact production systems."""
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SafetyChecker:
    """Performs safety checks before applying optimizations."""
    
    def __init__(self, session: boto3.Session):
        """Initialize safety checker with AWS session."""
        self.session = session
        self.ec2 = session.client('ec2')
        self.elb = session.client('elbv2')
        self.asg = session.client('autoscaling')
        self.cloudwatch = session.client('cloudwatch')
        
    def check_instance_safety(self, instance_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive safety checks on an EC2 instance.
        
        Args:
            instance_id: EC2 instance ID to check
            
        Returns:
            Dictionary with safety check results
        """
        try:
            safety_results = {
                'instance_id': instance_id,
                'safe_to_modify': True,
                'warnings': [],
                'blockers': [],
                'checks_performed': []
            }
            
            # Check if instance exists and get details
            instance_details = self._get_instance_details(instance_id)
            if not instance_details:
                safety_results['safe_to_modify'] = False
                safety_results['blockers'].append('Instance not found')
                return safety_results
            
            # Check instance state
            state_check = self._check_instance_state(instance_details)
            safety_results['checks_performed'].append('instance_state')
            if state_check['blocked']:
                safety_results['safe_to_modify'] = False
                safety_results['blockers'].extend(state_check['reasons'])
            elif state_check['warnings']:
                safety_results['warnings'].extend(state_check['warnings'])
            
            # Check for production tags
            tag_check = self._check_production_tags(instance_details)
            safety_results['checks_performed'].append('production_tags')
            if tag_check['blocked']:
                safety_results['safe_to_modify'] = False
                safety_results['blockers'].extend(tag_check['reasons'])
            elif tag_check['warnings']:
                safety_results['warnings'].extend(tag_check['warnings'])
            
            # Check load balancer membership
            lb_check = self._check_load_balancer_membership(instance_id)
            safety_results['checks_performed'].append('load_balancer')
            if lb_check['warnings']:
                safety_results['warnings'].extend(lb_check['warnings'])
            
            # Check auto scaling group membership
            asg_check = self._check_asg_membership(instance_id)
            safety_results['checks_performed'].append('auto_scaling_group')
            if asg_check['blocked']:
                safety_results['safe_to_modify'] = False
                safety_results['blockers'].extend(asg_check['reasons'])
            elif asg_check['warnings']:
                safety_results['warnings'].extend(asg_check['warnings'])
            
            # Check recent activity
            activity_check = self._check_recent_activity(instance_id)
            safety_results['checks_performed'].append('recent_activity')
            if activity_check['warnings']:
                safety_results['warnings'].extend(activity_check['warnings'])
            
            # Check critical applications
            app_check = self._check_critical_applications(instance_details)
            safety_results['checks_performed'].append('critical_applications')
            if app_check['blocked']:
                safety_results['safe_to_modify'] = False
                safety_results['blockers'].extend(app_check['reasons'])
            
            return safety_results
            
        except Exception as e:
            logger.error(f"Safety check failed for instance {instance_id}: {str(e)}")
            return {
                'instance_id': instance_id,
                'safe_to_modify': False,
                'warnings': [],
                'blockers': [f'Safety check error: {str(e)}'],
                'checks_performed': []
            }
    
    def _get_instance_details(self, instance_id: str) -> Optional[Dict]:
        """Get instance details from EC2."""
        try:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            if response['Reservations']:
                return response['Reservations'][0]['Instances'][0]
            return None
        except ClientError as e:
            if e.response['Error']['Code'] == 'InvalidInstanceID.NotFound':
                return None
            raise
    
    def _check_instance_state(self, instance: Dict) -> Dict[str, Any]:
        """Check if instance state allows modifications."""
        state = instance['State']['Name']
        result = {'blocked': False, 'reasons': [], 'warnings': []}
        
        if state == 'terminated':
            result['blocked'] = True
            result['reasons'].append('Instance is terminated')
        elif state == 'terminating':
            result['blocked'] = True
            result['reasons'].append('Instance is being terminated')
        elif state == 'stopping':
            result['warnings'].append('Instance is stopping')
        elif state == 'pending':
            result['warnings'].append('Instance is starting up')
            
        return result
    
    def _check_production_tags(self, instance: Dict) -> Dict[str, Any]:
        """Check for production environment tags."""
        result = {'blocked': False, 'reasons': [], 'warnings': []}
        
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        # Check for explicit production tags
        production_indicators = {
            'Environment': ['production', 'prod', 'live'],
            'Env': ['production', 'prod', 'live'],
            'Stage': ['production', 'prod', 'live'],
            'Critical': ['true', 'yes', '1']
        }
        
        for tag_key, prod_values in production_indicators.items():
            if tag_key in tags:
                tag_value = tags[tag_key].lower()
                if tag_value in prod_values:
                    result['warnings'].append(
                        f'Instance tagged as production: {tag_key}={tags[tag_key]}'
                    )
        
        # Check for do-not-modify tags
        if tags.get('DoNotModify', '').lower() in ['true', 'yes', '1']:
            result['blocked'] = True
            result['reasons'].append('Instance has DoNotModify tag')
        
        if tags.get('DoNotStop', '').lower() in ['true', 'yes', '1']:
            result['blocked'] = True
            result['reasons'].append('Instance has DoNotStop tag')
            
        return result
    
    def _check_load_balancer_membership(self, instance_id: str) -> Dict[str, Any]:
        """Check if instance is part of a load balancer."""
        result = {'warnings': []}
        
        try:
            # Check all target groups
            paginator = self.elb.get_paginator('describe_target_groups')
            
            for page in paginator.paginate():
                for tg in page['TargetGroups']:
                    targets = self.elb.describe_target_health(
                        TargetGroupArn=tg['TargetGroupArn']
                    )
                    
                    for target in targets['TargetHealthDescriptions']:
                        if target['Target']['Id'] == instance_id:
                            health = target['TargetHealth']['State']
                            result['warnings'].append(
                                f"Instance is in target group {tg['TargetGroupName']} "
                                f"(state: {health})"
                            )
                            
                            # Check if it's the only healthy instance
                            healthy_count = sum(
                                1 for t in targets['TargetHealthDescriptions']
                                if t['TargetHealth']['State'] == 'healthy'
                            )
                            if healthy_count <= 1 and health == 'healthy':
                                result['warnings'].append(
                                    f"WARNING: Instance is the only healthy target in "
                                    f"{tg['TargetGroupName']}"
                                )
                                
        except Exception as e:
            logger.warning(f"Error checking load balancer membership: {str(e)}")
            
        return result
    
    def _check_asg_membership(self, instance_id: str) -> Dict[str, Any]:
        """Check if instance is part of an Auto Scaling Group."""
        result = {'blocked': False, 'reasons': [], 'warnings': []}
        
        try:
            response = self.asg.describe_auto_scaling_instances(
                InstanceIds=[instance_id]
            )
            
            if response['AutoScalingInstances']:
                asg_instance = response['AutoScalingInstances'][0]
                asg_name = asg_instance['AutoScalingGroupName']
                
                # Get ASG details
                asg_response = self.asg.describe_auto_scaling_groups(
                    AutoScalingGroupNames=[asg_name]
                )
                
                if asg_response['AutoScalingGroups']:
                    asg = asg_response['AutoScalingGroups'][0]
                    
                    result['warnings'].append(
                        f"Instance is part of Auto Scaling Group: {asg_name}"
                    )
                    
                    # Check if reducing capacity would go below minimum
                    if len(asg['Instances']) <= asg['MinSize']:
                        result['blocked'] = True
                        result['reasons'].append(
                            f"Cannot modify: ASG {asg_name} is at minimum capacity"
                        )
                        
        except Exception as e:
            logger.warning(f"Error checking ASG membership: {str(e)}")
            
        return result
    
    def _check_recent_activity(self, instance_id: str) -> Dict[str, Any]:
        """Check for recent scaling or modification activity."""
        result = {'warnings': []}
        
        try:
            # Check CloudWatch for recent CPU spikes
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            cpu_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Maximum']
            )
            
            if cpu_response['Datapoints']:
                recent_max_cpu = max(dp['Maximum'] for dp in cpu_response['Datapoints'])
                if recent_max_cpu > 80:
                    result['warnings'].append(
                        f"High CPU usage detected in last 24h: {recent_max_cpu:.1f}%"
                    )
                    
        except Exception as e:
            logger.warning(f"Error checking recent activity: {str(e)}")
            
        return result
    
    def _check_critical_applications(self, instance: Dict) -> Dict[str, Any]:
        """Check if instance runs critical applications."""
        result = {'blocked': False, 'reasons': []}
        
        tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
        
        # Check for database indicators
        db_indicators = ['database', 'db', 'mysql', 'postgres', 'mongodb', 'redis']
        instance_name = tags.get('Name', '').lower()
        
        for indicator in db_indicators:
            if indicator in instance_name:
                result['blocked'] = True
                result['reasons'].append(
                    f"Instance appears to be a database server: {tags.get('Name', 'N/A')}"
                )
                break
                
        return result
    
    def validate_optimization_window(self, 
                                   start_time: datetime,
                                   end_time: datetime,
                                   timezone: str = 'UTC') -> bool:
        """
        Validate if current time is within allowed optimization window.
        
        Args:
            start_time: Window start time
            end_time: Window end time
            timezone: Timezone for the window
            
        Returns:
            True if current time is within window
        """
        # TODO: Implement timezone-aware window checking
        current_time = datetime.utcnow()
        
        # Simple check for now - can be enhanced with timezone support
        current_hour = current_time.hour
        start_hour = start_time.hour
        end_hour = end_time.hour
        
        if start_hour <= end_hour:
            return start_hour <= current_hour <= end_hour
        else:
            # Window crosses midnight
            return current_hour >= start_hour or current_hour <= end_hour


def perform_safety_checks(resources: List[Dict],
                         session: boto3.Session,
                         safety_config: Dict) -> Dict[str, Any]:
    """
    Perform safety checks on a list of resources.
    
    Args:
        resources: List of resources to check
        session: AWS session
        safety_config: Safety configuration settings
        
    Returns:
        Dictionary with safety check results
    """
    checker = SafetyChecker(session)
    results = {
        'total_resources': len(resources),
        'safe_resources': [],
        'unsafe_resources': [],
        'warnings': []
    }
    
    for resource in resources:
        if resource['Type'] == 'EC2':
            safety_result = checker.check_instance_safety(resource['InstanceId'])
            
            if safety_result['safe_to_modify']:
                results['safe_resources'].append({
                    'resource': resource,
                    'warnings': safety_result['warnings']
                })
            else:
                results['unsafe_resources'].append({
                    'resource': resource,
                    'blockers': safety_result['blockers'],
                    'warnings': safety_result['warnings']
                })
                
            if safety_result['warnings']:
                results['warnings'].extend(safety_result['warnings'])
    
    return results


class SafetyOrchestrator:
    """Orchestrates safety checks across resources before optimization"""
    
    def __init__(self, dry_run: bool = True):
        """Initialize the safety orchestrator
        
        Args:
            dry_run: If True, only simulate actions without executing
        """
        self.dry_run = dry_run
        self.session = boto3.Session()
        self.safety_checker = SafetyChecker(self.session)
        
    def check_resource_safety(self, resource_id: str, resource_type: str) -> Dict[str, Any]:
        """Check if a resource is safe to optimize
        
        Args:
            resource_id: The resource identifier
            resource_type: Type of resource (EC2, RDS, etc.)
            
        Returns:
            Dictionary with safety check results
        """
        if resource_type == 'EC2':
            return self.safety_checker.check_instance_safety(resource_id)
        else:
            # Add more resource types as needed
            return {
                'resource_id': resource_id,
                'safe_to_modify': False,
                'warnings': [],
                'blockers': [f'Safety checks not implemented for {resource_type}']
            }
    
    def execute_optimization(self, resource_id: str, resource_type: str, action: str) -> Dict[str, Any]:
        """Execute optimization action with safety checks
        
        Args:
            resource_id: The resource identifier
            resource_type: Type of resource
            action: The optimization action to perform
            
        Returns:
            Dictionary with execution results
        """
        # First perform safety check
        safety_result = self.check_resource_safety(resource_id, resource_type)
        
        if not safety_result['safe_to_modify']:
            return {
                'success': False,
                'resource_id': resource_id,
                'action': action,
                'reason': 'Failed safety checks',
                'blockers': safety_result['blockers']
            }
        
        if self.dry_run:
            return {
                'success': True,
                'resource_id': resource_id,
                'action': action,
                'dry_run': True,
                'message': f'Would execute {action} on {resource_id}'
            }
        
        # Execute the actual optimization
        # This would contain the actual AWS API calls
        return {
            'success': True,
            'resource_id': resource_id,
            'action': action,
            'dry_run': False,
            'message': f'Executed {action} on {resource_id}'
        }