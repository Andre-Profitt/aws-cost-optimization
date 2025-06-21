"""
Auto-Remediation Engine - Automatically applies cost optimization recommendations
Implements safe, reversible actions with approval workflows
"""
import boto3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class RemediationAction(Enum):
    """Types of remediation actions"""
    STOP_INSTANCE = "stop_instance"
    TERMINATE_INSTANCE = "terminate_instance"
    DELETE_SNAPSHOT = "delete_snapshot"
    DELETE_VOLUME = "delete_volume"
    RELEASE_ELASTIC_IP = "release_elastic_ip"
    DELETE_NAT_GATEWAY = "delete_nat_gateway"
    MODIFY_INSTANCE_TYPE = "modify_instance_type"
    PURCHASE_RESERVED_INSTANCE = "purchase_reserved_instance"
    ENABLE_S3_LIFECYCLE = "enable_s3_lifecycle"
    DELETE_OLD_BACKUPS = "delete_old_backups"
    ADJUST_AUTO_SCALING = "adjust_auto_scaling"
    CREATE_BUDGET_ALERT = "create_budget_alert"

class RemediationStatus(Enum):
    """Status of remediation actions"""
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class RemediationTask:
    """Represents a remediation task"""
    task_id: str
    action: RemediationAction
    resource_id: str
    resource_type: str
    region: str
    estimated_savings: float
    risk_level: str  # 'low', 'medium', 'high'
    parameters: Dict[str, Any]
    status: RemediationStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    approval_required: bool = True
    auto_rollback_enabled: bool = True

@dataclass
class RemediationPolicy:
    """Policy for auto-remediation"""
    max_monthly_savings: float  # Maximum savings to auto-approve
    allowed_actions: List[RemediationAction]
    require_approval_for_production: bool
    auto_rollback_on_error: bool
    business_hours_only: bool
    blackout_periods: List[Dict[str, Any]]  # Times when remediation is not allowed
    notification_endpoints: List[str]  # SNS topics, email, etc.

class AutoRemediationEngine:
    """Automated remediation engine with safety controls"""
    
    def __init__(self,
                 policy: RemediationPolicy,
                 dry_run: bool = False,
                 session: Optional[boto3.Session] = None):
        """
        Initialize Auto-Remediation Engine
        
        Args:
            policy: Remediation policy configuration
            dry_run: If True, simulate actions without executing
            session: Boto3 session
        """
        self.policy = policy
        self.dry_run = dry_run
        self.session = session or boto3.Session()
        
        # Initialize clients
        self.ec2 = self.session.client('ec2')
        self.s3 = self.session.client('s3')
        self.ce = self.session.client('ce')
        self.sns = self.session.client('sns')
        self.cloudwatch = self.session.client('cloudwatch')
        self.budgets = self.session.client('budgets')
        
        # Task tracking
        self.tasks: Dict[str, RemediationTask] = {}
        self.task_lock = threading.Lock()
        
        # Action handlers
        self.action_handlers = {
            RemediationAction.STOP_INSTANCE: self._stop_instance,
            RemediationAction.TERMINATE_INSTANCE: self._terminate_instance,
            RemediationAction.DELETE_SNAPSHOT: self._delete_snapshot,
            RemediationAction.DELETE_VOLUME: self._delete_volume,
            RemediationAction.RELEASE_ELASTIC_IP: self._release_elastic_ip,
            RemediationAction.DELETE_NAT_GATEWAY: self._delete_nat_gateway,
            RemediationAction.MODIFY_INSTANCE_TYPE: self._modify_instance_type,
            RemediationAction.ENABLE_S3_LIFECYCLE: self._enable_s3_lifecycle,
            RemediationAction.DELETE_OLD_BACKUPS: self._delete_old_backups,
            RemediationAction.ADJUST_AUTO_SCALING: self._adjust_auto_scaling,
            RemediationAction.CREATE_BUDGET_ALERT: self._create_budget_alert
        }
        
        # Rollback handlers
        self.rollback_handlers = {
            RemediationAction.STOP_INSTANCE: self._rollback_stop_instance,
            RemediationAction.DELETE_SNAPSHOT: self._rollback_delete_snapshot,
            RemediationAction.DELETE_VOLUME: self._rollback_delete_volume,
            RemediationAction.MODIFY_INSTANCE_TYPE: self._rollback_modify_instance_type
        }
    
    def create_remediation_task(self,
                              action: RemediationAction,
                              resource_id: str,
                              resource_type: str,
                              region: str,
                              estimated_savings: float,
                              risk_level: str,
                              parameters: Dict[str, Any] = None) -> RemediationTask:
        """Create a new remediation task"""
        task_id = f"{action.value}-{resource_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Determine if approval is required
        approval_required = self._requires_approval(action, risk_level, estimated_savings)
        
        task = RemediationTask(
            task_id=task_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            region=region,
            estimated_savings=estimated_savings,
            risk_level=risk_level,
            parameters=parameters or {},
            status=RemediationStatus.PENDING,
            created_at=datetime.utcnow(),
            approval_required=approval_required,
            auto_rollback_enabled=self.policy.auto_rollback_on_error
        )
        
        with self.task_lock:
            self.tasks[task_id] = task
        
        logger.info(f"Created remediation task: {task_id}")
        
        # Auto-approve if policy allows
        if not approval_required:
            self.approve_task(task_id)
        
        return task
    
    def _requires_approval(self, action: RemediationAction, risk_level: str, savings: float) -> bool:
        """Determine if a task requires manual approval"""
        # Always require approval for high-risk actions
        if risk_level == 'high':
            return True
        
        # Check if action is in allowed list
        if action not in self.policy.allowed_actions:
            return True
        
        # Check savings threshold
        if savings > self.policy.max_monthly_savings:
            return True
        
        # Check for production resources (simplified check)
        if self.policy.require_approval_for_production:
            # Would check tags, naming conventions, etc.
            return True
        
        return False
    
    def approve_task(self, task_id: str) -> bool:
        """Approve a remediation task"""
        with self.task_lock:
            if task_id not in self.tasks:
                logger.error(f"Task {task_id} not found")
                return False
            
            task = self.tasks[task_id]
            if task.status != RemediationStatus.PENDING:
                logger.error(f"Task {task_id} is not in pending state")
                return False
            
            task.status = RemediationStatus.APPROVED
            logger.info(f"Approved task: {task_id}")
            
        return True
    
    def execute_approved_tasks(self, max_concurrent: int = 5) -> Dict[str, Any]:
        """Execute all approved tasks"""
        results = {
            'executed': 0,
            'succeeded': 0,
            'failed': 0,
            'total_savings': 0,
            'task_results': []
        }
        
        # Get approved tasks
        approved_tasks = []
        with self.task_lock:
            for task in self.tasks.values():
                if task.status == RemediationStatus.APPROVED:
                    approved_tasks.append(task)
        
        if not approved_tasks:
            logger.info("No approved tasks to execute")
            return results
        
        # Check if we're in allowed execution window
        if not self._is_execution_allowed():
            logger.warning("Execution not allowed at this time (business hours/blackout period)")
            return results
        
        # Execute tasks concurrently
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_task = {
                executor.submit(self._execute_task, task): task 
                for task in approved_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success = future.result()
                    results['executed'] += 1
                    
                    if success:
                        results['succeeded'] += 1
                        results['total_savings'] += task.estimated_savings
                    else:
                        results['failed'] += 1
                    
                    results['task_results'].append({
                        'task_id': task.task_id,
                        'status': task.status.value,
                        'error': task.error_message
                    })
                    
                except Exception as e:
                    logger.error(f"Error executing task {task.task_id}: {e}")
                    results['failed'] += 1
        
        # Send summary notification
        self._send_execution_summary(results)
        
        return results
    
    def _is_execution_allowed(self) -> bool:
        """Check if execution is allowed at current time"""
        now = datetime.utcnow()
        
        # Check business hours restriction
        if self.policy.business_hours_only:
            # Simple check: weekday 9 AM - 5 PM UTC
            if now.weekday() >= 5:  # Weekend
                return False
            if now.hour < 9 or now.hour >= 17:
                return False
        
        # Check blackout periods
        for blackout in self.policy.blackout_periods:
            start = datetime.fromisoformat(blackout['start'])
            end = datetime.fromisoformat(blackout['end'])
            if start <= now <= end:
                return False
        
        return True
    
    def _execute_task(self, task: RemediationTask) -> bool:
        """Execute a single remediation task"""
        logger.info(f"Executing task: {task.task_id}")
        
        # Update status
        with self.task_lock:
            task.status = RemediationStatus.IN_PROGRESS
            task.executed_at = datetime.utcnow()
        
        try:
            # Get the appropriate handler
            handler = self.action_handlers.get(task.action)
            if not handler:
                raise ValueError(f"No handler for action: {task.action}")
            
            # Execute in the correct region
            regional_client = self._get_regional_client(task.resource_type, task.region)
            
            # Execute the action
            if self.dry_run:
                logger.info(f"DRY RUN: Would execute {task.action} on {task.resource_id}")
                rollback_info = {"dry_run": True}
            else:
                rollback_info = handler(task, regional_client)
            
            # Update task status
            with self.task_lock:
                task.status = RemediationStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.rollback_info = rollback_info
            
            logger.info(f"Successfully completed task: {task.task_id}")
            
            # Send success notification
            self._send_task_notification(task, success=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute task {task.task_id}: {e}")
            
            # Update task status
            with self.task_lock:
                task.status = RemediationStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()
            
            # Attempt rollback if enabled
            if task.auto_rollback_enabled and task.rollback_info:
                self._attempt_rollback(task)
            
            # Send failure notification
            self._send_task_notification(task, success=False)
            
            return False
    
    def _get_regional_client(self, resource_type: str, region: str):
        """Get appropriate boto3 client for the region"""
        client_map = {
            'instance': 'ec2',
            'volume': 'ec2',
            'snapshot': 'ec2',
            'elastic_ip': 'ec2',
            'nat_gateway': 'ec2',
            'bucket': 's3',
            'auto_scaling_group': 'autoscaling'
        }
        
        service = client_map.get(resource_type, 'ec2')
        return self.session.client(service, region_name=region)
    
    # Action handlers
    def _stop_instance(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Stop an EC2 instance"""
        instance_id = task.resource_id
        
        # Get current state for rollback
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        current_state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        if current_state == 'running':
            ec2_client.stop_instances(InstanceIds=[instance_id])
            logger.info(f"Stopped instance: {instance_id}")
        
        return {'previous_state': current_state}
    
    def _terminate_instance(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Terminate an EC2 instance (no rollback possible)"""
        instance_id = task.resource_id
        
        # Create AMI backup before termination
        ami_response = ec2_client.create_image(
            InstanceId=instance_id,
            Name=f"backup-before-termination-{instance_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            Description=f"Backup created by auto-remediation before terminating {instance_id}"
        )
        
        ami_id = ami_response['ImageId']
        
        # Wait for AMI to be available
        waiter = ec2_client.get_waiter('image_available')
        waiter.wait(ImageIds=[ami_id])
        
        # Terminate instance
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        logger.info(f"Terminated instance: {instance_id}, backup AMI: {ami_id}")
        
        return {'backup_ami_id': ami_id}
    
    def _delete_snapshot(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Delete an EBS snapshot"""
        snapshot_id = task.resource_id
        
        # Get snapshot details for potential recovery
        response = ec2_client.describe_snapshots(SnapshotIds=[snapshot_id])
        snapshot_info = response['Snapshots'][0]
        
        # Store snapshot metadata
        metadata = {
            'volume_id': snapshot_info.get('VolumeId'),
            'volume_size': snapshot_info.get('VolumeSize'),
            'description': snapshot_info.get('Description'),
            'tags': snapshot_info.get('Tags', [])
        }
        
        ec2_client.delete_snapshot(SnapshotId=snapshot_id)
        logger.info(f"Deleted snapshot: {snapshot_id}")
        
        return {'snapshot_metadata': metadata}
    
    def _delete_volume(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Delete an EBS volume"""
        volume_id = task.resource_id
        
        # Create snapshot before deletion
        snapshot_response = ec2_client.create_snapshot(
            VolumeId=volume_id,
            Description=f"Backup before auto-deletion of {volume_id}"
        )
        
        snapshot_id = snapshot_response['SnapshotId']
        
        # Wait for snapshot to complete
        waiter = ec2_client.get_waiter('snapshot_completed')
        waiter.wait(SnapshotIds=[snapshot_id])
        
        # Delete volume
        ec2_client.delete_volume(VolumeId=volume_id)
        logger.info(f"Deleted volume: {volume_id}, backup snapshot: {snapshot_id}")
        
        return {'backup_snapshot_id': snapshot_id}
    
    def _release_elastic_ip(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Release an Elastic IP"""
        allocation_id = task.resource_id
        
        # Get IP details before release
        response = ec2_client.describe_addresses(AllocationIds=[allocation_id])
        ip_info = response['Addresses'][0]
        
        ec2_client.release_address(AllocationId=allocation_id)
        logger.info(f"Released Elastic IP: {allocation_id} ({ip_info.get('PublicIp')})")
        
        return {'public_ip': ip_info.get('PublicIp')}
    
    def _delete_nat_gateway(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Delete a NAT Gateway"""
        nat_gateway_id = task.resource_id
        
        # Get NAT Gateway details
        response = ec2_client.describe_nat_gateways(NatGatewayIds=[nat_gateway_id])
        nat_info = response['NatGateways'][0]
        
        # Store configuration for potential recreation
        config = {
            'subnet_id': nat_info.get('SubnetId'),
            'allocation_id': nat_info.get('NatGatewayAddresses', [{}])[0].get('AllocationId')
        }
        
        ec2_client.delete_nat_gateway(NatGatewayId=nat_gateway_id)
        logger.info(f"Deleted NAT Gateway: {nat_gateway_id}")
        
        return {'nat_gateway_config': config}
    
    def _modify_instance_type(self, task: RemediationTask, ec2_client) -> Dict[str, Any]:
        """Modify EC2 instance type"""
        instance_id = task.resource_id
        new_type = task.parameters.get('new_instance_type')
        
        # Get current instance type
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        current_type = instance['InstanceType']
        current_state = instance['State']['Name']
        
        # Stop instance if running
        if current_state == 'running':
            ec2_client.stop_instances(InstanceIds=[instance_id])
            waiter = ec2_client.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[instance_id])
        
        # Modify instance type
        ec2_client.modify_instance_attribute(
            InstanceId=instance_id,
            InstanceType={'Value': new_type}
        )
        
        # Start instance if it was running
        if current_state == 'running':
            ec2_client.start_instances(InstanceIds=[instance_id])
        
        logger.info(f"Modified instance type: {instance_id} from {current_type} to {new_type}")
        
        return {
            'previous_type': current_type,
            'previous_state': current_state
        }
    
    def _enable_s3_lifecycle(self, task: RemediationTask, s3_client) -> Dict[str, Any]:
        """Enable S3 lifecycle policy"""
        bucket_name = task.resource_id
        days_to_glacier = task.parameters.get('days_to_glacier', 30)
        days_to_delete = task.parameters.get('days_to_delete', 365)
        
        lifecycle_policy = {
            'Rules': [{
                'ID': 'auto-remediation-lifecycle',
                'Status': 'Enabled',
                'Transitions': [{
                    'Days': days_to_glacier,
                    'StorageClass': 'GLACIER'
                }],
                'Expiration': {
                    'Days': days_to_delete
                }
            }]
        }
        
        # Get existing lifecycle policy for rollback
        try:
            existing = s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
            existing_policy = existing.get('Rules', [])
        except s3_client.exceptions.NoSuchLifecycleConfiguration:
            existing_policy = None
        
        # Apply new policy
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
        
        logger.info(f"Enabled lifecycle policy on S3 bucket: {bucket_name}")
        
        return {'previous_policy': existing_policy}
    
    def _delete_old_backups(self, task: RemediationTask, client) -> Dict[str, Any]:
        """Delete old backup snapshots"""
        days_old = task.parameters.get('days_old', 90)
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        deleted_snapshots = []
        
        # Get snapshots owned by this account
        response = client.describe_snapshots(OwnerIds=['self'])
        
        for snapshot in response['Snapshots']:
            start_time = snapshot['StartTime'].replace(tzinfo=None)
            
            if start_time < cutoff_date:
                # Check if snapshot is tagged as backup
                tags = {tag['Key']: tag['Value'] for tag in snapshot.get('Tags', [])}
                
                if tags.get('Type') == 'backup' or 'backup' in snapshot.get('Description', '').lower():
                    snapshot_id = snapshot['SnapshotId']
                    client.delete_snapshot(SnapshotId=snapshot_id)
                    deleted_snapshots.append(snapshot_id)
                    logger.info(f"Deleted old backup snapshot: {snapshot_id}")
        
        return {'deleted_snapshots': deleted_snapshots}
    
    def _adjust_auto_scaling(self, task: RemediationTask, autoscaling_client) -> Dict[str, Any]:
        """Adjust Auto Scaling Group settings"""
        asg_name = task.resource_id
        new_min = task.parameters.get('min_size')
        new_max = task.parameters.get('max_size')
        new_desired = task.parameters.get('desired_capacity')
        
        # Get current configuration
        response = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[asg_name]
        )
        asg = response['AutoScalingGroups'][0]
        
        old_config = {
            'min_size': asg['MinSize'],
            'max_size': asg['MaxSize'],
            'desired_capacity': asg['DesiredCapacity']
        }
        
        # Update configuration
        update_params = {'AutoScalingGroupName': asg_name}
        if new_min is not None:
            update_params['MinSize'] = new_min
        if new_max is not None:
            update_params['MaxSize'] = new_max
        if new_desired is not None:
            update_params['DesiredCapacity'] = new_desired
        
        autoscaling_client.update_auto_scaling_group(**update_params)
        
        logger.info(f"Adjusted Auto Scaling Group: {asg_name}")
        
        return {'previous_config': old_config}
    
    def _create_budget_alert(self, task: RemediationTask, budgets_client) -> Dict[str, Any]:
        """Create a budget alert"""
        budget_name = task.parameters.get('budget_name', f"auto-budget-{datetime.utcnow().strftime('%Y%m%d')}")
        amount = task.parameters.get('amount', 1000)
        
        budget = {
            'BudgetName': budget_name,
            'BudgetLimit': {
                'Amount': str(amount),
                'Unit': 'USD'
            },
            'TimeUnit': 'MONTHLY',
            'BudgetType': 'COST'
        }
        
        notifications = [{
            'NotificationType': 'ACTUAL',
            'ComparisonOperator': 'GREATER_THAN',
            'Threshold': 80,
            'ThresholdType': 'PERCENTAGE',
            'NotificationState': 'ALARM'
        }]
        
        subscribers = [{
            'SubscriptionType': 'SNS',
            'Address': topic_arn
        } for topic_arn in self.policy.notification_endpoints]
        
        budgets_client.create_budget(
            AccountId=self.session.client('sts').get_caller_identity()['Account'],
            Budget=budget,
            NotificationsWithSubscribers=[{
                'Notification': notifications[0],
                'Subscribers': subscribers
            }]
        )
        
        logger.info(f"Created budget alert: {budget_name}")
        
        return {'budget_name': budget_name}
    
    # Rollback handlers
    def _rollback_stop_instance(self, task: RemediationTask, ec2_client):
        """Rollback instance stop by starting it"""
        if task.rollback_info.get('previous_state') == 'running':
            ec2_client.start_instances(InstanceIds=[task.resource_id])
            logger.info(f"Rolled back: Started instance {task.resource_id}")
    
    def _rollback_delete_snapshot(self, task: RemediationTask, ec2_client):
        """Cannot rollback snapshot deletion, log metadata"""
        logger.warning(f"Cannot rollback snapshot deletion. Metadata: {task.rollback_info}")
    
    def _rollback_delete_volume(self, task: RemediationTask, ec2_client):
        """Cannot rollback volume deletion, but we have snapshot"""
        snapshot_id = task.rollback_info.get('backup_snapshot_id')
        logger.warning(f"Cannot rollback volume deletion. Backup snapshot: {snapshot_id}")
    
    def _rollback_modify_instance_type(self, task: RemediationTask, ec2_client):
        """Rollback instance type modification"""
        instance_id = task.resource_id
        previous_type = task.rollback_info.get('previous_type')
        previous_state = task.rollback_info.get('previous_state')
        
        # Stop instance
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        current_state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        if current_state == 'running':
            ec2_client.stop_instances(InstanceIds=[instance_id])
            waiter = ec2_client.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[instance_id])
        
        # Revert instance type
        ec2_client.modify_instance_attribute(
            InstanceId=instance_id,
            InstanceType={'Value': previous_type}
        )
        
        # Restore previous state
        if previous_state == 'running':
            ec2_client.start_instances(InstanceIds=[instance_id])
        
        logger.info(f"Rolled back instance type for {instance_id} to {previous_type}")
    
    def _attempt_rollback(self, task: RemediationTask):
        """Attempt to rollback a failed task"""
        logger.info(f"Attempting rollback for task: {task.task_id}")
        
        rollback_handler = self.rollback_handlers.get(task.action)
        if not rollback_handler:
            logger.warning(f"No rollback handler for action: {task.action}")
            return
        
        try:
            regional_client = self._get_regional_client(task.resource_type, task.region)
            rollback_handler(task, regional_client)
            
            with self.task_lock:
                task.status = RemediationStatus.ROLLED_BACK
                
            logger.info(f"Successfully rolled back task: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to rollback task {task.task_id}: {e}")
    
    def _send_task_notification(self, task: RemediationTask, success: bool):
        """Send notification about task execution"""
        if not self.policy.notification_endpoints:
            return
        
        subject = f"Auto-Remediation {'Success' if success else 'Failed'}: {task.action.value}"
        
        message = f"""
Auto-Remediation Task {'Completed' if success else 'Failed'}

Task ID: {task.task_id}
Action: {task.action.value}
Resource: {task.resource_id}
Region: {task.region}
Status: {task.status.value}
Estimated Savings: ${task.estimated_savings:.2f}/month

{'Error: ' + task.error_message if task.error_message else ''}

Execution Time: {task.executed_at.strftime('%Y-%m-%d %H:%M:%S UTC') if task.executed_at else 'N/A'}
Completion Time: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC') if task.completed_at else 'N/A'}
"""
        
        for endpoint in self.policy.notification_endpoints:
            try:
                self.sns.publish(
                    TopicArn=endpoint,
                    Subject=subject,
                    Message=message
                )
            except Exception as e:
                logger.error(f"Failed to send notification to {endpoint}: {e}")
    
    def _send_execution_summary(self, results: Dict[str, Any]):
        """Send summary of execution results"""
        if not self.policy.notification_endpoints:
            return
        
        subject = "Auto-Remediation Execution Summary"
        
        message = f"""
Auto-Remediation Execution Summary

Total Tasks Executed: {results['executed']}
Succeeded: {results['succeeded']}
Failed: {results['failed']}

Total Estimated Savings: ${results['total_savings']:.2f}/month
Annual Savings: ${results['total_savings'] * 12:.2f}

Execution Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        for endpoint in self.policy.notification_endpoints:
            try:
                self.sns.publish(
                    TopicArn=endpoint,
                    Subject=subject,
                    Message=message
                )
            except Exception as e:
                logger.error(f"Failed to send summary to {endpoint}: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[RemediationTask]:
        """Get status of a specific task"""
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self, status_filter: Optional[RemediationStatus] = None) -> List[RemediationTask]:
        """Get all tasks, optionally filtered by status"""
        with self.task_lock:
            tasks = list(self.tasks.values())
            
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def export_task_report(self, output_file: str = 'remediation_report.json'):
        """Export detailed task report"""
        tasks_data = []
        
        for task in self.get_all_tasks():
            tasks_data.append({
                'task_id': task.task_id,
                'action': task.action.value,
                'resource_id': task.resource_id,
                'region': task.region,
                'status': task.status.value,
                'estimated_savings': task.estimated_savings,
                'risk_level': task.risk_level,
                'created_at': task.created_at.isoformat(),
                'executed_at': task.executed_at.isoformat() if task.executed_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error_message': task.error_message
            })
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_tasks': len(tasks_data),
            'tasks_by_status': {},
            'total_savings': sum(t['estimated_savings'] for t in tasks_data if t['status'] == 'completed'),
            'tasks': tasks_data
        }
        
        # Count by status
        for task in tasks_data:
            status = task['status']
            report['tasks_by_status'][status] = report['tasks_by_status'].get(status, 0) + 1
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported remediation report to {output_file}")