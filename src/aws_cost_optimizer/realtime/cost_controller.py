"""
Real-time Cost Control System

Implements circuit breakers, EventBridge integration, and automated responses
to cost anomalies and threshold breaches.
"""

import json
import boto3
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class ControlAction(Enum):
    """Types of cost control actions"""
    ALERT = "alert"
    THROTTLE = "throttle"
    SHUTDOWN = "shutdown"
    SCALE_DOWN = "scale_down"
    REQUIRE_APPROVAL = "require_approval"
    EMERGENCY_STOP = "emergency_stop"


class ThresholdType(Enum):
    """Types of cost thresholds"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SERVICE = "service"
    RESOURCE = "resource"
    ANOMALY = "anomaly"


@dataclass
class CostThreshold:
    """Defines a cost threshold and associated action"""
    threshold_id: str
    threshold_type: ThresholdType
    value: float
    action: ControlAction
    target: Optional[str] = None  # Service or resource ID
    notification_targets: List[str] = field(default_factory=list)  # SNS topics
    cooldown_minutes: int = 60
    enabled: bool = True
    
    
@dataclass
class CircuitBreaker:
    """Circuit breaker for cost control"""
    breaker_id: str
    service: str
    threshold: float
    current_spend: float = 0.0
    is_open: bool = False
    opened_at: Optional[datetime] = None
    failure_count: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)
    
    
@dataclass
class CostEvent:
    """Represents a cost-related event"""
    event_id: str
    event_type: str
    timestamp: datetime
    service: Optional[str]
    resource_id: Optional[str]
    current_cost: float
    threshold_breached: Optional[CostThreshold]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealtimeCostController:
    """Real-time cost monitoring and control system"""
    
    def __init__(self,
                 thresholds: List[CostThreshold],
                 enable_circuit_breakers: bool = True,
                 eventbridge_client=None,
                 ce_client=None,
                 sns_client=None,
                 lambda_client=None):
        """
        Initialize real-time cost controller
        
        Args:
            thresholds: List of cost thresholds to monitor
            enable_circuit_breakers: Enable circuit breaker functionality
            eventbridge_client: Boto3 EventBridge client
            ce_client: Boto3 Cost Explorer client
            sns_client: Boto3 SNS client
            lambda_client: Boto3 Lambda client
        """
        self.thresholds = {t.threshold_id: t for t in thresholds}
        self.enable_circuit_breakers = enable_circuit_breakers
        
        # AWS clients
        self.eventbridge = eventbridge_client or boto3.client('events')
        self.ce_client = ce_client or boto3.client('ce')
        self.sns_client = sns_client or boto3.client('sns')
        self.lambda_client = lambda_client or boto3.client('lambda')
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Action history
        self.action_history: List[Dict[str, Any]] = []
        
    def setup_eventbridge_rules(self, rule_prefix: str = 'CostOptimizer') -> List[str]:
        """
        Set up EventBridge rules for cost monitoring
        
        Args:
            rule_prefix: Prefix for rule names
            
        Returns:
            List of created rule ARNs
        """
        logger.info("Setting up EventBridge rules for cost monitoring")
        rule_arns = []
        
        # Rule 1: Daily cost check
        daily_rule = self._create_scheduled_rule(
            rule_name=f"{rule_prefix}-DailyCostCheck",
            schedule="rate(1 hour)",
            description="Hourly cost monitoring and threshold checks",
            target_lambda=self._get_or_create_lambda_function("cost_monitor")
        )
        rule_arns.append(daily_rule)
        
        # Rule 2: Anomaly detection
        anomaly_rule = self._create_scheduled_rule(
            rule_name=f"{rule_prefix}-AnomalyDetection",
            schedule="rate(15 minutes)",
            description="Real-time cost anomaly detection",
            target_lambda=self._get_or_create_lambda_function("anomaly_detector")
        )
        rule_arns.append(anomaly_rule)
        
        # Rule 3: EC2 state change monitoring
        ec2_rule = self._create_event_pattern_rule(
            rule_name=f"{rule_prefix}-EC2StateChange",
            event_pattern={
                "source": ["aws.ec2"],
                "detail-type": ["EC2 Instance State-change Notification"],
                "detail": {
                    "state": ["running", "terminated", "stopped"]
                }
            },
            description="Monitor EC2 state changes for cost impact",
            target_lambda=self._get_or_create_lambda_function("resource_change_handler")
        )
        rule_arns.append(ec2_rule)
        
        # Rule 4: Auto Scaling events
        asg_rule = self._create_event_pattern_rule(
            rule_name=f"{rule_prefix}-AutoScalingEvents",
            event_pattern={
                "source": ["aws.autoscaling"],
                "detail-type": [
                    "EC2 Instance Launch Successful",
                    "EC2 Instance Terminate Successful"
                ]
            },
            description="Monitor Auto Scaling events for cost impact",
            target_lambda=self._get_or_create_lambda_function("scaling_event_handler")
        )
        rule_arns.append(asg_rule)
        
        # Rule 5: Spot instance interruption
        spot_rule = self._create_event_pattern_rule(
            rule_name=f"{rule_prefix}-SpotInterruption",
            event_pattern={
                "source": ["aws.ec2"],
                "detail-type": ["EC2 Spot Instance Interruption Warning"]
            },
            description="Handle Spot instance interruptions",
            target_lambda=self._get_or_create_lambda_function("spot_interruption_handler")
        )
        rule_arns.append(spot_rule)
        
        logger.info(f"Created {len(rule_arns)} EventBridge rules")
        return rule_arns
    
    def check_thresholds(self, current_costs: Dict[str, float]) -> List[CostEvent]:
        """
        Check current costs against defined thresholds
        
        Args:
            current_costs: Dictionary of service/resource costs
            
        Returns:
            List of cost events for breached thresholds
        """
        events = []
        
        for threshold_id, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
            
            # Check if threshold is breached
            breached = False
            current_value = 0.0
            
            if threshold.threshold_type == ThresholdType.SERVICE:
                if threshold.target in current_costs:
                    current_value = current_costs[threshold.target]
                    breached = current_value > threshold.value
            
            elif threshold.threshold_type == ThresholdType.DAILY:
                total_daily = sum(current_costs.values())
                current_value = total_daily
                breached = total_daily > threshold.value
            
            if breached:
                event = CostEvent(
                    event_id=f"breach_{threshold_id}_{datetime.utcnow().timestamp()}",
                    event_type="threshold_breach",
                    timestamp=datetime.utcnow(),
                    service=threshold.target,
                    resource_id=None,
                    current_cost=current_value,
                    threshold_breached=threshold,
                    metadata={
                        'threshold_value': threshold.value,
                        'exceeded_by': current_value - threshold.value,
                        'percentage_over': ((current_value - threshold.value) / threshold.value) * 100
                    }
                )
                events.append(event)
                
                # Execute threshold action
                self._execute_threshold_action(event)
        
        return events
    
    def update_circuit_breaker(self, service: str, current_spend: float) -> Optional[CircuitBreaker]:
        """
        Update circuit breaker state for a service
        
        Args:
            service: Service name
            current_spend: Current spend amount
            
        Returns:
            CircuitBreaker if tripped, None otherwise
        """
        if not self.enable_circuit_breakers:
            return None
        
        breaker = self.circuit_breakers.get(service)
        if not breaker:
            # Create new circuit breaker with default threshold
            breaker = CircuitBreaker(
                breaker_id=f"cb_{service}",
                service=service,
                threshold=self._get_service_threshold(service),
                current_spend=current_spend
            )
            self.circuit_breakers[service] = breaker
        else:
            breaker.current_spend = current_spend
        
        # Check if breaker should trip
        if not breaker.is_open and breaker.current_spend > breaker.threshold:
            breaker.is_open = True
            breaker.opened_at = datetime.utcnow()
            breaker.failure_count += 1
            
            logger.warning(f"Circuit breaker tripped for {service}. Spend: ${current_spend:.2f}, Threshold: ${breaker.threshold:.2f}")
            
            # Execute circuit breaker action
            self._execute_circuit_breaker_action(breaker)
            
            return breaker
        
        # Check if breaker should reset
        if breaker.is_open and breaker.opened_at:
            reset_time = breaker.opened_at + timedelta(hours=1)
            if datetime.utcnow() > reset_time:
                breaker.is_open = False
                breaker.last_reset = datetime.utcnow()
                logger.info(f"Circuit breaker reset for {service}")
        
        return None
    
    def handle_cost_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming cost event from EventBridge
        
        Args:
            event: EventBridge event
            
        Returns:
            Response dictionary
        """
        logger.info(f"Handling cost event: {event.get('detail-type', 'Unknown')}")
        
        detail = event.get('detail', {})
        source = event.get('source', '')
        
        response = {
            'status': 'success',
            'actions_taken': []
        }
        
        try:
            # Route to appropriate handler
            if source == 'aws.ec2':
                response['actions_taken'].extend(
                    self._handle_ec2_event(detail)
                )
            elif source == 'aws.autoscaling':
                response['actions_taken'].extend(
                    self._handle_autoscaling_event(detail)
                )
            elif event.get('detail-type') == 'Scheduled Event':
                response['actions_taken'].extend(
                    self._handle_scheduled_check()
                )
            
            # Run custom event handlers
            event_type = event.get('detail-type', 'unknown')
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    handler_result = handler(event)
                    response['actions_taken'].append(handler_result)
                    
        except Exception as e:
            logger.error(f"Error handling cost event: {e}")
            response['status'] = 'error'
            response['error'] = str(e)
        
        return response
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register custom event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_current_costs(self, granularity: str = 'HOURLY') -> Dict[str, float]:
        """Get current costs from Cost Explorer"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1) if granularity == 'HOURLY' else end_time - timedelta(days=1)
        
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_time.strftime('%Y-%m-%d'),
                'End': end_time.strftime('%Y-%m-%d')
            },
            Granularity=granularity,
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        )
        
        costs = {}
        for result in response['ResultsByTime']:
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                costs[service] = costs.get(service, 0) + cost
        
        return costs
    
    def _create_scheduled_rule(self, 
                             rule_name: str,
                             schedule: str,
                             description: str,
                             target_lambda: str) -> str:
        """Create scheduled EventBridge rule"""
        try:
            response = self.eventbridge.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule,
                State='ENABLED',
                Description=description
            )
            
            # Add Lambda target
            self.eventbridge.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': target_lambda,
                        'Input': json.dumps({
                            'rule_name': rule_name,
                            'schedule': schedule
                        })
                    }
                ]
            )
            
            return response['RuleArn']
            
        except Exception as e:
            logger.error(f"Error creating scheduled rule {rule_name}: {e}")
            raise
    
    def _create_event_pattern_rule(self,
                                 rule_name: str,
                                 event_pattern: Dict[str, Any],
                                 description: str,
                                 target_lambda: str) -> str:
        """Create event pattern EventBridge rule"""
        try:
            response = self.eventbridge.put_rule(
                Name=rule_name,
                EventPattern=json.dumps(event_pattern),
                State='ENABLED',
                Description=description
            )
            
            # Add Lambda target
            self.eventbridge.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': target_lambda
                    }
                ]
            )
            
            return response['RuleArn']
            
        except Exception as e:
            logger.error(f"Error creating event pattern rule {rule_name}: {e}")
            raise
    
    def _get_or_create_lambda_function(self, function_name: str) -> str:
        """Get Lambda function ARN or create placeholder"""
        # In production, would create actual Lambda functions
        # For now, return a placeholder ARN
        return f"arn:aws:lambda:us-east-1:123456789012:function:{function_name}"
    
    def _execute_threshold_action(self, event: CostEvent):
        """Execute action for threshold breach"""
        if not event.threshold_breached:
            return
        
        threshold = event.threshold_breached
        action = threshold.action
        
        logger.info(f"Executing {action.value} for threshold breach: {threshold.threshold_id}")
        
        # Record action
        self.action_history.append({
            'timestamp': datetime.utcnow(),
            'action': action.value,
            'threshold_id': threshold.threshold_id,
            'event_id': event.event_id,
            'cost': event.current_cost
        })
        
        if action == ControlAction.ALERT:
            self._send_alert(event, threshold)
            
        elif action == ControlAction.THROTTLE:
            self._throttle_service(event.service or 'unknown')
            
        elif action == ControlAction.SHUTDOWN:
            self._shutdown_non_critical_resources(event.service)
            
        elif action == ControlAction.SCALE_DOWN:
            self._scale_down_service(event.service)
            
        elif action == ControlAction.REQUIRE_APPROVAL:
            self._create_approval_request(event)
            
        elif action == ControlAction.EMERGENCY_STOP:
            self._emergency_stop_all()
    
    def _execute_circuit_breaker_action(self, breaker: CircuitBreaker):
        """Execute action when circuit breaker trips"""
        logger.warning(f"Executing circuit breaker action for {breaker.service}")
        
        # Send alert
        self._send_circuit_breaker_alert(breaker)
        
        # Take protective action based on service
        if breaker.service in ['Amazon EC2', 'EC2-Instances']:
            self._limit_ec2_launches(breaker.service)
        elif breaker.service in ['Amazon RDS', 'RDS']:
            self._prevent_rds_scaling(breaker.service)
        else:
            self._generic_service_protection(breaker.service)
    
    def _send_alert(self, event: CostEvent, threshold: CostThreshold):
        """Send alert for threshold breach"""
        message = {
            'default': f"Cost threshold breached for {threshold.threshold_id}",
            'email': f"""
Cost Threshold Alert

Threshold: {threshold.threshold_id}
Type: {threshold.threshold_type.value}
Limit: ${threshold.value:.2f}
Current: ${event.current_cost:.2f}
Exceeded by: ${event.current_cost - threshold.value:.2f} ({event.metadata.get('percentage_over', 0):.1f}%)

Service: {event.service or 'Total'}
Timestamp: {event.timestamp}

Action Required: {threshold.action.value}
""",
            'sms': f"COST ALERT: {threshold.threshold_id} exceeded by {event.metadata.get('percentage_over', 0):.0f}%"
        }
        
        for topic_arn in threshold.notification_targets:
            try:
                self.sns_client.publish(
                    TopicArn=topic_arn,
                    Message=json.dumps(message),
                    MessageStructure='json',
                    Subject=f"Cost Alert: {threshold.threshold_id}"
                )
                logger.info(f"Alert sent to {topic_arn}")
            except Exception as e:
                logger.error(f"Failed to send alert to {topic_arn}: {e}")
    
    def _throttle_service(self, service: str):
        """Implement service throttling"""
        logger.info(f"Throttling service: {service}")
        
        # Service-specific throttling logic
        if service == 'Amazon EC2':
            # Reduce Auto Scaling max capacity
            self._reduce_asg_capacity(reduction_percent=50)
        elif service == 'AWS Lambda':
            # Set concurrent execution limits
            self._set_lambda_concurrency_limit(limit=100)
        
    def _shutdown_non_critical_resources(self, service: Optional[str]):
        """Shutdown non-critical resources"""
        logger.warning(f"Shutting down non-critical resources for service: {service or 'all'}")
        
        # Implementation would identify and stop non-critical resources
        # based on tags like Environment=dev, Critical=false
        
    def _scale_down_service(self, service: Optional[str]):
        """Scale down service capacity"""
        logger.info(f"Scaling down service: {service or 'all'}")
        
        # Implementation would reduce capacity while maintaining minimum service levels
        
    def _create_approval_request(self, event: CostEvent):
        """Create approval request for manual intervention"""
        logger.info(f"Creating approval request for event: {event.event_id}")
        
        # Implementation would create ticket in ticketing system
        # or send to approval workflow
        
    def _emergency_stop_all(self):
        """Emergency stop all non-critical services"""
        logger.critical("EMERGENCY STOP: Halting all non-critical services")
        
        # Implementation would stop all services tagged as non-critical
        # This is the nuclear option for runaway costs
        
    def _send_circuit_breaker_alert(self, breaker: CircuitBreaker):
        """Send alert when circuit breaker trips"""
        message = f"""
Circuit Breaker Tripped!

Service: {breaker.service}
Current Spend: ${breaker.current_spend:.2f}
Threshold: ${breaker.threshold:.2f}
Failure Count: {breaker.failure_count}
Opened At: {breaker.opened_at}

Automatic cost control measures have been activated.
"""
        
        # Send to all configured SNS topics
        for threshold in self.thresholds.values():
            if threshold.notification_targets:
                for topic_arn in threshold.notification_targets:
                    try:
                        self.sns_client.publish(
                            TopicArn=topic_arn,
                            Message=message,
                            Subject=f"Circuit Breaker: {breaker.service}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send circuit breaker alert: {e}")
    
    def _get_service_threshold(self, service: str) -> float:
        """Get threshold for service circuit breaker"""
        # Look for service-specific threshold
        for threshold in self.thresholds.values():
            if threshold.threshold_type == ThresholdType.SERVICE and threshold.target == service:
                return threshold.value
        
        # Default thresholds by service
        defaults = {
            'Amazon EC2': 10000.0,
            'Amazon RDS': 5000.0,
            'AWS Lambda': 1000.0,
            'Amazon S3': 2000.0
        }
        
        return defaults.get(service, 5000.0)
    
    def _handle_ec2_event(self, detail: Dict[str, Any]) -> List[str]:
        """Handle EC2 state change events"""
        actions = []
        
        instance_id = detail.get('instance-id')
        state = detail.get('state')
        
        if state == 'running':
            # Check if this instance should be running
            if self._is_unauthorized_launch(instance_id):
                actions.append(f"Detected unauthorized EC2 launch: {instance_id}")
                # Could automatically stop if circuit breaker is open
                
        return actions
    
    def _handle_autoscaling_event(self, detail: Dict[str, Any]) -> List[str]:
        """Handle Auto Scaling events"""
        actions = []
        
        # Check if scaling is happening during cost control
        if any(cb.is_open for cb in self.circuit_breakers.values()):
            actions.append("Auto Scaling event during active cost control")
            # Could prevent or limit scaling
            
        return actions
    
    def _handle_scheduled_check(self) -> List[str]:
        """Handle scheduled cost checks"""
        actions = []
        
        # Get current costs
        current_costs = self.get_current_costs()
        
        # Check thresholds
        events = self.check_thresholds(current_costs)
        
        for event in events:
            actions.append(f"Threshold breach: {event.threshold_breached.threshold_id}")
        
        # Update circuit breakers
        for service, cost in current_costs.items():
            breaker = self.update_circuit_breaker(service, cost)
            if breaker:
                actions.append(f"Circuit breaker tripped: {service}")
        
        return actions
    
    def _is_unauthorized_launch(self, instance_id: str) -> bool:
        """Check if instance launch is unauthorized"""
        # Implementation would check against approved instance lists,
        # time windows, budgets, etc.
        return False
    
    def _reduce_asg_capacity(self, reduction_percent: int):
        """Reduce Auto Scaling Group capacity"""
        # Implementation would reduce ASG max capacity
        pass
    
    def _set_lambda_concurrency_limit(self, limit: int):
        """Set Lambda concurrent execution limit"""
        # Implementation would set account-level concurrent execution limit
        pass
    
    def _limit_ec2_launches(self, service: str):
        """Limit EC2 instance launches"""
        # Implementation could:
        # - Modify IAM policies
        # - Set Service Control Policies
        # - Update launch templates
        pass
    
    def _prevent_rds_scaling(self, service: str):
        """Prevent RDS instance scaling"""
        # Implementation would prevent RDS modifications
        pass
    
    def _generic_service_protection(self, service: str):
        """Generic protection for any service"""
        # Implementation would apply service-specific protections
        pass
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export controller metrics"""
        return {
            'thresholds_configured': len(self.thresholds),
            'thresholds_enabled': sum(1 for t in self.thresholds.values() if t.enabled),
            'circuit_breakers_active': len(self.circuit_breakers),
            'circuit_breakers_open': sum(1 for cb in self.circuit_breakers.values() if cb.is_open),
            'actions_taken': len(self.action_history),
            'last_action': self.action_history[-1] if self.action_history else None
        }
    
    def create_lambda_handlers(self, output_dir: str = 'lambda_functions'):
        """Generate Lambda function code for event handlers"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Cost monitor handler
        cost_monitor_code = '''import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """Monitor costs and check thresholds"""
    ce_client = boto3.client('ce')
    sns_client = boto3.client('sns')
    
    # Get current costs
    # Check thresholds
    # Send alerts if needed
    
    return {
        'statusCode': 200,
        'body': json.dumps('Cost check completed')
    }
'''
        
        with open(os.path.join(output_dir, 'cost_monitor.py'), 'w') as f:
            f.write(cost_monitor_code)
        
        # Additional handler templates would be created here
        
        logger.info(f"Lambda handler templates created in {output_dir}")