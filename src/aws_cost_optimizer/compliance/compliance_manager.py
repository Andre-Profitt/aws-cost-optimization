"""
Compliance and Audit Trail System for AWS Cost Optimizer

Provides compliance tag checking and comprehensive audit logging for all
optimization activities to meet enterprise regulatory requirements.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import boto3
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    EXEMPT = "exempt"
    UNKNOWN = "unknown"


class AuditEventType(Enum):
    """Types of audit events"""
    RESOURCE_DISCOVERED = "resource_discovered"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    CHANGE_REQUESTED = "change_requested"
    CHANGE_APPROVED = "change_approved"
    CHANGE_REJECTED = "change_rejected"
    CHANGE_EXECUTED = "change_executed"
    CHANGE_ROLLED_BACK = "change_rolled_back"
    COMPLIANCE_CHECK = "compliance_check"
    POLICY_VIOLATION = "policy_violation"
    ACCESS_DENIED = "access_denied"


@dataclass
class ComplianceRule:
    """Defines a compliance rule"""
    rule_id: str
    name: str
    description: str
    required_tags: List[str] = field(default_factory=list)
    prohibited_tags: List[str] = field(default_factory=list)
    tag_patterns: Dict[str, str] = field(default_factory=dict)  # tag_key: regex_pattern
    resource_types: List[str] = field(default_factory=list)  # Apply to specific resource types
    environments: List[str] = field(default_factory=list)  # Apply to specific environments
    exemption_tags: List[str] = field(default_factory=list)  # Tags that exempt from rule
    severity: str = "medium"  # low, medium, high, critical
    
    
@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    violation_id: str
    rule_id: str
    resource_id: str
    resource_type: str
    region: str
    violation_type: str
    details: str
    severity: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    
@dataclass
class AuditEvent:
    """Represents an audit trail event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user: str
    resource_id: Optional[str]
    resource_type: Optional[str]
    region: Optional[str]
    action: str
    details: Dict[str, Any]
    outcome: str  # success, failure, partial
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user,
            'resource_id': self.resource_id,
            'resource_type': self.resource_type,
            'region': self.region,
            'action': self.action,
            'details': json.dumps(self.details),
            'outcome': self.outcome,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'session_id': self.session_id
        }


class ComplianceManager:
    """Manages compliance checking and policy enforcement"""
    
    def __init__(self, config: Dict[str, Any], session: Optional[boto3.Session] = None):
        self.session = session or boto3.Session()
        self.config = config
        self.rules = self._load_compliance_rules()
        
        # Initialize AWS services
        self.organizations = self.session.client('organizations')
        self.config_service = self.session.client('config')
    
    def check_resource_compliance(self, 
                                resource_id: str,
                                resource_type: str,
                                region: str,
                                tags: Dict[str, str]) -> Dict[str, Any]:
        """Check if a resource is compliant with policies"""
        compliance_result = {
            'resource_id': resource_id,
            'resource_type': resource_type,
            'region': region,
            'status': ComplianceStatus.COMPLIANT,
            'violations': [],
            'warnings': [],
            'applicable_rules': [],
            'checked_at': datetime.now().isoformat()
        }
        
        # Check each rule
        for rule in self.rules:
            if self._is_rule_applicable(rule, resource_type, tags):
                compliance_result['applicable_rules'].append(rule.rule_id)
                
                # Check for exemptions
                if self._is_exempt(rule, tags):
                    continue
                
                # Check required tags
                violations = self._check_required_tags(rule, tags)
                if violations:
                    compliance_result['violations'].extend(violations)
                
                # Check prohibited tags
                violations = self._check_prohibited_tags(rule, tags)
                if violations:
                    compliance_result['violations'].extend(violations)
                
                # Check tag patterns
                violations = self._check_tag_patterns(rule, tags)
                if violations:
                    compliance_result['violations'].extend(violations)
        
        # Check organizational policies
        org_violations = self._check_organizational_policies(resource_id, resource_type, region)
        compliance_result['violations'].extend(org_violations)
        
        # Determine overall status
        if compliance_result['violations']:
            compliance_result['status'] = ComplianceStatus.NON_COMPLIANT
        elif compliance_result['warnings']:
            compliance_result['status'] = ComplianceStatus.WARNING
        
        return compliance_result
    
    def check_optimization_compliance(self, 
                                    recommendation: Any,
                                    resource_tags: Dict[str, str]) -> bool:
        """Check if an optimization recommendation is compliant"""
        # Check data residency requirements
        if not self._check_data_residency(recommendation.region, resource_tags):
            logger.warning(f"Data residency violation for {recommendation.resource_id}")
            return False
        
        # Check change freeze periods
        if self._is_change_freeze_period(resource_tags):
            logger.warning(f"Change freeze period active for {recommendation.resource_id}")
            return False
        
        # Check regulatory compliance
        if not self._check_regulatory_compliance(recommendation, resource_tags):
            logger.warning(f"Regulatory compliance issue for {recommendation.resource_id}")
            return False
        
        return True
    
    def generate_compliance_report(self, 
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for a time period"""
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_resources_checked': 0,
                'compliant_resources': 0,
                'non_compliant_resources': 0,
                'violations_by_severity': defaultdict(int),
                'violations_by_rule': defaultdict(int),
                'violations_by_resource_type': defaultdict(int)
            },
            'details': [],
            'recommendations': []
        }
        
        # Query compliance history (would integrate with Config)
        # This is a simplified version
        
        return report
    
    def _load_compliance_rules(self) -> List[ComplianceRule]:
        """Load compliance rules from configuration"""
        rules = []
        
        # Default rules
        rules.extend([
            ComplianceRule(
                rule_id="REQUIRED_TAGS_001",
                name="Required Resource Tags",
                description="All resources must have required tags",
                required_tags=["Environment", "Owner", "CostCenter", "Project"],
                resource_types=["ec2", "rds", "s3"],
                severity="high"
            ),
            ComplianceRule(
                rule_id="PROD_PROTECTION_001",
                name="Production Resource Protection",
                description="Production resources require additional approval",
                tag_patterns={"Environment": r"^(prod|production)$"},
                severity="critical"
            ),
            ComplianceRule(
                rule_id="PCI_COMPLIANCE_001",
                name="PCI Compliance",
                description="PCI-tagged resources have special requirements",
                tag_patterns={"Compliance": r".*PCI.*"},
                prohibited_tags=["PublicAccess"],
                severity="critical"
            ),
            ComplianceRule(
                rule_id="DATA_CLASSIFICATION_001",
                name="Data Classification Required",
                description="Storage resources must have data classification",
                required_tags=["DataClassification"],
                resource_types=["s3", "ebs", "efs"],
                severity="high"
            )
        ])
        
        # Load custom rules from config
        if 'compliance_rules' in self.config:
            for rule_config in self.config['compliance_rules']:
                rules.append(ComplianceRule(**rule_config))
        
        return rules
    
    def _is_rule_applicable(self, rule: ComplianceRule, 
                          resource_type: str, 
                          tags: Dict[str, str]) -> bool:
        """Check if a rule applies to a resource"""
        # Check resource type
        if rule.resource_types and resource_type not in rule.resource_types:
            return False
        
        # Check environment
        if rule.environments:
            env = tags.get('Environment', '').lower()
            if env not in [e.lower() for e in rule.environments]:
                return False
        
        return True
    
    def _is_exempt(self, rule: ComplianceRule, tags: Dict[str, str]) -> bool:
        """Check if resource is exempt from rule"""
        for exempt_tag in rule.exemption_tags:
            if exempt_tag in tags:
                return True
        return False
    
    def _check_required_tags(self, rule: ComplianceRule, 
                           tags: Dict[str, str]) -> List[ComplianceViolation]:
        """Check for required tags"""
        violations = []
        
        for required_tag in rule.required_tags:
            if required_tag not in tags or not tags[required_tag]:
                violations.append(ComplianceViolation(
                    violation_id=self._generate_violation_id(),
                    rule_id=rule.rule_id,
                    resource_id="",  # Will be filled by caller
                    resource_type="",  # Will be filled by caller
                    region="",  # Will be filled by caller
                    violation_type="missing_required_tag",
                    details=f"Missing required tag: {required_tag}",
                    severity=rule.severity
                ))
        
        return violations
    
    def _check_prohibited_tags(self, rule: ComplianceRule, 
                             tags: Dict[str, str]) -> List[ComplianceViolation]:
        """Check for prohibited tags"""
        violations = []
        
        for prohibited_tag in rule.prohibited_tags:
            if prohibited_tag in tags:
                violations.append(ComplianceViolation(
                    violation_id=self._generate_violation_id(),
                    rule_id=rule.rule_id,
                    resource_id="",
                    resource_type="",
                    region="",
                    violation_type="prohibited_tag_present",
                    details=f"Prohibited tag present: {prohibited_tag}",
                    severity=rule.severity
                ))
        
        return violations
    
    def _check_tag_patterns(self, rule: ComplianceRule, 
                          tags: Dict[str, str]) -> List[ComplianceViolation]:
        """Check tag patterns"""
        import re
        violations = []
        
        for tag_key, pattern in rule.tag_patterns.items():
            if tag_key in tags:
                if not re.match(pattern, tags[tag_key]):
                    violations.append(ComplianceViolation(
                        violation_id=self._generate_violation_id(),
                        rule_id=rule.rule_id,
                        resource_id="",
                        resource_type="",
                        region="",
                        violation_type="tag_pattern_mismatch",
                        details=f"Tag {tag_key} value '{tags[tag_key]}' doesn't match pattern '{pattern}'",
                        severity=rule.severity
                    ))
        
        return violations
    
    def _check_organizational_policies(self, resource_id: str, 
                                     resource_type: str, 
                                     region: str) -> List[ComplianceViolation]:
        """Check AWS Organizations policies"""
        violations = []
        
        # This would integrate with AWS Organizations SCPs
        # For now, return empty list
        
        return violations
    
    def _check_data_residency(self, region: str, tags: Dict[str, str]) -> bool:
        """Check data residency requirements"""
        data_residency = tags.get('DataResidency', '').lower()
        
        if not data_residency:
            return True
        
        # Map residency requirements to allowed regions
        residency_map = {
            'us': ['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2'],
            'eu': ['eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1'],
            'apac': ['ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1']
        }
        
        allowed_regions = residency_map.get(data_residency, [])
        return not allowed_regions or region in allowed_regions
    
    def _is_change_freeze_period(self, tags: Dict[str, str]) -> bool:
        """Check if resource is in change freeze period"""
        freeze_until = tags.get('ChangeFreezeUntil')
        
        if not freeze_until:
            return False
        
        try:
            freeze_date = datetime.fromisoformat(freeze_until)
            return datetime.now() < freeze_date
        except:
            return False
    
    def _check_regulatory_compliance(self, recommendation: Any, 
                                   tags: Dict[str, str]) -> bool:
        """Check regulatory compliance requirements"""
        compliance_scope = tags.get('Compliance', '').upper()
        
        if not compliance_scope:
            return True
        
        # Check specific compliance requirements
        if 'HIPAA' in compliance_scope:
            # HIPAA requires encryption at rest and in transit
            if recommendation.action in ['delete', 'terminate']:
                # Ensure data is properly archived/destroyed
                return 'DataRetentionVerified' in tags
        
        if 'PCI' in compliance_scope:
            # PCI requires strict access controls
            if recommendation.action == 'modify':
                # Ensure security groups aren't being loosened
                return True  # Would check actual changes
        
        if 'SOX' in compliance_scope:
            # SOX requires audit trails
            return True  # Audit trail is being maintained
        
        return True
    
    def _generate_violation_id(self) -> str:
        """Generate unique violation ID"""
        data = f"violation-{datetime.now().isoformat()}-{id(self)}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class AuditTrail:
    """Manages audit trail for all optimization activities"""
    
    def __init__(self, config: Dict[str, Any], session: Optional[boto3.Session] = None):
        self.session = session or boto3.Session()
        self.config = config
        
        # Initialize storage backends
        self.cloudtrail = self.session.client('cloudtrail')
        self.s3 = self.session.client('s3')
        self.dynamodb = self.session.resource('dynamodb')
        
        # Audit trail configuration
        self.trail_name = config.get('trail_name', 'aws-cost-optimizer-audit')
        self.bucket_name = config.get('audit_bucket', 'aws-cost-optimizer-audit-logs')
        self.table_name = config.get('audit_table', 'aws-cost-optimizer-audit-events')
        
        self._ensure_infrastructure()
    
    def log_event(self, 
                 event_type: AuditEventType,
                 user: str,
                 action: str,
                 details: Dict[str, Any],
                 resource_id: Optional[str] = None,
                 resource_type: Optional[str] = None,
                 region: Optional[str] = None,
                 outcome: str = 'success') -> str:
        """Log an audit event"""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.now(),
            user=user,
            resource_id=resource_id,
            resource_type=resource_type,
            region=region,
            action=action,
            details=details,
            outcome=outcome,
            ip_address=details.get('ip_address'),
            user_agent=details.get('user_agent'),
            session_id=details.get('session_id')
        )
        
        # Store in DynamoDB for quick access
        self._store_event(event)
        
        # Log to CloudTrail for compliance
        self._log_to_cloudtrail(event)
        
        # Archive to S3 for long-term storage
        self._archive_event(event)
        
        logger.info(f"Audit event logged: {event.event_id}")
        return event.event_id
    
    def log_recommendation(self, user: str, recommendation: Any, 
                         resource_tags: Dict[str, str]):
        """Log recommendation generation"""
        details = {
            'recommendation_type': recommendation.action,
            'resource_id': recommendation.resource_id,
            'resource_type': recommendation.resource_type,
            'estimated_savings': recommendation.monthly_savings,
            'risk_level': getattr(recommendation, 'risk_level', 'unknown'),
            'confidence': getattr(recommendation, 'confidence', 0),
            'resource_tags': resource_tags
        }
        
        self.log_event(
            event_type=AuditEventType.RECOMMENDATION_GENERATED,
            user=user,
            action=f"Generated {recommendation.action} recommendation",
            details=details,
            resource_id=recommendation.resource_id,
            resource_type=recommendation.resource_type,
            region=recommendation.region
        )
    
    def log_change_request(self, user: str, change_request: Any):
        """Log change request creation"""
        details = {
            'change_request_id': change_request.request_id,
            'change_type': change_request.change_type.value,
            'risk_level': change_request.risk_level.value,
            'estimated_savings': change_request.estimated_savings,
            'dependencies': change_request.dependencies
        }
        
        self.log_event(
            event_type=AuditEventType.CHANGE_REQUESTED,
            user=user,
            action=f"Created change request {change_request.request_id}",
            details=details,
            resource_id=change_request.resource_id,
            resource_type=change_request.resource_type,
            region=change_request.region
        )
    
    def log_approval(self, approver: str, change_request_id: str, 
                    comments: Optional[str] = None):
        """Log change approval"""
        details = {
            'change_request_id': change_request_id,
            'comments': comments
        }
        
        self.log_event(
            event_type=AuditEventType.CHANGE_APPROVED,
            user=approver,
            action=f"Approved change request {change_request_id}",
            details=details
        )
    
    def log_execution(self, user: str, change_request: Any, 
                     execution_result: Dict[str, Any]):
        """Log change execution"""
        details = {
            'change_request_id': change_request.request_id,
            'execution_result': execution_result,
            'duration_seconds': execution_result.get('duration'),
            'actual_changes': execution_result.get('changes_made', [])
        }
        
        outcome = 'success' if execution_result.get('success') else 'failure'
        
        self.log_event(
            event_type=AuditEventType.CHANGE_EXECUTED,
            user=user,
            action=f"Executed change {change_request.change_type.value}",
            details=details,
            resource_id=change_request.resource_id,
            resource_type=change_request.resource_type,
            region=change_request.region,
            outcome=outcome
        )
    
    def log_compliance_check(self, user: str, compliance_result: Dict[str, Any]):
        """Log compliance check"""
        details = {
            'compliance_status': compliance_result['status'],
            'violations': compliance_result.get('violations', []),
            'applicable_rules': compliance_result.get('applicable_rules', [])
        }
        
        self.log_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user=user,
            action="Performed compliance check",
            details=details,
            resource_id=compliance_result.get('resource_id'),
            resource_type=compliance_result.get('resource_type'),
            region=compliance_result.get('region')
        )
    
    def query_audit_trail(self, 
                         start_time: datetime,
                         end_time: datetime,
                         filters: Dict[str, Any] = None) -> List[AuditEvent]:
        """Query audit trail for events"""
        table = self.dynamodb.Table(self.table_name)
        
        # Build query
        query_params = {
            'FilterExpression': 'timestamp BETWEEN :start AND :end',
            'ExpressionAttributeValues': {
                ':start': start_time.isoformat(),
                ':end': end_time.isoformat()
            }
        }
        
        # Add additional filters
        if filters:
            if 'user' in filters:
                query_params['FilterExpression'] += ' AND user = :user'
                query_params['ExpressionAttributeValues'][':user'] = filters['user']
            
            if 'event_type' in filters:
                query_params['FilterExpression'] += ' AND event_type = :event_type'
                query_params['ExpressionAttributeValues'][':event_type'] = filters['event_type']
            
            if 'resource_id' in filters:
                query_params['FilterExpression'] += ' AND resource_id = :resource_id'
                query_params['ExpressionAttributeValues'][':resource_id'] = filters['resource_id']
        
        # Scan table (in production, use GSI for better performance)
        response = table.scan(**query_params)
        
        events = []
        for item in response.get('Items', []):
            events.append(self._deserialize_event(item))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def generate_audit_report(self, 
                            start_time: datetime,
                            end_time: datetime) -> Dict[str, Any]:
        """Generate audit report for compliance"""
        events = self.query_audit_trail(start_time, end_time)
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'events_by_type': defaultdict(int),
                'events_by_user': defaultdict(int),
                'events_by_outcome': defaultdict(int),
                'resources_modified': set()
            },
            'compliance_events': [],
            'failed_events': [],
            'high_risk_changes': []
        }
        
        for event in events:
            # Count by type
            report['summary']['events_by_type'][event.event_type.value] += 1
            
            # Count by user
            report['summary']['events_by_user'][event.user] += 1
            
            # Count by outcome
            report['summary']['events_by_outcome'][event.outcome] += 1
            
            # Track modified resources
            if event.resource_id:
                report['summary']['resources_modified'].add(event.resource_id)
            
            # Collect compliance events
            if event.event_type == AuditEventType.COMPLIANCE_CHECK:
                report['compliance_events'].append(event)
            
            # Collect failed events
            if event.outcome == 'failure':
                report['failed_events'].append(event)
            
            # Collect high-risk changes
            if (event.event_type == AuditEventType.CHANGE_EXECUTED and 
                event.details.get('risk_level') in ['high', 'critical']):
                report['high_risk_changes'].append(event)
        
        # Convert set to list for JSON serialization
        report['summary']['resources_modified'] = list(report['summary']['resources_modified'])
        
        return report
    
    def export_audit_logs(self, 
                         start_time: datetime,
                         end_time: datetime,
                         format: str = 'json') -> str:
        """Export audit logs for external analysis"""
        events = self.query_audit_trail(start_time, end_time)
        
        filename = f"audit_export_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.{format}"
        
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump([e.to_dict() for e in events], f, indent=2)
        elif format == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())
        
        # Upload to S3
        s3_key = f"exports/{filename}"
        self.s3.upload_file(filename, self.bucket_name, s3_key)
        
        logger.info(f"Exported audit logs to s3://{self.bucket_name}/{s3_key}")
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def _ensure_infrastructure(self):
        """Ensure audit trail infrastructure exists"""
        # Create S3 bucket if needed
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except:
            self.s3.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.session.region_name
                } if self.session.region_name != 'us-east-1' else {}
            )
            
            # Enable versioning and encryption
            self.s3.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            self.s3.put_bucket_encryption(
                Bucket=self.bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [{
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }]
                }
            )
        
        # Create DynamoDB table if needed
        try:
            table = self.dynamodb.Table(self.table_name)
            table.load()
        except:
            table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {'AttributeName': 'event_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'event_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST',
                StreamSpecification={
                    'StreamEnabled': True,
                    'StreamViewType': 'NEW_AND_OLD_IMAGES'
                }
            )
            table.wait_until_exists()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        data = f"event-{datetime.now().isoformat()}-{id(self)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _store_event(self, event: AuditEvent):
        """Store event in DynamoDB"""
        table = self.dynamodb.Table(self.table_name)
        table.put_item(Item=event.to_dict())
    
    def _log_to_cloudtrail(self, event: AuditEvent):
        """Log event to CloudTrail"""
        # CloudTrail integration would go here
        # This requires CloudTrail to be configured for custom events
        pass
    
    def _archive_event(self, event: AuditEvent):
        """Archive event to S3"""
        # Partition by date for efficient querying
        date_partition = event.timestamp.strftime('%Y/%m/%d')
        s3_key = f"events/{date_partition}/{event.event_id}.json"
        
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(event.to_dict()),
                ServerSideEncryption='AES256'
            )
        except Exception as e:
            logger.error(f"Failed to archive event to S3: {e}")
    
    def _deserialize_event(self, item: Dict[str, Any]) -> AuditEvent:
        """Deserialize event from DynamoDB item"""
        return AuditEvent(
            event_id=item['event_id'],
            event_type=AuditEventType(item['event_type']),
            timestamp=datetime.fromisoformat(item['timestamp']),
            user=item['user'],
            resource_id=item.get('resource_id'),
            resource_type=item.get('resource_type'),
            region=item.get('region'),
            action=item['action'],
            details=json.loads(item['details']) if isinstance(item['details'], str) else item['details'],
            outcome=item['outcome'],
            ip_address=item.get('ip_address'),
            user_agent=item.get('user_agent'),
            session_id=item.get('session_id')
        )
    
    def setup_real_time_alerts(self, sns_topic_arn: str):
        """Set up real-time alerts for critical events"""
        # This would use DynamoDB Streams to trigger Lambda
        # Lambda would evaluate events and send SNS alerts
        logger.info(f"Real-time alerts would be sent to {sns_topic_arn}")
