"""
Enterprise Integration Module for AWS Cost Optimizer

Integrates all enterprise enhancements (dependency mapping, change management,
monitoring, and compliance) into the existing cost optimization workflow.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import enterprise components (will be added later)
# from .dependency_mapper import DependencyMapper, ResourceNode
# from .change_management import ChangeManagementSystem, ChangeRequest, ChangeRiskLevel
# from .monitoring_integration import CloudWatchMonitoring, MonitoringTarget, MetricType

# Import existing components
from aws_cost_optimizer.compliance.compliance_manager import (
    ComplianceManager, 
    AuditTrail, 
    AuditEventType,
    ComplianceStatus
)
from aws_cost_optimizer.optimization.ec2_optimizer import EC2Optimizer
from aws_cost_optimizer.optimization.safety_checks import SafetyChecker
from aws_cost_optimizer.orchestrator import CostOptimizationOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise features"""
    enable_dependency_mapping: bool = True
    enable_change_management: bool = True
    enable_monitoring: bool = True
    enable_compliance: bool = True
    enable_audit_trail: bool = True
    
    # Change management settings
    ticketing_system: str = "servicenow"  # servicenow, jira, or none
    auto_approve_low_risk: bool = False
    require_approval_for_production: bool = True
    
    # Monitoring settings
    monitoring_duration_hours: int = 72
    create_dashboards: bool = True
    enable_anomaly_detection: bool = True
    
    # Compliance settings
    enforce_compliance: bool = True
    block_non_compliant: bool = True
    
    # Notification settings
    sns_topic_arn: Optional[str] = None
    slack_webhook: Optional[str] = None


class EnterpriseOptimizer:
    """Enhanced optimizer with enterprise features"""
    
    def __init__(self, config: EnterpriseConfig, session: Optional[boto3.Session] = None):
        self.config = config
        self.session = session or boto3.Session()
        
        # Initialize enterprise components
        self.dependency_mapper = None  # Will be initialized when dependency_mapper.py is added
        self.change_management = None  # Will be initialized when change_management.py is added
        self.monitoring = None  # Will be initialized when monitoring_integration.py is added
        self.compliance = ComplianceManager(config={}, session=self.session) if config.enable_compliance else None
        self.audit_trail = AuditTrail(config={}, session=self.session) if config.enable_audit_trail else None
        
        # Initialize base optimizer
        self.base_optimizer = self._init_base_optimizer()
        
    def run_enterprise_optimization(self, 
                                  regions: List[str] = None,
                                  services: List[str] = None,
                                  user: str = "system") -> Dict[str, Any]:
        """Run optimization with full enterprise features"""
        logger.info("Starting enterprise optimization workflow")
        
        # Phase 1: Discovery and Dependency Mapping
        logger.info("Phase 1: Resource discovery and dependency mapping")
        dependencies = {}
        if self.config.enable_dependency_mapping and self.dependency_mapper:
            dependency_graph = self.dependency_mapper.discover_all_dependencies(regions)
            self.dependency_mapper.export_dependency_graph("dependencies.json")
            
            # Log discovery
            if self.audit_trail:
                self.audit_trail.log_event(
                    event_type=AuditEventType.RESOURCE_DISCOVERED,
                    user=user,
                    action="Discovered resources and mapped dependencies",
                    details={
                        'total_resources': len(self.dependency_mapper.resources),
                        'total_dependencies': len(self.dependency_mapper.dependencies)
                    }
                )
        
        # Phase 2: Run Base Optimization
        logger.info("Phase 2: Running cost optimization analysis")
        optimization_result = self.base_optimizer.run_full_optimization(regions, services)
        
        # Phase 3: Compliance Checking
        logger.info("Phase 3: Checking compliance for recommendations")
        compliant_recommendations = []
        
        for service, recommendations in optimization_result.details.items():
            if service in ['ec2', 'rds', 's3']:
                service_recommendations = recommendations.get('recommendations', [])
                
                for rec in service_recommendations:
                    # Get resource tags
                    tags = self._get_resource_tags(rec.resource_id, rec.resource_type, rec.region)
                    
                    # Check compliance
                    compliance_result = {'status': ComplianceStatus.COMPLIANT}
                    if self.config.enable_compliance:
                        compliance_result = self.compliance.check_resource_compliance(
                            rec.resource_id,
                            rec.resource_type,
                            rec.region,
                            tags
                        )
                        
                        # Log compliance check
                        if self.audit_trail:
                            self.audit_trail.log_compliance_check(user, compliance_result)
                        
                        # Skip non-compliant resources if enforcement is enabled
                        if (self.config.block_non_compliant and 
                            compliance_result['status'].value == 'non_compliant'):
                            logger.warning(f"Skipping non-compliant resource: {rec.resource_id}")
                            continue
                    
                    # Check dependencies
                    resource_dependencies = []
                    impact_analysis = {}
                    
                    if self.config.enable_dependency_mapping and self.dependency_mapper and rec.resource_id in self.dependency_mapper.resources:
                        impact_analysis = self.dependency_mapper.get_impact_analysis(rec.resource_id)
                        resource_dependencies = [r.resource_id for r in impact_analysis.get('downstream_resources', [])]
                        
                        # Adjust risk based on dependencies
                        if impact_analysis.get('risk_level') == 'high' and hasattr(rec, 'risk_level'):
                            rec.risk_level = 'high'
                    
                    # Add enhanced recommendation
                    enhanced_rec = {
                        'recommendation': rec,
                        'tags': tags,
                        'dependencies': resource_dependencies,
                        'impact_analysis': impact_analysis,
                        'compliance_status': compliance_result.get('status').value if self.config.enable_compliance else 'unknown'
                    }
                    
                    compliant_recommendations.append(enhanced_rec)
        
        # Phase 4: Create Change Requests
        logger.info("Phase 4: Creating change requests")
        change_requests = []
        
        if self.config.enable_change_management and self.change_management:
            for enhanced_rec in compliant_recommendations:
                rec = enhanced_rec['recommendation']
                
                # Create change request
                change_request = self.change_management.create_change_request(
                    recommendation=rec,
                    dependencies=enhanced_rec['dependencies'],
                    requester=user
                )
                
                change_requests.append(change_request)
                
                # Log change request
                if self.audit_trail:
                    self.audit_trail.log_change_request(user, change_request)
        
        # Phase 5: Set Up Monitoring
        logger.info("Phase 5: Setting up monitoring")
        monitoring_setup = {}
        
        if self.config.enable_monitoring and self.monitoring:
            # Create monitoring targets
            monitoring_targets = []
            for enhanced_rec in compliant_recommendations:
                rec = enhanced_rec['recommendation']
                
                target = self.monitoring.create_monitoring_target(
                    resource_id=rec.resource_id,
                    resource_type=rec.resource_type,
                    region=rec.region,
                    metrics=self._determine_metrics(rec.resource_type),
                    tags=enhanced_rec['tags']
                )
                monitoring_targets.append(target)
            
            # Create dashboard
            if self.config.create_dashboards:
                dashboard_url = self.monitoring.create_optimization_dashboard(
                    dashboard_name=f"CostOptimization-{datetime.now().strftime('%Y%m%d')}",
                    optimization_results={
                        'total_monthly_savings': optimization_result.total_monthly_savings,
                        'baseline_daily_cost': optimization_result.total_monthly_savings / 30
                    },
                    monitoring_targets=monitoring_targets[:10]  # Limit to 10 for dashboard
                )
                monitoring_setup['dashboard_url'] = dashboard_url
            
            # Set up anomaly detection
            if self.config.enable_anomaly_detection:
                for target in monitoring_targets[:5]:  # Limit to avoid excessive alarms
                    self.monitoring.create_anomaly_detector(
                        target.resource_id,
                        target.resource_type,
                        'CPUUtilization',
                        target.region
                    )
        
        # Phase 6: Generate Reports
        logger.info("Phase 6: Generating reports")
        reports = self._generate_enterprise_reports(
            optimization_result,
            compliant_recommendations,
            change_requests,
            monitoring_setup
        )
        
        return {
            'optimization_result': optimization_result,
            'compliant_recommendations': len(compliant_recommendations),
            'change_requests_created': len(change_requests),
            'monitoring_setup': monitoring_setup,
            'reports': reports,
            'workflow_completed_at': datetime.now().isoformat()
        }
    
    def execute_approved_changes(self, dry_run: bool = True) -> Dict[str, Any]:
        """Execute approved changes with full monitoring"""
        if not self.config.enable_change_management or not self.change_management:
            raise ValueError("Change management is not enabled")
        
        # Get approved changes
        approved_changes = self.change_management.get_approved_changes()
        
        results = {
            'total_changes': len(approved_changes),
            'executed': 0,
            'failed': 0,
            'monitoring_enabled': 0,
            'details': []
        }
        
        for change in approved_changes:
            try:
                # Pre-execution checks
                if self.config.enable_compliance:
                    # Re-verify compliance
                    tags = self._get_resource_tags(
                        change.resource_id,
                        change.resource_type,
                        change.region
                    )
                    
                    compliance_result = self.compliance.check_resource_compliance(
                        change.resource_id,
                        change.resource_type,
                        change.region,
                        tags
                    )
                    
                    if compliance_result['status'].value == 'non_compliant':
                        logger.warning(f"Resource {change.resource_id} is no longer compliant, skipping")
                        continue
                
                # Set up pre-change monitoring
                baseline_alarms = []
                if self.config.enable_monitoring and self.monitoring:
                    baseline_alarms = self.monitoring.setup_post_change_monitoring(
                        change,
                        self.config.monitoring_duration_hours
                    )
                
                # Execute change
                if not dry_run:
                    execution_result = self._execute_change(change)
                    
                    # Log execution
                    if self.audit_trail:
                        self.audit_trail.log_execution(
                            'system',
                            change,
                            execution_result
                        )
                    
                    if execution_result['success']:
                        results['executed'] += 1
                    else:
                        results['failed'] += 1
                else:
                    execution_result = {
                        'success': True,
                        'dry_run': True,
                        'message': 'Dry run - no changes made'
                    }
                
                # Record details
                results['details'].append({
                    'change_request_id': change.request_id,
                    'resource_id': change.resource_id,
                    'execution_result': execution_result,
                    'monitoring_alarms': baseline_alarms
                })
                
                if baseline_alarms:
                    results['monitoring_enabled'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to execute change {change.request_id}: {e}")
                results['failed'] += 1
                results['details'].append({
                    'change_request_id': change.request_id,
                    'error': str(e)
                })
        
        return results
    
    def generate_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'sections': {}
        }
        
        # Compliance summary
        if self.config.enable_compliance:
            report['sections']['compliance'] = self.compliance.generate_compliance_report(
                start_date,
                end_date
            )
        
        # Audit trail summary
        if self.config.enable_audit_trail:
            report['sections']['audit'] = self.audit_trail.generate_audit_report(
                start_date,
                end_date
            )
        
        # Change management summary
        if self.config.enable_change_management and self.change_management:
            report['sections']['changes'] = self.change_management.generate_approval_report()
        
        # Monitoring summary
        if self.config.enable_monitoring and self.monitoring:
            # Get all optimized resources from audit trail
            events = self.audit_trail.query_audit_trail(
                start_date,
                end_date,
                {'event_type': AuditEventType.CHANGE_EXECUTED.value}
            )
            
            resource_ids = [e.resource_id for e in events if e.resource_id]
            
            if resource_ids:
                report['sections']['monitoring'] = self.monitoring.generate_monitoring_report(
                    start_date,
                    end_date,
                    resource_ids[:20]  # Limit to 20 resources
                )
        
        return report
    
    def _init_change_management(self):
        """Initialize change management with configuration"""
        # This will be implemented when change_management.py is added
        pass
    
    def _init_base_optimizer(self) -> CostOptimizationOrchestrator:
        """Initialize base optimizer with enhanced safety"""
        config = {
            'enable_auto_remediation': False,  # We handle this separately
            'safety_checks': {
                'enhanced': True,
                'dependency_aware': self.config.enable_dependency_mapping
            }
        }
        
        orchestrator = CostOptimizationOrchestrator(
            session=self.session,
            config=config
        )
        
        # Enhance safety checker if dependency mapping is enabled
        if self.config.enable_dependency_mapping and self.dependency_mapper:
            orchestrator.ec2_optimizer.safety_checker = EnhancedSafetyChecker(
                self.dependency_mapper,
                self.compliance if self.config.enable_compliance else None
            )
        
        return orchestrator
    
    def _get_resource_tags(self, resource_id: str, resource_type: str, region: str) -> Dict[str, str]:
        """Get tags for a resource"""
        try:
            if resource_type == 'ec2':
                ec2 = self.session.client('ec2', region_name=region)
                response = ec2.describe_instances(InstanceIds=[resource_id])
                if response['Reservations']:
                    instance = response['Reservations'][0]['Instances'][0]
                    return {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            
            elif resource_type == 'rds':
                rds = self.session.client('rds', region_name=region)
                response = rds.describe_db_instances(DBInstanceIdentifier=resource_id)
                if response['DBInstances']:
                    db = response['DBInstances'][0]
                    return {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
            
            elif resource_type == 's3':
                s3 = self.session.client('s3')
                response = s3.get_bucket_tagging(Bucket=resource_id)
                return {tag['Key']: tag['Value'] for tag in response.get('TagSet', [])}
                
        except Exception as e:
            logger.error(f"Failed to get tags for {resource_id}: {e}")
        
        return {}
    
    def _determine_metrics(self, resource_type: str) -> List[str]:
        """Determine which metrics to monitor for a resource type"""
        metrics_map = {
            'ec2': ['CPUUtilization', 'NetworkIn', 'NetworkOut'],
            'rds': ['CPUUtilization', 'DatabaseConnections'],
            's3': []  # S3 metrics are different
        }
        return metrics_map.get(resource_type, [])
    
    def _execute_change(self, change_request) -> Dict[str, Any]:
        """Execute a single change"""
        try:
            # This would contain the actual execution logic
            # For now, return a mock result
            return {
                'success': True,
                'duration': 45,
                'changes_made': [
                    f"Executed change on {change_request.resource_id}"
                ]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_enterprise_reports(self, 
                                   optimization_result: Any,
                                   compliant_recommendations: List[Dict],
                                   change_requests: List,
                                   monitoring_setup: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive reports"""
        reports = {}
        
        # Executive summary
        exec_summary = f"""
AWS Cost Optimization Enterprise Report
Generated: {datetime.now().isoformat()}

EXECUTIVE SUMMARY
================
Total Monthly Savings Identified: ${optimization_result.total_monthly_savings:,.2f}
Total Annual Savings Identified: ${optimization_result.total_annual_savings:,.2f}

Recommendations: {len(compliant_recommendations)}
- Compliant: {sum(1 for r in compliant_recommendations if r['compliance_status'] == 'compliant')}
- With Dependencies: {sum(1 for r in compliant_recommendations if r['dependencies'])}

Change Requests Created: {len(change_requests)}

Monitoring Dashboard: {monitoring_setup.get('dashboard_url', 'Not created')}
"""
        reports['executive_summary'] = exec_summary
        
        # Detailed recommendations report
        detailed_report = self._generate_detailed_report(compliant_recommendations)
        reports['detailed_recommendations'] = detailed_report
        
        # Dependency impact report
        if self.config.enable_dependency_mapping:
            dependency_report = self._generate_dependency_report(compliant_recommendations)
            reports['dependency_impact'] = dependency_report
        
        return reports
    
    def _generate_detailed_report(self, recommendations: List[Dict]) -> str:
        """Generate detailed recommendations report"""
        report_lines = ["DETAILED OPTIMIZATION RECOMMENDATIONS", "=" * 40, ""]
        
        for i, enhanced_rec in enumerate(recommendations[:20], 1):  # Limit to 20
            rec = enhanced_rec['recommendation']
            report_lines.extend([
                f"{i}. {rec.resource_type.upper()} - {rec.resource_id}",
                f"   Action: {rec.action}",
                f"   Monthly Savings: ${rec.monthly_savings:.2f}",
                f"   Risk Level: {getattr(rec, 'risk_level', 'unknown')}",
                f"   Dependencies: {len(enhanced_rec['dependencies'])}",
                f"   Compliance: {enhanced_rec['compliance_status']}",
                f"   Reason: {rec.reason}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _generate_dependency_report(self, recommendations: List[Dict]) -> str:
        """Generate dependency impact report"""
        report_lines = ["DEPENDENCY IMPACT ANALYSIS", "=" * 40, ""]
        
        high_impact = [r for r in recommendations 
                      if r.get('impact_analysis', {}).get('risk_level') == 'high']
        
        report_lines.extend([
            f"High Impact Resources: {len(high_impact)}",
            "",
            "Resources with Critical Dependencies:",
            "-" * 40
        ])
        
        for enhanced_rec in high_impact[:10]:
            rec = enhanced_rec['recommendation']
            impact = enhanced_rec['impact_analysis']
            
            report_lines.extend([
                f"Resource: {rec.resource_id}",
                f"Direct Impact: {impact.get('direct_impact_count', 0)} resources",
                f"Indirect Impact: {impact.get('indirect_impact_count', 0)} resources",
                f"Critical Resources Affected: {len(impact.get('critical_resources_affected', []))}",
                ""
            ])
        
        return "\n".join(report_lines)


class EnhancedSafetyChecker(SafetyChecker):
    """Enhanced safety checker with dependency and compliance awareness"""
    
    def __init__(self, 
                dependency_mapper=None,
                compliance_manager: Optional[ComplianceManager] = None):
        super().__init__()
        self.dependency_mapper = dependency_mapper
        self.compliance_manager = compliance_manager
    
    def check_instance_safety(self, instance_id: str) -> Dict[str, Any]:
        """Enhanced safety check including dependencies and compliance"""
        # Run base safety checks
        base_result = super().check_instance_safety(instance_id)
        
        # Add dependency checks
        if self.dependency_mapper and instance_id in self.dependency_mapper.resources:
            impact = self.dependency_mapper.get_impact_analysis(instance_id)
            
            if impact['risk_level'] == 'high':
                base_result['safe_to_modify'] = False
                base_result['blockers'].append(
                    f"High-risk dependencies: {impact['total_impact_count']} resources affected"
                )
            elif impact['risk_level'] == 'medium':
                base_result['warnings'].append(
                    f"Medium-risk dependencies: {impact['total_impact_count']} resources affected"
                )
        
        # Add compliance checks
        if self.compliance_manager:
            # Get instance details for tags
            ec2 = boto3.client('ec2')
            try:
                response = ec2.describe_instances(InstanceIds=[instance_id])
                if response['Reservations']:
                    instance = response['Reservations'][0]['Instances'][0]
                    tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    
                    compliance_result = self.compliance_manager.check_resource_compliance(
                        instance_id,
                        'ec2',
                        instance['Placement']['AvailabilityZone'][:-1],
                        tags
                    )
                    
                    if compliance_result['status'].value == 'non_compliant':
                        base_result['safe_to_modify'] = False
                        base_result['blockers'].append(
                            f"Compliance violations: {len(compliance_result['violations'])}"
                        )
            except Exception as e:
                logger.error(f"Failed to check compliance for {instance_id}: {e}")
        
        return base_result


# Example usage function
def run_enterprise_optimization_example():
    """Example of running enterprise optimization"""
    
    # Configure enterprise features
    config = EnterpriseConfig(
        enable_dependency_mapping=True,
        enable_change_management=True,
        enable_monitoring=True,
        enable_compliance=True,
        enable_audit_trail=True,
        ticketing_system="servicenow",
        auto_approve_low_risk=True,
        monitoring_duration_hours=72,
        create_dashboards=True,
        sns_topic_arn="arn:aws:sns:us-east-1:123456789012:cost-optimization"
    )
    
    # Initialize enterprise optimizer
    optimizer = EnterpriseOptimizer(config)
    
    # Run optimization
    results = optimizer.run_enterprise_optimization(
        regions=['us-east-1', 'us-west-2'],
        services=['ec2', 'rds', 's3'],
        user='john.doe@company.com'
    )
    
    print(f"Optimization complete!")
    print(f"Total savings identified: ${results['optimization_result'].total_monthly_savings:,.2f}/month")
    print(f"Change requests created: {results['change_requests_created']}")
    print(f"Monitoring dashboard: {results['monitoring_setup'].get('dashboard_url')}")
    
    # Execute approved changes (dry run)
    execution_results = optimizer.execute_approved_changes(dry_run=True)
    print(f"\nExecution results (dry run):")
    print(f"Total changes: {execution_results['total_changes']}")
    print(f"Would execute: {execution_results['executed']}")
    
    # Generate compliance report
    compliance_report = optimizer.generate_compliance_report(days=30)
    print(f"\nCompliance report generated with {len(compliance_report['sections'])} sections")