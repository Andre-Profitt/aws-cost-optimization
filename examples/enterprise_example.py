#!/usr/bin/env python3
"""
Enterprise AWS Cost Optimizer - Complete Usage Example

This script demonstrates how to use all enterprise features together
for a production-grade cost optimization workflow.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import boto3
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import enterprise components
from aws_cost_optimizer.enterprise.integration import (
    EnterpriseOptimizer,
    EnterpriseConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_configuration(config_path: str = 'config/enterprise_config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(obj):
        if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        elif isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        return obj
    
    return replace_env_vars(config)


def setup_enterprise_config(config: Dict[str, Any]) -> EnterpriseConfig:
    """Set up enterprise configuration from loaded config"""
    enterprise_settings = config.get('enterprise', {})
    
    return EnterpriseConfig(
        # Dependency mapping
        enable_dependency_mapping=enterprise_settings.get('dependency_mapping', {}).get('enabled', True),
        
        # Change management
        enable_change_management=enterprise_settings.get('change_management', {}).get('enabled', True),
        ticketing_system=enterprise_settings.get('change_management', {}).get('ticketing_system', 'servicenow'),
        auto_approve_low_risk=enterprise_settings.get('change_management', {})
            .get('approval_rules', {}).get('auto_approve_low_risk', False),
        require_approval_for_production=enterprise_settings.get('change_management', {})
            .get('approval_rules', {}).get('require_approval_for_production', True),
        
        # Monitoring
        enable_monitoring=enterprise_settings.get('monitoring', {}).get('enabled', True),
        monitoring_duration_hours=enterprise_settings.get('monitoring', {})
            .get('monitoring_duration_hours', 72),
        create_dashboards=enterprise_settings.get('monitoring', {}).get('create_dashboards', True),
        enable_anomaly_detection=enterprise_settings.get('monitoring', {})
            .get('enable_anomaly_detection', True),
        
        # Compliance
        enable_compliance=enterprise_settings.get('compliance', {}).get('enabled', True),
        enforce_compliance=enterprise_settings.get('compliance', {}).get('enforce_compliance', True),
        block_non_compliant=enterprise_settings.get('compliance', {}).get('block_non_compliant', True),
        
        # Audit trail
        enable_audit_trail=enterprise_settings.get('audit', {}).get('enabled', True),
        
        # Notifications
        sns_topic_arn=config.get('notifications', {}).get('sns_topics', {}).get('alerts')
    )


def main():
    """Main execution function"""
    logger.info("Starting Enterprise AWS Cost Optimizer")
    
    # Load configuration
    config = load_configuration()
    enterprise_config = setup_enterprise_config(config)
    
    # Get user identity
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    user = identity.get('Arn', 'system').split('/')[-1]
    
    logger.info(f"Running as user: {user}")
    
    # Initialize enterprise optimizer
    logger.info("Initializing enterprise optimizer...")
    optimizer = EnterpriseOptimizer(enterprise_config)
    
    # Phase 1: Run comprehensive optimization analysis
    logger.info("=" * 50)
    logger.info("PHASE 1: OPTIMIZATION ANALYSIS")
    logger.info("=" * 50)
    
    regions = config.get('aws', {}).get('regions', ['us-east-1'])
    services = ['ec2', 'rds', 's3']  # Focus on main services
    
    optimization_results = optimizer.run_enterprise_optimization(
        regions=regions,
        services=services,
        user=user
    )
    
    # Display results
    logger.info(f"\nOptimization Analysis Complete:")
    logger.info(f"  Total Monthly Savings: ${optimization_results['optimization_result'].total_monthly_savings:,.2f}")
    logger.info(f"  Total Annual Savings: ${optimization_results['optimization_result'].total_annual_savings:,.2f}")
    logger.info(f"  Compliant Recommendations: {optimization_results['compliant_recommendations']}")
    logger.info(f"  Change Requests Created: {optimization_results['change_requests_created']}")
    
    if optimization_results.get('monitoring_setup', {}).get('dashboard_url'):
        logger.info(f"  Monitoring Dashboard: {optimization_results['monitoring_setup']['dashboard_url']}")
    
    # Save reports
    reports_dir = Path(config.get('reporting', {}).get('output_dir', 'reports'))
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for report_name, report_content in optimization_results.get('reports', {}).items():
        report_path = reports_dir / f"{report_name}_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        logger.info(f"  Saved report: {report_path}")
    
    # Phase 2: Review and approve changes
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: CHANGE APPROVAL WORKFLOW")
    logger.info("=" * 50)
    
    if enterprise_config.enable_change_management and optimizer.change_management:
        # Get pending changes
        pending_changes = optimizer.change_management.get_pending_changes()
        
        logger.info(f"\nPending Change Requests: {len(pending_changes)}")
        
        # Display pending changes
        for i, change in enumerate(pending_changes[:5], 1):  # Show first 5
            logger.info(f"\n{i}. Change Request: {change.request_id}")
            logger.info(f"   Resource: {change.resource_type} - {change.resource_id}")
            logger.info(f"   Action: {change.change_type.value}")
            logger.info(f"   Risk Level: {change.risk_level.value}")
            logger.info(f"   Monthly Savings: ${change.estimated_savings:.2f}")
            logger.info(f"   Dependencies: {len(change.dependencies)}")
            
            if change.ticket_id:
                logger.info(f"   Ticket: {change.ticket_system} - {change.ticket_id}")
        
        # In a real scenario, approvals would come through the ticketing system
        # For this example, we'll simulate approval of low-risk changes
        if enterprise_config.auto_approve_low_risk:
            logger.info("\n[Auto-approving low-risk changes...]")
            # Auto-approval is handled by the change management system
    
    # Phase 3: Execute approved changes
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: CHANGE EXECUTION")
    logger.info("=" * 50)
    
    # Check if we're in a safe window for changes
    safety_config = config.get('safety', {})
    if safety_config.get('enhanced_checks', {}).get('check_business_hours', True):
        from datetime import datetime
        import pytz
        
        tz = pytz.timezone(safety_config.get('business_hours', {}).get('timezone', 'UTC'))
        now = datetime.now(tz)
        
        business_hours = safety_config.get('business_hours', {})
        if (business_hours.get('start_hour', 0) <= now.hour < business_hours.get('end_hour', 24) and
            now.weekday() in business_hours.get('work_days', range(7))):
            logger.info("✓ Within business hours - safe to proceed")
        else:
            logger.warning("⚠ Outside business hours - deferring execution")
            # In production, you might want to schedule for the next business day
    
    # Execute changes (dry run by default)
    dry_run = safety_config.get('dry_run', True)
    
    logger.info(f"\nExecuting approved changes (dry_run={dry_run})...")
    execution_results = optimizer.execute_approved_changes(dry_run=dry_run)
    
    logger.info(f"\nExecution Results:")
    logger.info(f"  Total Changes: {execution_results['total_changes']}")
    logger.info(f"  Executed: {execution_results['executed']}")
    logger.info(f"  Failed: {execution_results['failed']}")
    logger.info(f"  Monitoring Enabled: {execution_results['monitoring_enabled']}")
    
    # Phase 4: Generate compliance and audit reports
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: COMPLIANCE AND REPORTING")
    logger.info("=" * 50)
    
    # Generate 30-day compliance report
    compliance_report = optimizer.generate_compliance_report(days=30)
    
    logger.info("\nCompliance Report Summary:")
    for section_name, section_data in compliance_report.get('sections', {}).items():
        logger.info(f"\n  {section_name.upper()}:")
        
        if section_name == 'compliance' and 'summary' in section_data:
            summary = section_data['summary']
            logger.info(f"    Total Resources Checked: {summary.get('total_resources_checked', 0)}")
            logger.info(f"    Compliant: {summary.get('compliant_resources', 0)}")
            logger.info(f"    Non-Compliant: {summary.get('non_compliant_resources', 0)}")
        
        elif section_name == 'audit' and 'summary' in section_data:
            summary = section_data['summary']
            logger.info(f"    Total Events: {summary.get('total_events', 0)}")
            logger.info(f"    Failed Events: {len(section_data.get('failed_events', []))}")
            logger.info(f"    High Risk Changes: {len(section_data.get('high_risk_changes', []))}")
        
        elif section_name == 'changes':
            logger.info(f"    Total Changes: {section_data.get('total_changes', 0)}")
            logger.info(f"    Approved: ${section_data.get('total_savings_approved', 0):,.2f}")
            logger.info(f"    Pending: ${section_data.get('total_savings_pending', 0):,.2f}")
    
    # Export audit logs for compliance
    if enterprise_config.enable_audit_trail and optimizer.audit_trail:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        audit_export = optimizer.audit_trail.export_audit_logs(
            start_date,
            end_date,
            format='json'
        )
        logger.info(f"\n  Audit logs exported to: {audit_export}")
    
    # Phase 5: Set up continuous monitoring
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: CONTINUOUS MONITORING")
    logger.info("=" * 50)
    
    if enterprise_config.enable_monitoring:
        logger.info("\nMonitoring Setup:")
        logger.info(f"  Post-change monitoring duration: {enterprise_config.monitoring_duration_hours} hours")
        logger.info(f"  Anomaly detection: {'Enabled' if enterprise_config.enable_anomaly_detection else 'Disabled'}")
        logger.info(f"  CloudWatch dashboards: {'Created' if enterprise_config.create_dashboards else 'Disabled'}")
        
        # The monitoring alarms are already set up during execution
        # Here we could set up additional monitoring or schedule reports
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("OPTIMIZATION WORKFLOW COMPLETE")
    logger.info("=" * 50)
    
    logger.info("\nNext Steps:")
    logger.info("1. Review the generated reports in the 'reports' directory")
    logger.info("2. Monitor the CloudWatch dashboard for post-change metrics")
    logger.info("3. Review and approve pending change requests in your ticketing system")
    logger.info("4. Schedule the next optimization run (recommended: weekly)")
    
    # Send summary notification
    if config.get('notifications', {}).get('slack', {}).get('enabled'):
        # In production, send Slack notification with summary
        logger.info("\n[Slack notification would be sent with summary]")


def schedule_optimization(cron_expression: str = "0 9 * * 1"):
    """Schedule regular optimization runs"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    
    scheduler = BlockingScheduler()
    
    # Schedule weekly optimization
    scheduler.add_job(
        main,
        CronTrigger.from_crontab(cron_expression),
        id='weekly_optimization',
        name='Weekly Cost Optimization',
        replace_existing=True
    )
    
    logger.info(f"Scheduled optimization with cron: {cron_expression}")
    logger.info("Press Ctrl+C to exit")
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise AWS Cost Optimizer')
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Run in scheduled mode (weekly by default)'
    )
    parser.add_argument(
        '--cron',
        default="0 9 * * 1",
        help='Cron expression for scheduled runs (default: Mondays at 9 AM)'
    )
    parser.add_argument(
        '--config',
        default='config/enterprise_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    if args.schedule:
        schedule_optimization(args.cron)
    else:
        main()