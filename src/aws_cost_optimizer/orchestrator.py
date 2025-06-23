"""
AWS Cost Optimization Orchestrator - Integrates all optimization components
Provides a unified interface for comprehensive cost optimization
"""
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all optimization components
from .optimization.ec2_optimizer import EC2Optimizer
from .optimization.network_optimizer import NetworkOptimizer
from .optimization.s3_optimizer import S3Optimizer
from .optimization.reserved_instance_analyzer import ReservedInstanceAnalyzer
from .optimization.auto_remediation_engine import (
    AutoRemediationEngine, RemediationPolicy, RemediationAction
)
from .analysis.cost_anomaly_detector import CostAnomalyDetector
from .analysis.pattern_detector import PatternDetector
from .discovery.multi_account import MultiAccountInventory
from .discovery.s3_discovery import S3Discovery
from .reporting.excel_reporter import ExcelReporter
from .compliance import ComplianceManager, AuditTrail, AuditEventType

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Consolidated optimization results"""
    timestamp: datetime
    total_monthly_savings: float
    total_annual_savings: float
    ec2_savings: float
    network_savings: float
    s3_savings: float
    ri_savings: float
    anomalies_detected: int
    recommendations_count: int
    auto_remediation_tasks: int
    execution_time: float
    details: Dict[str, Any]

class CostOptimizationOrchestrator:
    """Main orchestrator for AWS cost optimization"""
    
    def __init__(self,
                 session: Optional[boto3.Session] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the orchestrator
        
        Args:
            session: Boto3 session
            config: Configuration dictionary
        """
        self.session = session or boto3.Session()
        self.config = config or {}
        
        # Initialize components
        self.ec2_optimizer = EC2Optimizer(session=self.session)
        self.network_optimizer = NetworkOptimizer(session=self.session)
        self.s3_optimizer = S3Optimizer(
            session=self.session,
            size_threshold_gb=self.config.get('s3_size_threshold_gb', 1024),
            observation_days=self.config.get('s3_observation_days', 90)
        )
        self.s3_discovery = S3Discovery(session=self.session)
        self.ri_analyzer = ReservedInstanceAnalyzer(
            session=self.session,
            lookback_days=self.config.get('ri_lookback_days', 90)
        )
        self.anomaly_detector = CostAnomalyDetector(
            session=self.session,
            lookback_days=self.config.get('anomaly_lookback_days', 90),
            anomaly_threshold=self.config.get('anomaly_threshold', 2.5)
        )
        self.pattern_detector = PatternDetector(session=self.session)
        self.excel_reporter = ExcelReporter()
        
        # Initialize auto-remediation if enabled
        self.auto_remediation = None
        if self.config.get('enable_auto_remediation', False):
            remediation_policy = self._create_remediation_policy()
            self.auto_remediation = AutoRemediationEngine(
                policy=remediation_policy,
                dry_run=self.config.get('remediation_dry_run', True),
                session=self.session
            )
        
        # Multi-account support
        self.multi_account = None
        if self.config.get('enable_multi_account', False):
            self.multi_account = MultiAccountInventory(session=self.session)
        
        # Compliance and audit trail
        self.compliance_manager = None
        self.audit_trail = None
        if self.config.get('enable_compliance', True):
            self.compliance_manager = ComplianceManager(
                config=self.config.get('compliance_config', {}),
                session=self.session
            )
            self.audit_trail = AuditTrail(
                config=self.config.get('audit_config', {}),
                session=self.session
            )
    
    def run_full_optimization(self, 
                            regions: List[str] = None,
                            services: List[str] = None) -> OptimizationResult:
        """
        Run comprehensive cost optimization analysis
        
        Args:
            regions: List of regions to analyze (None = all regions)
            services: List of services to analyze (None = all services)
            
        Returns:
            Consolidated optimization results
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive cost optimization analysis...")
        
        results = {
            'ec2': {},
            'network': {},
            'reserved_instances': {},
            'anomalies': [],
            'patterns': {},
            'remediation_tasks': []
        }
        
        # Use ThreadPoolExecutor for parallel analysis
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            # EC2 Optimization
            futures['ec2'] = executor.submit(
                self._run_ec2_optimization, regions
            )
            
            # Network Optimization
            futures['network'] = executor.submit(
                self._run_network_optimization, regions
            )
            
            # Reserved Instance Analysis
            futures['ri'] = executor.submit(
                self._run_ri_analysis
            )
            
            # Anomaly Detection
            futures['anomalies'] = executor.submit(
                self._run_anomaly_detection, services
            )
            
            # Pattern Detection
            futures['patterns'] = executor.submit(
                self._run_pattern_detection
            )
            
            # Collect results
            for key, future in futures.items():
                try:
                    if key == 'ec2':
                        results['ec2'] = future.result()
                    elif key == 'network':
                        results['network'] = future.result()
                    elif key == 'ri':
                        results['reserved_instances'] = future.result()
                    elif key == 'anomalies':
                        results['anomalies'] = future.result()
                    elif key == 'patterns':
                        results['patterns'] = future.result()
                except Exception as e:
                    logger.error(f"Error in {key} analysis: {e}")
        
        # Create remediation tasks if enabled
        if self.auto_remediation:
            results['remediation_tasks'] = self._create_remediation_tasks(results)
        
        # Calculate total savings
        total_monthly_savings = (
            results['ec2'].get('total_monthly_savings', 0) +
            results['network'].get('total_monthly_savings', 0) +
            results['reserved_instances'].get('total_monthly_savings', 0)
        )
        
        # Create optimization result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        optimization_result = OptimizationResult(
            timestamp=datetime.now(),
            total_monthly_savings=total_monthly_savings,
            total_annual_savings=total_monthly_savings * 12,
            ec2_savings=results['ec2'].get('total_monthly_savings', 0),
            network_savings=results['network'].get('total_monthly_savings', 0),
            ri_savings=results['reserved_instances'].get('total_monthly_savings', 0),
            anomalies_detected=len(results['anomalies']),
            recommendations_count=self._count_recommendations(results),
            auto_remediation_tasks=len(results['remediation_tasks']),
            execution_time=execution_time,
            details=results
        )
        
        logger.info(f"Optimization analysis completed in {execution_time:.2f} seconds")
        logger.info(f"Total potential monthly savings: ${total_monthly_savings:,.2f}")
        
        return optimization_result
    
    def _run_ec2_optimization(self, regions: List[str] = None) -> Dict[str, Any]:
        """Run EC2 optimization analysis"""
        logger.info("Running EC2 optimization...")
        
        recommendations = self.ec2_optimizer.analyze_all_instances(regions)
        
        # Apply compliance filtering if enabled
        if self.compliance_manager:
            compliant_recommendations = []
            for rec in recommendations:
                # Get instance tags (simplified - would need actual tag lookup)
                tags = getattr(rec, 'tags', {})
                if self.compliance_manager.check_optimization_compliance(rec, tags):
                    compliant_recommendations.append(rec)
                else:
                    logger.info(f"Filtered non-compliant recommendation for {rec.instance_id}")
            recommendations = compliant_recommendations
        
        # Log recommendations to audit trail
        if self.audit_trail:
            user = self.config.get('user', 'system')
            for rec in recommendations:
                self.audit_trail.log_recommendation(user, rec, getattr(rec, 'tags', {}))
        
        total_savings = sum(r.monthly_savings for r in recommendations)
        
        return {
            'recommendations': recommendations,
            'total_monthly_savings': total_savings,
            'instance_count': len(recommendations),
            'top_opportunities': sorted(
                recommendations, 
                key=lambda x: x.monthly_savings, 
                reverse=True
            )[:5]
        }
    
    def _run_network_optimization(self, regions: List[str] = None) -> Dict[str, Any]:
        """Run network optimization analysis"""
        logger.info("Running network optimization...")
        
        recommendations = self.network_optimizer.analyze_all_network_costs(regions)
        
        total_savings = sum(r.estimated_monthly_savings for r in recommendations)
        
        # Group by type
        by_type = {}
        for rec in recommendations:
            if rec.resource_type not in by_type:
                by_type[rec.resource_type] = []
            by_type[rec.resource_type].append(rec)
        
        return {
            'recommendations': recommendations,
            'total_monthly_savings': total_savings,
            'by_type': by_type,
            'nat_gateway_savings': sum(
                r.estimated_monthly_savings 
                for r in recommendations 
                if r.resource_type == 'nat_gateway'
            )
        }
    
    def _run_ri_analysis(self) -> Dict[str, Any]:
        """Run Reserved Instance analysis"""
        logger.info("Running Reserved Instance analysis...")
        
        analysis = self.ri_analyzer.analyze_all_opportunities()
        
        return analysis
    
    def _run_anomaly_detection(self, services: List[str] = None) -> List[Any]:
        """Run cost anomaly detection"""
        logger.info("Running anomaly detection...")
        
        anomalies = self.anomaly_detector.detect_anomalies(
            real_time=True,
            services_filter=services
        )
        
        # Send alerts for critical anomalies if configured
        if self.config.get('anomaly_alerts_enabled', False):
            sns_topic = self.config.get('anomaly_sns_topic')
            if sns_topic:
                self.anomaly_detector.send_alerts(anomalies, sns_topic)
        
        return anomalies
    
    def _run_pattern_detection(self) -> Dict[str, Any]:
        """Run usage pattern detection"""
        logger.info("Running pattern detection...")
        
        patterns = self.pattern_detector.detect_all_patterns()
        
        return patterns
    
    def _create_remediation_policy(self) -> RemediationPolicy:
        """Create remediation policy from config"""
        return RemediationPolicy(
            max_monthly_savings=self.config.get('max_auto_remediation_savings', 500),
            allowed_actions=[
                RemediationAction.STOP_INSTANCE,
                RemediationAction.DELETE_SNAPSHOT,
                RemediationAction.RELEASE_ELASTIC_IP,
                RemediationAction.ENABLE_S3_LIFECYCLE,
                RemediationAction.CREATE_BUDGET_ALERT
            ],
            require_approval_for_production=self.config.get('require_prod_approval', True),
            auto_rollback_on_error=True,
            business_hours_only=self.config.get('business_hours_only', True),
            blackout_periods=self.config.get('blackout_periods', []),
            notification_endpoints=self.config.get('notification_endpoints', [])
        )
    
    def _create_remediation_tasks(self, results: Dict[str, Any]) -> List[Any]:
        """Create auto-remediation tasks from recommendations"""
        tasks = []
        
        # EC2 recommendations
        for rec in results['ec2'].get('recommendations', [])[:10]:  # Limit to top 10
            if rec.action == 'stop' and rec.confidence > 0.8:
                task = self.auto_remediation.create_remediation_task(
                    action=RemediationAction.STOP_INSTANCE,
                    resource_id=rec.instance_id,
                    resource_type='instance',
                    region=rec.region,
                    estimated_savings=rec.monthly_savings,
                    risk_level='low' if rec.confidence > 0.9 else 'medium'
                )
                tasks.append(task)
        
        # Network recommendations
        for rec in results['network'].get('recommendations', [])[:5]:
            if rec.resource_type == 'elastic_ip' and rec.confidence > 0.9:
                task = self.auto_remediation.create_remediation_task(
                    action=RemediationAction.RELEASE_ELASTIC_IP,
                    resource_id=rec.resource_id,
                    resource_type='elastic_ip',
                    region=rec.region,
                    estimated_savings=rec.estimated_monthly_savings,
                    risk_level='low'
                )
                tasks.append(task)
        
        logger.info(f"Created {len(tasks)} remediation tasks")
        
        return tasks
    
    def _count_recommendations(self, results: Dict[str, Any]) -> int:
        """Count total recommendations across all components"""
        count = 0
        
        count += len(results['ec2'].get('recommendations', []))
        count += len(results['network'].get('recommendations', []))
        count += len(results['reserved_instances'].get('ri_recommendations', []))
        count += len(results['reserved_instances'].get('sp_recommendations', []))
        
        return count
    
    def execute_remediation_tasks(self) -> Dict[str, Any]:
        """Execute approved remediation tasks"""
        if not self.auto_remediation:
            logger.warning("Auto-remediation is not enabled")
            return {}
        
        return self.auto_remediation.execute_approved_tasks()
    
    def generate_executive_report(self, 
                                result: OptimizationResult,
                                output_file: str = 'executive_report.html') -> str:
        """Generate executive summary report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AWS Cost Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #232F3E; }}
                h2 {{ color: #FF9900; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 20px; text-align: center; }}
                .metric-value {{ font-size: 36px; font-weight: bold; color: #232F3E; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #232F3E; color: white; }}
                .savings {{ color: #2ecc71; font-weight: bold; }}
                .warning {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>AWS Cost Optimization Report</h1>
            <p>Generated on: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value savings">${result.total_monthly_savings:,.0f}</div>
                    <div class="metric-label">Monthly Savings</div>
                </div>
                <div class="metric">
                    <div class="metric-value savings">${result.total_annual_savings:,.0f}</div>
                    <div class="metric-label">Annual Savings</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.recommendations_count}</div>
                    <div class="metric-label">Recommendations</div>
                </div>
                <div class="metric">
                    <div class="metric-value warning">{result.anomalies_detected}</div>
                    <div class="metric-label">Anomalies Detected</div>
                </div>
            </div>
            
            <h2>Savings Breakdown</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Monthly Savings</th>
                    <th>Annual Savings</th>
                    <th>Percentage of Total</th>
                </tr>
                <tr>
                    <td>EC2 Instances</td>
                    <td class="savings">${result.ec2_savings:,.2f}</td>
                    <td class="savings">${result.ec2_savings * 12:,.2f}</td>
                    <td>{(result.ec2_savings / result.total_monthly_savings * 100):.1f}%</td>
                </tr>
                <tr>
                    <td>Network Resources</td>
                    <td class="savings">${result.network_savings:,.2f}</td>
                    <td class="savings">${result.network_savings * 12:,.2f}</td>
                    <td>{(result.network_savings / result.total_monthly_savings * 100):.1f}%</td>
                </tr>
                <tr>
                    <td>Reserved Instances</td>
                    <td class="savings">${result.ri_savings:,.2f}</td>
                    <td class="savings">${result.ri_savings * 12:,.2f}</td>
                    <td>{(result.ri_savings / result.total_monthly_savings * 100):.1f}%</td>
                </tr>
            </table>
            
            <h2>Top EC2 Optimization Opportunities</h2>
            <table>
                <tr>
                    <th>Instance ID</th>
                    <th>Type</th>
                    <th>Action</th>
                    <th>Monthly Savings</th>
                </tr>
        """
        
        # Add top EC2 recommendations
        for rec in result.details['ec2'].get('top_opportunities', [])[:5]:
            html_content += f"""
                <tr>
                    <td>{rec.instance_id}</td>
                    <td>{rec.instance_type}</td>
                    <td>{rec.action}</td>
                    <td class="savings">${rec.monthly_savings:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Cost Anomalies</h2>
            <table>
                <tr>
                    <th>Service</th>
                    <th>Severity</th>
                    <th>Current Cost</th>
                    <th>Expected Cost</th>
                    <th>Impact</th>
                </tr>
        """
        
        # Add anomalies
        for anomaly in result.details['anomalies'][:5]:
            severity_class = 'warning' if anomaly.severity in ['high', 'critical'] else ''
            html_content += f"""
                <tr>
                    <td>{anomaly.service}</td>
                    <td class="{severity_class}">{anomaly.severity.upper()}</td>
                    <td>${anomaly.current_daily_cost:.2f}</td>
                    <td>${anomaly.expected_daily_cost:.2f}</td>
                    <td class="warning">+${anomaly.cost_impact:.2f}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <h2>Auto-Remediation Status</h2>
            <p>{result.auto_remediation_tasks} tasks created for automatic remediation.</p>
            
            <hr>
            <p><small>Report generated in {result.execution_time:.2f} seconds</small></p>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Executive report generated: {output_file}")
        
        return output_file
    
    def export_detailed_results(self, 
                              result: OptimizationResult,
                              output_file: str = 'detailed_optimization_report.xlsx'):
        """Export detailed results to Excel"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Analysis Timestamp',
                    'Total Monthly Savings',
                    'Total Annual Savings',
                    'EC2 Savings',
                    'Network Savings',
                    'RI/SP Savings',
                    'Anomalies Detected',
                    'Total Recommendations',
                    'Remediation Tasks',
                    'Execution Time (seconds)'
                ],
                'Value': [
                    result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    f"${result.total_monthly_savings:,.2f}",
                    f"${result.total_annual_savings:,.2f}",
                    f"${result.ec2_savings:,.2f}",
                    f"${result.network_savings:,.2f}",
                    f"${result.ri_savings:,.2f}",
                    result.anomalies_detected,
                    result.recommendations_count,
                    result.auto_remediation_tasks,
                    f"{result.execution_time:.2f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # EC2 Recommendations
            if result.details['ec2'].get('recommendations'):
                ec2_data = []
                for rec in result.details['ec2']['recommendations']:
                    ec2_data.append({
                        'Instance ID': rec.instance_id,
                        'Region': rec.region,
                        'Type': rec.instance_type,
                        'Action': rec.action,
                        'Reason': rec.reason,
                        'Monthly Savings': f"${rec.monthly_savings:.2f}",
                        'Confidence': f"{rec.confidence:.0%}"
                    })
                pd.DataFrame(ec2_data).to_excel(writer, sheet_name='EC2 Optimization', index=False)
            
            # Network Recommendations
            if result.details['network'].get('recommendations'):
                network_data = []
                for rec in result.details['network']['recommendations']:
                    network_data.append({
                        'Resource ID': rec.resource_id,
                        'Type': rec.resource_type,
                        'Region': rec.region,
                        'Action': rec.recommended_action,
                        'Monthly Savings': f"${rec.estimated_monthly_savings:.2f}",
                        'Risk': rec.risk_level,
                        'Reason': rec.reason
                    })
                pd.DataFrame(network_data).to_excel(writer, sheet_name='Network Optimization', index=False)
            
            # Cost Anomalies
            if result.details['anomalies']:
                anomaly_data = []
                for anomaly in result.details['anomalies']:
                    anomaly_data.append({
                        'Service': anomaly.service,
                        'Type': anomaly.anomaly_type,
                        'Severity': anomaly.severity,
                        'Current Daily Cost': f"${anomaly.current_daily_cost:.2f}",
                        'Expected Daily Cost': f"${anomaly.expected_daily_cost:.2f}",
                        'Impact': f"${anomaly.cost_impact:.2f}",
                        'Increase %': f"{anomaly.percentage_increase:.1f}%",
                        'Confidence': f"{anomaly.confidence_score:.0%}"
                    })
                pd.DataFrame(anomaly_data).to_excel(writer, sheet_name='Cost Anomalies', index=False)
        
        logger.info(f"Detailed report exported to {output_file}")
        
        return output_file
    
    def generate_compliance_report(self,
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 output_file: str = 'compliance_report.html') -> str:
        """Generate compliance report"""
        if not self.compliance_manager:
            logger.warning("Compliance manager not enabled")
            return ""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        compliance_report = self.compliance_manager.generate_compliance_report(
            start_date, end_date
        )
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AWS Cost Optimizer Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #232F3E; }}
                h2 {{ color: #FF9900; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; }}
                .compliant {{ color: #2ecc71; }}
                .non-compliant {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #232F3E; color: white; }}
            </style>
        </head>
        <body>
            <h1>Compliance Report</h1>
            <p>Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Resources Checked: {compliance_report['summary']['total_resources_checked']}</p>
                <p class="compliant">Compliant Resources: {compliance_report['summary']['compliant_resources']}</p>
                <p class="non-compliant">Non-Compliant Resources: {compliance_report['summary']['non_compliant_resources']}</p>
            </div>
            
            <h2>Violations by Severity</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Count</th>
                </tr>
        """
        
        for severity, count in compliance_report['summary']['violations_by_severity'].items():
            html_content += f"""
                <tr>
                    <td>{severity}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Compliance report generated: {output_file}")
        return output_file
    
    def generate_audit_report(self,
                            start_date: datetime = None,
                            end_date: datetime = None,
                            output_file: str = 'audit_report.json') -> str:
        """Generate audit trail report"""
        if not self.audit_trail:
            logger.warning("Audit trail not enabled")
            return ""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        audit_report = self.audit_trail.generate_audit_report(start_date, end_date)
        
        with open(output_file, 'w') as f:
            json.dump(audit_report, f, indent=2, default=str)
        
        logger.info(f"Audit report generated: {output_file}")
        return output_file
    
    def run_enterprise_optimization(self, 
                                  regions: List[str] = None,
                                  services: List[str] = None,
                                  user: str = "system") -> Dict[str, Any]:
        """
        Run optimization with enterprise features enabled
        
        This method delegates to the EnterpriseOptimizer for advanced features
        like dependency mapping, change management, and enhanced monitoring
        """
        from .enterprise import EnterpriseConfig, EnterpriseOptimizer
        
        # Create enterprise config from existing config
        enterprise_config = EnterpriseConfig(
            enable_dependency_mapping=self.config.get('enterprise', {}).get('enable_dependency_mapping', True),
            enable_change_management=self.config.get('enterprise', {}).get('enable_change_management', True),
            enable_monitoring=self.config.get('enterprise', {}).get('enable_monitoring', True),
            enable_compliance=self.config.get('enable_compliance', True),
            enable_audit_trail=self.config.get('enable_compliance', True),
            ticketing_system=self.config.get('enterprise', {}).get('ticketing_system', 'none'),
            auto_approve_low_risk=self.config.get('enterprise', {}).get('auto_approve_low_risk', False),
            monitoring_duration_hours=self.config.get('enterprise', {}).get('monitoring_duration_hours', 72),
            create_dashboards=self.config.get('enterprise', {}).get('create_dashboards', True),
            sns_topic_arn=self.config.get('enterprise', {}).get('sns_topic_arn')
        )
        
        # Initialize enterprise optimizer
        enterprise_optimizer = EnterpriseOptimizer(enterprise_config, self.session)
        
        # Run enterprise optimization
        return enterprise_optimizer.run_enterprise_optimization(regions, services, user)