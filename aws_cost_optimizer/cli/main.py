import click
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
from datetime import datetime

from ..optimization.ec2_optimizer import EC2Optimizer
from ..patterns.pattern_detector import PatternDetector
from ..safety.safety_checker import SafetyChecker


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), default='config/config.yaml',
              help='Path to configuration file')
@click.option('--profile', '-p', help='AWS profile to use')
@click.option('--region', '-r', multiple=True, help='AWS regions to analyze')
@click.pass_context
def cli(ctx, config, profile, region):
    """AWS Cost Optimizer - Find and implement cost savings in your AWS infrastructure"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if os.path.exists(config):
        with open(config, 'r') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        ctx.obj['config'] = {}
    
    # Override with command line options
    if profile:
        ctx.obj['config'].setdefault('aws', {})['profile'] = profile
    if region:
        ctx.obj['config'].setdefault('aws', {})['regions'] = list(region)


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'excel', 'table']), 
              default='table', help='Output format')
@click.option('--output', '-o', help='Output file path')
@click.option('--service', '-s', multiple=True, 
              type=click.Choice(['ec2', 's3', 'rds', 'ebs', 'all']), 
              default=['all'], help='Services to analyze')
@click.pass_context
def analyze(ctx, format, output, service):
    """Analyze AWS resources and generate optimization recommendations"""
    config = ctx.obj['config']
    recommendations = {}
    
    # Initialize components
    pattern_detector = PatternDetector()
    safety_checker = SafetyChecker()
    
    # Analyze EC2
    if 'all' in service or 'ec2' in service:
        click.echo("Analyzing EC2 instances...")
        ec2_config = config.get('optimization', {}).get('ec2', {})
        
        ec2_optimizer = EC2Optimizer(
            cpu_threshold=ec2_config.get('cpu_threshold', 10.0),
            memory_threshold=ec2_config.get('memory_threshold', 20.0),
            network_threshold=ec2_config.get('network_threshold', 5.0),
            observation_days=ec2_config.get('observation_days', 14)
        )
        
        ec2_optimizer.set_pattern_detector(pattern_detector)
        ec2_optimizer.set_safety_checker(safety_checker)
        
        regions = config.get('aws', {}).get('regions')
        ec2_recommendations = ec2_optimizer.analyze(regions)
        
        if ec2_recommendations:
            recommendations['ec2'] = ec2_recommendations
            
            # Count recommendations
            total_recs = sum(len(recs) for recs in ec2_recommendations.values())
            total_savings = sum(
                rec.monthly_savings 
                for recs in ec2_recommendations.values() 
                for rec in recs
            )
            
            click.echo(f"Found {total_recs} EC2 optimization opportunities")
            click.echo(f"Potential monthly savings: ${total_savings:,.2f}")
    
    # Output results
    if output:
        _save_recommendations(recommendations, output, format)
        click.echo(f"Results saved to {output}")
    else:
        _display_recommendations(recommendations, format)


@cli.command('quick-scan')
@click.option('--type', '-t', 'scan_type', required=True,
              type=click.Choice(['ec2', 's3', 'ebs', 'ri', 'anomalies']),
              help='Type of quick scan to perform')
@click.option('--region', '-r', help='AWS region (defaults to all regions)')
@click.pass_context
def quick_scan(ctx, scan_type, region):
    """Perform a quick scan for specific optimization opportunities"""
    config = ctx.obj['config']
    
    if scan_type == 'ec2':
        _quick_scan_ec2(config, region)
    elif scan_type == 's3':
        click.echo("S3 quick scan not yet implemented")
    elif scan_type == 'ebs':
        click.echo("EBS quick scan not yet implemented")
    elif scan_type == 'ri':
        click.echo("Reserved Instance recommendations not yet implemented")
    elif scan_type == 'anomalies':
        _quick_scan_anomalies(config, region)


@cli.command()
@click.option('--recommendation-file', '-f', required=True,
              type=click.Path(exists=True),
              help='Path to recommendations file')
@click.option('--dry-run/--no-dry-run', default=True,
              help='Perform a dry run without making changes')
@click.option('--filter-action', '-a', multiple=True,
              type=click.Choice(['stop', 'rightsize', 'delete']),
              help='Filter by action type')
@click.pass_context
def remediate(ctx, recommendation_file, dry_run, filter_action):
    """Apply optimization recommendations (with safety checks)"""
    config = ctx.obj['config']
    
    # Load recommendations
    with open(recommendation_file, 'r') as f:
        if recommendation_file.endswith('.json'):
            recommendations = json.load(f)
        else:
            click.echo("Only JSON recommendation files are supported")
            return
    
    if dry_run:
        click.echo("DRY RUN MODE - No changes will be made")
    
    # Process recommendations
    for service, service_recs in recommendations.items():
        if service == 'ec2':
            _remediate_ec2(service_recs, dry_run, filter_action)


def _quick_scan_ec2(config: Dict[str, Any], region: str = None):
    """Perform quick EC2 idle instance scan"""
    ec2_optimizer = EC2Optimizer(
        cpu_threshold=5.0,  # More aggressive for quick scan
        observation_days=7   # Shorter lookback
    )
    
    ec2_optimizer.set_safety_checker(SafetyChecker())
    
    regions = [region] if region else config.get('aws', {}).get('regions')
    recommendations = ec2_optimizer.analyze(regions)
    
    click.echo("\n=== EC2 Quick Scan Results ===")
    
    for region, recs in recommendations.items():
        idle_instances = [r for r in recs if r.action == 'stop']
        
        if idle_instances:
            click.echo(f"\nRegion: {region}")
            click.echo(f"Found {len(idle_instances)} potentially idle instances:")
            
            for rec in idle_instances:
                click.echo(f"  - {rec.instance_id} ({rec.instance_type}): "
                          f"${rec.monthly_savings:.2f}/month - {rec.reason}")


def _quick_scan_anomalies(config: Dict[str, Any], region: str = None):
    """Perform quick cost anomaly detection"""
    pattern_detector = PatternDetector()
    
    # Get account ID (simplified - in production, get from STS)
    account_id = 'current'
    
    cost_patterns = pattern_detector.analyze_cost_patterns(account_id, lookback_days=30)
    
    click.echo("\n=== Cost Anomaly Detection ===")
    
    if cost_patterns.get('anomalies'):
        click.echo(f"Found {len(cost_patterns['anomalies'])} cost anomalies:")
        
        for anomaly in cost_patterns['anomalies']:
            severity_color = 'red' if anomaly['severity'] == 'high' else 'yellow'
            click.secho(f"  Day {anomaly['day_index']}: ${anomaly['cost']:.2f} "
                       f"(Z-score: {anomaly['z_score']:.2f})",
                       fg=severity_color)
    else:
        click.echo("No significant cost anomalies detected")
    
    # Show trends
    click.echo(f"\nTotal spend (30 days): ${cost_patterns.get('total_spend', 0):,.2f}")
    click.echo(f"Average daily spend: ${cost_patterns.get('average_daily_spend', 0):,.2f}")
    
    if cost_patterns.get('cost_trend'):
        trend = cost_patterns['cost_trend']
        trend_color = 'red' if trend['trend'] == 'increasing' else 'green'
        click.secho(f"Cost trend: {trend['trend']} "
                   f"({trend.get('percentage_change', 0):.1f}% change)",
                   fg=trend_color)


def _save_recommendations(recommendations: Dict, output_path: str, format: str):
    """Save recommendations to file"""
    if format == 'json':
        # Convert recommendations to JSON-serializable format
        json_data = {}
        for service, regions in recommendations.items():
            json_data[service] = {}
            for region, recs in regions.items():
                json_data[service][region] = [
                    {
                        'instance_id': rec.instance_id,
                        'instance_type': rec.instance_type,
                        'action': rec.action,
                        'reason': rec.reason,
                        'monthly_savings': rec.monthly_savings,
                        'annual_savings': rec.annual_savings,
                        'risk_level': rec.risk_level,
                        'implementation_steps': rec.implementation_steps,
                        'tags': rec.tags,
                        'metrics': rec.metrics
                    }
                    for rec in recs
                ]
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    elif format == 'excel':
        # Create Excel workbook with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for service, regions in recommendations.items():
                data = []
                for region, recs in regions.items():
                    for rec in recs:
                        data.append({
                            'Region': region,
                            'Instance ID': rec.instance_id,
                            'Instance Type': rec.instance_type,
                            'Action': rec.action,
                            'Reason': rec.reason,
                            'Monthly Savings': rec.monthly_savings,
                            'Annual Savings': rec.annual_savings,
                            'Risk Level': rec.risk_level,
                            'Environment': rec.tags.get('Environment', 'Unknown')
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=service.upper(), index=False)
    
    elif format == 'csv':
        # Flatten all recommendations into a single CSV
        data = []
        for service, regions in recommendations.items():
            for region, recs in regions.items():
                for rec in recs:
                    data.append({
                        'Service': service,
                        'Region': region,
                        'Resource ID': rec.instance_id,
                        'Resource Type': rec.instance_type,
                        'Action': rec.action,
                        'Reason': rec.reason,
                        'Monthly Savings': rec.monthly_savings,
                        'Annual Savings': rec.annual_savings,
                        'Risk Level': rec.risk_level
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)


def _display_recommendations(recommendations: Dict, format: str):
    """Display recommendations in the specified format"""
    if format == 'table':
        for service, regions in recommendations.items():
            click.echo(f"\n=== {service.upper()} Recommendations ===")
            
            for region, recs in regions.items():
                if recs:
                    click.echo(f"\nRegion: {region}")
                    
                    for rec in recs:
                        click.echo(f"\nResource: {rec.instance_id} ({rec.instance_type})")
                        click.echo(f"Action: {rec.action}")
                        click.echo(f"Reason: {rec.reason}")
                        click.echo(f"Monthly Savings: ${rec.monthly_savings:,.2f}")
                        click.echo(f"Risk Level: {rec.risk_level}")
                        
                        if rec.implementation_steps:
                            click.echo("Implementation Steps:")
                            for step in rec.implementation_steps:
                                click.echo(f"  {step}")
    
    elif format == 'json':
        # Convert to JSON and print
        json_data = {}
        for service, regions in recommendations.items():
            json_data[service] = {}
            for region, recs in regions.items():
                json_data[service][region] = [
                    {
                        'instance_id': rec.instance_id,
                        'action': rec.action,
                        'monthly_savings': rec.monthly_savings,
                        'risk_level': rec.risk_level
                    }
                    for rec in recs
                ]
        
        click.echo(json.dumps(json_data, indent=2))


def _remediate_ec2(recommendations: Dict, dry_run: bool, filter_action: List[str]):
    """Apply EC2 remediation actions"""
    import boto3
    
    for region, recs in recommendations.items():
        if not recs:
            continue
        
        ec2 = boto3.client('ec2', region_name=region)
        
        for rec_data in recs:
            # Filter by action if specified
            if filter_action and rec_data['action'] not in filter_action:
                continue
            
            instance_id = rec_data['instance_id']
            action = rec_data['action']
            
            click.echo(f"\nProcessing {instance_id} in {region}...")
            click.echo(f"Action: {action}")
            click.echo(f"Reason: {rec_data['reason']}")
            click.echo(f"Potential savings: ${rec_data['monthly_savings']:.2f}/month")
            
            if not dry_run and click.confirm("Apply this optimization?"):
                try:
                    if action == 'stop':
                        ec2.stop_instances(InstanceIds=[instance_id])
                        click.echo(f"Stopped instance {instance_id}")
                    elif action == 'rightsize':
                        click.echo("Rightsizing requires manual intervention")
                        click.echo("Please follow the implementation steps provided")
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
            else:
                click.echo("Skipped (dry run or user declined)")


if __name__ == '__main__':
    cli()