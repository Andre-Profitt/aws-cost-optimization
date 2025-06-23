import click
import yaml
import json
import boto3
from pathlib import Path
from datetime import datetime
from .discovery.multi_account import MultiAccountInventory
from .orchestrator import CostOptimizationOrchestrator

@click.group()
@click.pass_context
def cli(ctx):
    """AWS Cost Optimizer - Comprehensive cost reduction toolkit"""
    ctx.ensure_object(dict)
    
    # Load configuration
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path) as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        ctx.obj['config'] = {}

@cli.command()
@click.option('--regions', '-r', multiple=True, help='AWS regions to analyze')
@click.option('--services', '-s', multiple=True, help='Services to analyze')
@click.option('--output', '-o', default='optimization_report', help='Output file prefix')
@click.option('--format', type=click.Choice(['json', 'excel', 'html']), default='html')
@click.pass_context
def analyze(ctx, regions, services, output, format):
    """Run comprehensive cost optimization analysis"""
    click.echo("🔍 Starting comprehensive cost optimization analysis...")
    
    # Initialize orchestrator
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    # Convert tuples to lists
    regions_list = list(regions) if regions else None
    services_list = list(services) if services else None
    
    # Run analysis
    with click.progressbar(length=100, label='Analyzing') as bar:
        bar.update(10)
        result = orchestrator.run_full_optimization(
            regions=regions_list,
            services=services_list
        )
        bar.update(90)
    
    # Generate reports
    click.echo(f"\n💰 Total potential monthly savings: ${result.total_monthly_savings:,.2f}")
    click.echo(f"📊 Found {result.recommendations_count} optimization opportunities")
    click.echo(f"⚠️  Detected {result.anomalies_detected} cost anomalies")
    
    # Export results
    if format == 'json':
        output_file = f"{output}.json"
        with open(output_file, 'w') as f:
            json.dump(result.details, f, indent=2, default=str)
        click.echo(f"\n✅ Results saved to {output_file}")
    elif format == 'excel':
        output_file = f"{output}.xlsx"
        orchestrator.export_detailed_results(result, output_file)
        click.echo(f"\n✅ Results saved to {output_file}")
    else:  # html
        output_file = f"{output}.html"
        orchestrator.generate_executive_report(result, output_file)
        click.echo(f"\n✅ Executive report saved to {output_file}")

@cli.command()
@click.option('--type', type=click.Choice(['ec2', 'network', 'ri', 'anomalies']), help='Analysis type')
@click.option('--region', '-r', help='Specific region to analyze')
@click.pass_context
def quick_scan(ctx, type, region):
    """Run a quick scan for specific optimization type"""
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    regions = [region] if region else None
    
    if type == 'ec2':
        click.echo("🖥️  Scanning EC2 instances...")
        result = orchestrator._run_ec2_optimization(regions)
        click.echo(f"Found {len(result['recommendations'])} EC2 optimization opportunities")
        click.echo(f"Potential monthly savings: ${result['total_monthly_savings']:,.2f}")
    
    elif type == 'network':
        click.echo("🌐 Scanning network resources...")
        result = orchestrator._run_network_optimization(regions)
        click.echo(f"Found {len(result['recommendations'])} network optimization opportunities")
        click.echo(f"Potential monthly savings: ${result['total_monthly_savings']:,.2f}")
    
    elif type == 'ri':
        click.echo("📋 Analyzing Reserved Instance opportunities...")
        result = orchestrator._run_ri_analysis()
        click.echo(f"Found {len(result['ri_recommendations'])} RI recommendations")
        click.echo(f"Found {len(result['sp_recommendations'])} Savings Plan recommendations")
        click.echo(f"Potential monthly savings: ${result['total_monthly_savings']:,.2f}")
    
    elif type == 'anomalies':
        click.echo("🚨 Detecting cost anomalies...")
        anomalies = orchestrator._run_anomaly_detection()
        click.echo(f"Detected {len(anomalies)} cost anomalies")
        for anomaly in anomalies[:5]:
            click.echo(f"  - {anomaly.service}: {anomaly.severity} severity, ${anomaly.cost_impact:.2f} impact")

@cli.command()
@click.option('--dry-run/--execute', default=True)
@click.option('--auto-approve', is_flag=True, help='Auto-approve low-risk remediations')
@click.pass_context
def remediate(ctx, dry_run, auto_approve):
    """Execute auto-remediation tasks"""
    mode = "DRY RUN" if dry_run else "EXECUTE"
    click.echo(f"🚀 Starting auto-remediation in {mode} mode...")
    
    # Update config for remediation
    config = ctx.obj.get('config', {})
    config['enable_auto_remediation'] = True
    config['remediation_dry_run'] = dry_run
    
    orchestrator = CostOptimizationOrchestrator(config=config)
    
    if not orchestrator.auto_remediation:
        click.echo("❌ Auto-remediation is not enabled in configuration")
        return
    
    # Run analysis to create tasks
    click.echo("Analyzing for remediation opportunities...")
    result = orchestrator.run_full_optimization()
    
    # Get pending tasks
    tasks = orchestrator.auto_remediation.get_all_tasks()
    
    if not tasks:
        click.echo("No remediation tasks created")
        return
    
    click.echo(f"\nCreated {len(tasks)} remediation tasks:")
    for task in tasks[:10]:
        click.echo(f"  - {task.action.value}: {task.resource_id} (${task.estimated_savings:.2f}/month)")
    
    if auto_approve:
        # Auto-approve low-risk tasks
        approved = 0
        for task in tasks:
            if task.risk_level == 'low' and task.estimated_savings < 500:
                orchestrator.auto_remediation.approve_task(task.task_id)
                approved += 1
        click.echo(f"\n✅ Auto-approved {approved} low-risk tasks")
    
    # Execute approved tasks
    if click.confirm('\nExecute approved remediation tasks?'):
        results = orchestrator.execute_remediation_tasks()
        click.echo(f"\n✅ Executed {results['executed']} tasks")
        click.echo(f"   Succeeded: {results['succeeded']}")
        click.echo(f"   Failed: {results['failed']}")
        click.echo(f"   Total savings: ${results['total_savings']:.2f}/month")


@cli.command()
@click.option('--output', '-o', default='cost_report.html')
@click.pass_context
def report(ctx, output):
    """Generate executive cost optimization report"""
    click.echo("📊 Generating executive report...")
    
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    # Run full analysis
    result = orchestrator.run_full_optimization()
    
    # Generate report
    report_file = orchestrator.generate_executive_report(result, output)
    
    click.echo(f"\n✅ Executive report generated: {report_file}")
    click.echo(f"\nKey Findings:")
    click.echo(f"  💰 Total Monthly Savings: ${result.total_monthly_savings:,.2f}")
    click.echo(f"  📈 Annual Savings: ${result.total_annual_savings:,.2f}")
    click.echo(f"  🎯 Recommendations: {result.recommendations_count}")
    click.echo(f"  ⚠️  Anomalies: {result.anomalies_detected}")


@cli.command()
@click.pass_context
def configure(ctx):
    """Interactive configuration setup"""
    click.echo("⚙️  AWS Cost Optimizer Configuration")
    
    config = {}
    
    # AWS Configuration
    click.echo("\n📍 AWS Configuration:")
    config['aws_profile'] = click.prompt('AWS Profile name', default='default')
    config['regions'] = click.prompt('Regions to analyze (comma-separated)', default='us-east-1,us-west-2').split(',')
    
    # Optimization Settings
    click.echo("\n🎯 Optimization Settings:")
    config['enable_auto_remediation'] = click.confirm('Enable auto-remediation?', default=False)
    if config['enable_auto_remediation']:
        config['max_auto_remediation_savings'] = click.prompt('Max monthly savings to auto-approve', type=float, default=500.0)
    
    # Anomaly Detection
    click.echo("\n🚨 Anomaly Detection:")
    config['anomaly_lookback_days'] = click.prompt('Days of history for anomaly detection', type=int, default=90)
    config['anomaly_threshold'] = click.prompt('Anomaly threshold (z-score)', type=float, default=2.5)
    
    # Notifications
    click.echo("\n📧 Notifications:")
    config['anomaly_alerts_enabled'] = click.confirm('Enable anomaly alerts?', default=False)
    if config['anomaly_alerts_enabled']:
        config['anomaly_sns_topic'] = click.prompt('SNS Topic ARN for alerts')
    
    # Save configuration
    config_path = Path('config/config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    click.echo(f"\n✅ Configuration saved to {config_path}")

if __name__ == '__main__':
    cli(obj={})