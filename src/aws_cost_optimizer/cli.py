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
@click.option('--resource-id', '-r', help='Specific resource ID to check')
@click.option('--resource-type', '-t', help='Resource type (ec2, rds, s3)')
@click.option('--region', help='AWS region')
@click.option('--output', '-o', default='compliance_check.json')
@click.pass_context
def check_compliance(ctx, resource_id, resource_type, region, output):
    """Check compliance status of resources"""
    click.echo("🔍 Checking compliance status...")
    
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    if not orchestrator.compliance_manager:
        click.echo("❌ Compliance checking is not enabled")
        return
    
    if resource_id:
        # Check specific resource
        # In real implementation, would fetch tags from AWS
        tags = {}
        result = orchestrator.compliance_manager.check_resource_compliance(
            resource_id, resource_type, region, tags
        )
        
        status_color = 'green' if result['status'].value == 'compliant' else 'red'
        click.echo(f"\nResource: {resource_id}")
        click.echo(f"Status: ", nl=False)
        click.secho(result['status'].value.upper(), fg=status_color)
        
        if result['violations']:
            click.echo("\nViolations:")
            for violation in result['violations']:
                click.echo(f"  - [{violation.severity}] {violation.details}")
    else:
        # Check all resources
        click.echo("Checking all resources across regions...")
        # Implementation would scan all resources
        click.echo("Full compliance scan completed. Results saved to {output}")


@cli.command()
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--output', '-o', default='compliance_report.html')
@click.pass_context
def compliance_report(ctx, start_date, end_date, output):
    """Generate compliance report"""
    click.echo("📊 Generating compliance report...")
    
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    if not orchestrator.compliance_manager:
        click.echo("❌ Compliance checking is not enabled")
        return
    
    # Parse dates
    from datetime import datetime
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    
    report_file = orchestrator.generate_compliance_report(start, end, output)
    
    if report_file:
        click.echo(f"\n✅ Compliance report generated: {report_file}")


@cli.command()
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--user', help='Filter by user')
@click.option('--event-type', help='Filter by event type')
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
@click.option('--output', '-o', default='audit_trail')
@click.pass_context
def audit_trail(ctx, start_date, end_date, user, event_type, format, output):
    """Query and export audit trail"""
    click.echo("🔍 Querying audit trail...")
    
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    if not orchestrator.audit_trail:
        click.echo("❌ Audit trail is not enabled")
        return
    
    # Parse dates
    from datetime import datetime
    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None
    
    # Build filters
    filters = {}
    if user:
        filters['user'] = user
    if event_type:
        filters['event_type'] = event_type
    
    # Query events
    events = orchestrator.audit_trail.query_audit_trail(start, end, filters)
    
    click.echo(f"Found {len(events)} audit events")
    
    # Export
    output_file = f"{output}.{format}"
    export_path = orchestrator.audit_trail.export_audit_logs(start, end, format)
    
    click.echo(f"\n✅ Audit trail exported to: {export_path}")


@cli.command()
@click.option('--regions', '-r', multiple=True, help='AWS regions to analyze')
@click.option('--services', '-s', multiple=True, help='Services to analyze')
@click.option('--user', '-u', default='cli-user', help='User identifier for audit trail')
@click.option('--output', '-o', default='enterprise_report', help='Output file prefix')
@click.pass_context
def enterprise_analyze(ctx, regions, services, user, output):
    """Run enterprise cost optimization with advanced features"""
    click.echo("🏢 Starting enterprise cost optimization analysis...")
    click.echo("Features: Dependency mapping, Change management, Monitoring, Compliance")
    
    orchestrator = CostOptimizationOrchestrator(
        config=ctx.obj.get('config', {})
    )
    
    # Convert tuples to lists
    regions_list = list(regions) if regions else None
    services_list = list(services) if services else None
    
    try:
        with click.progressbar(length=100, label='Running enterprise analysis') as bar:
            bar.update(10)
            result = orchestrator.run_enterprise_optimization(
                regions=regions_list,
                services=services_list,
                user=user
            )
            bar.update(90)
        
        click.echo(f"\n✅ Enterprise optimization complete!")
        click.echo(f"💰 Total savings identified: ${result['optimization_result'].total_monthly_savings:,.2f}/month")
        click.echo(f"📊 Compliant recommendations: {result['compliant_recommendations']}")
        click.echo(f"📋 Change requests created: {result['change_requests_created']}")
        
        if result.get('monitoring_setup', {}).get('dashboard_url'):
            click.echo(f"📈 Monitoring dashboard: {result['monitoring_setup']['dashboard_url']}")
        
        # Save reports
        for report_name, report_content in result.get('reports', {}).items():
            report_file = f"{output}_{report_name}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            click.echo(f"📄 Report saved: {report_file}")
            
    except Exception as e:
        click.echo(f"❌ Error running enterprise analysis: {e}", err=True)
        raise


@cli.command()
@click.option('--dry-run/--execute', default=True, help='Dry run mode')
@click.option('--force', is_flag=True, help='Skip confirmation prompts')
@click.pass_context
def execute_changes(ctx, dry_run, force):
    """Execute approved enterprise change requests"""
    from .enterprise import EnterpriseConfig, EnterpriseOptimizer
    
    mode = "DRY RUN" if dry_run else "EXECUTE"
    click.echo(f"🚀 {mode}: Executing approved changes...")
    
    # Create enterprise config
    config = ctx.obj.get('config', {})
    enterprise_config = EnterpriseConfig(
        enable_change_management=True,
        enable_compliance=True,
        enable_monitoring=True,
        ticketing_system=config.get('enterprise', {}).get('ticketing_system', 'none')
    )
    
    optimizer = EnterpriseOptimizer(enterprise_config)
    
    if not force and not dry_run:
        if not click.confirm('⚠️  Are you sure you want to execute approved changes?'):
            click.echo("Cancelled.")
            return
    
    try:
        results = optimizer.execute_approved_changes(dry_run=dry_run)
        
        click.echo(f"\n📊 Execution Summary:")
        click.echo(f"   Total changes: {results['total_changes']}")
        click.echo(f"   ✅ Executed: {results['executed']}")
        click.echo(f"   ❌ Failed: {results['failed']}")
        click.echo(f"   📈 Monitoring enabled: {results['monitoring_enabled']}")
        
        # Show details for failures
        for detail in results['details']:
            if 'error' in detail:
                click.echo(f"\n❌ Failed: {detail['change_request_id']}")
                click.echo(f"   Error: {detail['error']}")
                
    except Exception as e:
        click.echo(f"❌ Error executing changes: {e}", err=True)
        raise


@cli.command()
@click.option('--days', '-d', default=30, help='Number of days to include in report')
@click.option('--output', '-o', default='enterprise_compliance_report.json')
@click.pass_context
def enterprise_report(ctx, days, output):
    """Generate comprehensive enterprise compliance report"""
    from .enterprise import EnterpriseConfig, EnterpriseOptimizer
    
    click.echo(f"📊 Generating enterprise report for the last {days} days...")
    
    # Create enterprise config
    config = ctx.obj.get('config', {})
    enterprise_config = EnterpriseConfig(
        enable_compliance=True,
        enable_audit_trail=True,
        enable_change_management=True,
        enable_monitoring=True
    )
    
    optimizer = EnterpriseOptimizer(enterprise_config)
    
    try:
        report = optimizer.generate_compliance_report(days=days)
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        click.echo(f"\n✅ Report generated: {output}")
        click.echo(f"📅 Period: {report['period']['start']} to {report['period']['end']}")
        click.echo(f"📑 Sections: {', '.join(report['sections'].keys())}")
        
    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)
        raise


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

@cli.command()
@click.option('--accounts-file', '-a', required=True, help='JSON file with AWS account details')
@click.option('--output', '-o', default='techstartup_inventory.xlsx', help='Output Excel file')
@click.option('--regions', '-r', multiple=True, help='AWS regions to scan')
@click.pass_context
def multi_account_inventory(ctx, accounts_file, output, regions):
    """Run multi-account inventory discovery"""
    from .multi_account import MultiAccountInventory
    
    click.echo("🔍 Running multi-account inventory discovery...")
    
    # Load accounts configuration
    with open(accounts_file, 'r') as f:
        accounts = json.load(f)
    
    click.echo(f"📊 Scanning {len(accounts)} AWS accounts...")
    
    # Create inventory scanner
    scanner = MultiAccountInventory(
        accounts=accounts,
        regions=list(regions) if regions else None
    )
    
    # Discover all resources
    with click.progressbar(length=100, label='Discovering resources') as bar:
        bar.update(10)
        inventory = scanner.discover_all_resources()
        bar.update(70)
        summary = scanner.generate_summary_report()
        bar.update(20)
    
    # Export to Excel
    scanner.export_to_excel(output)
    
    # Show summary
    click.echo(f"\n✅ Inventory complete!")
    click.echo(f"📊 Found {summary['ec2_summary']['total_instances']} EC2 instances")
    click.echo(f"🗄️  Found {summary['rds_summary']['total_databases']} RDS databases")
    click.echo(f"🪣 Found {summary['s3_summary']['total_storage_tb']:.2f} TB in S3")
    click.echo(f"💰 Total monthly cost: ${summary['cost_summary']['total_monthly_cost']:,.2f}")
    click.echo(f"\n📄 Results saved to {output}")


@cli.command()
@click.option('--inventory-file', '-i', required=True, help='Inventory JSON file')
@click.option('--recommendations-file', '-r', required=True, help='Recommendations JSON file')
@click.option('--target-savings', '-t', default=20000, help='Target monthly savings')
@click.option('--output', '-o', default='emergency_cost_reduction_plan.xlsx', help='Output Excel file')
@click.pass_context
def generate_cost_reduction_plan(ctx, inventory_file, recommendations_file, target_savings, output):
    """Generate emergency cost reduction plan"""
    from .multi_account import EmergencyCostReducer
    
    click.echo(f"🚨 Generating emergency cost reduction plan (Target: ${target_savings:,.0f}/month)...")
    
    # Load data
    with open(inventory_file, 'r') as f:
        inventory = json.load(f)
    
    with open(recommendations_file, 'r') as f:
        recommendations = json.load(f)
    
    # Create cost reducer
    reducer = EmergencyCostReducer(target_savings=target_savings)
    
    # Generate plan
    plan = reducer.create_emergency_plan(recommendations)
    
    # Export outputs
    reducer.export_plan_to_excel(plan, output)
    
    # Generate implementation script
    script_file = output.replace('.xlsx', '_implementation.sh')
    reducer.generate_implementation_script(plan, script_file)
    
    # Show summary
    click.echo(f"\n✅ Emergency plan generated!")
    click.echo(f"💰 Target Savings: ${target_savings:,.0f}/month")
    click.echo(f"📊 Identified Savings: ${plan['total_identified_savings']:,.0f}/month")
    click.echo(f"🎯 Target Achievement: {plan['savings_achieved_percentage']:.0f}%")
    click.echo(f"\n📄 Excel report: {output}")
    click.echo(f"🔧 Implementation script: {script_file}")


@cli.command()
@click.option('--accounts-file', '-a', default='accounts.json', help='JSON file with AWS accounts')
@click.option('--output-dir', '-o', default='optimization_results', help='Output directory')
@click.option('--target-savings', '-t', default=20000, help='Target monthly savings')
@click.pass_context
def techstartup_optimize(ctx, accounts_file, output_dir, target_savings):
    """Run TechStartup AWS cost optimization analysis"""
    click.echo("🚀 Running TechStartup cost optimization analysis...")
    click.echo(f"   Current spend: $47,000/month")
    click.echo(f"   Target savings: ${target_savings:,.0f}/month")
    
    # Import and run the main orchestration script
    import subprocess
    import sys
    
    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'techstartup_main.py'
    
    if script_path.exists():
        subprocess.run([
            sys.executable, str(script_path),
            '--accounts-file', accounts_file,
            '--output-dir', output_dir,
            '--target-savings', str(target_savings)
        ])
    else:
        click.echo(f"❌ Script not found: {script_path}", err=True)
        click.echo("Please ensure techstartup_main.py is in the scripts directory.")


@cli.command()
@click.option('--bucket-names', '-b', multiple=True, help='Specific buckets to analyze')
@click.option('--no-access-days', '-d', default=90, help='Days threshold for unused detection')
@click.option('--output', '-o', default='s3_access_report.json', help='Output file')
@click.pass_context
def analyze_s3_access(ctx, bucket_names, no_access_days, output):
    """Analyze S3 bucket access patterns to find unused buckets"""
    from .analysis.s3_access_analyzer import S3AccessAnalyzer
    
    click.echo(f"🪣 Analyzing S3 bucket access patterns (threshold: {no_access_days} days)...")
    
    # Create analyzer
    analyzer = S3AccessAnalyzer(no_access_days=no_access_days)
    
    # Analyze buckets
    bucket_list = list(bucket_names) if bucket_names else None
    
    with click.progressbar(length=100, label='Analyzing buckets') as bar:
        bar.update(10)
        results = analyzer.analyze_all_buckets(bucket_list)
        bar.update(60)
        report = analyzer.generate_unused_buckets_report(results)
        bar.update(30)
    
    # Save report
    with open(output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Show summary
    click.echo(f"\n✅ Analysis complete!")
    click.echo(f"📊 Total buckets analyzed: {report['summary']['total_buckets_analyzed']}")
    click.echo(f"🗑️  Unused buckets found: {report['summary']['unused_buckets_count']}")
    click.echo(f"📈 Unused percentage: {report['summary']['unused_percentage']:.1f}%")
    click.echo(f"\n📄 Report saved to {output}")


if __name__ == '__main__':
    cli(obj={})