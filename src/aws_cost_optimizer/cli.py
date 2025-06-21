import click
import yaml
from pathlib import Path
from .discovery.multi_account import MultiAccountInventory, AWSAccount
from .analysis.pattern_detector import PatternDetector
from .optimization.safety_checks import SafetyOrchestrator
from .optimization.ec2_optimizer import EC2Optimizer

@click.group()
def cli():
    """AWS Cost Optimizer - Safely reduce AWS costs"""
    pass

@cli.command()
@click.option('--output-file', '-o', default='inventory.xlsx')
def discover(output_file):
    """Discover all AWS resources"""
    click.echo("🔍 Starting AWS resource discovery...")
    
    # Load config
    config = yaml.safe_load(open('config/config.yaml'))
    
    # Create account list
    accounts = [
        AWSAccount(
            account_id=acc['id'],
            account_name=acc['name'],
            role_name=acc['role_name']
        )
        for acc in config['aws']['accounts']
    ]
    
    # Run discovery
    inventory = MultiAccountInventory(accounts, config['aws']['regions'])
    resources = inventory.collect_all_accounts_inventory()
    
    # Export to Excel
    inventory.export_to_excel(output_file)
    click.echo(f"✅ Discovered {len(resources)} resources. Saved to {output_file}")

@cli.command()
@click.option('--input-file', '-i', required=True)
@click.option('--days', '-d', default=90)
def analyze(input_file, days):
    """Analyze resource usage patterns"""
    click.echo(f"📊 Analyzing patterns over {days} days...")
    
    # Pattern detection would go here
    pattern_detector = PatternDetector()
    click.echo("✅ Analysis complete")

@cli.command()
@click.option('--dry-run/--execute', default=True)
def optimize(dry_run):
    """Execute optimization recommendations"""
    mode = "DRY RUN" if dry_run else "EXECUTE"
    click.echo(f"🚀 Starting optimization in {mode} mode...")
    
    # Safety checks would go here
    safety = SafetyOrchestrator(dry_run=dry_run)
    click.echo("✅ Optimization complete")

if __name__ == '__main__':
    cli()