#!/usr/bin/env python3
"""
TechStartup AWS Cost Optimization - Main Orchestration Script
Solves the practice problem: Find $20K/month savings from $47K/month spend
"""
import os
import sys
import json
import click
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws_cost_optimizer.optimization.s3_optimizer import S3Optimizer
from aws_cost_optimizer.optimization.rds_optimizer import RDSOptimizer
from aws_cost_optimizer.optimization.ec2_optimizer import EC2Optimizer
from aws_cost_optimizer.multi_account.inventory import MultiAccountInventory
from aws_cost_optimizer.multi_account.cost_reducer import EmergencyCostReducer
from aws_cost_optimizer.analysis.s3_access_analyzer import S3AccessAnalyzer, integrate_with_s3_optimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechStartupCostOptimizer:
    """Main orchestrator for TechStartup cost optimization"""
    
    def __init__(self, accounts_config: str, output_dir: str = "optimization_results"):
        self.accounts_config = accounts_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.inventory = None
        self.recommendations = {
            'ec2': [],
            'rds': [],
            's3': [],
            'summary': {}
        }
        
    def run_full_analysis(self):
        """Run complete cost optimization analysis"""
        logger.info("="*60)
        logger.info("TECHSTARTUP AWS COST OPTIMIZATION ANALYSIS")
        logger.info("="*60)
        logger.info(f"Started at: {datetime.now()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Multi-account inventory discovery
        logger.info("\n📊 STEP 1: Discovering resources across all accounts...")
        self.run_inventory_discovery()
        
        # Step 2: Analyze EC2 instances for optimization
        logger.info("\n🖥️  STEP 2: Analyzing EC2 instances...")
        self.analyze_ec2_instances()
        
        # Step 3: Analyze RDS databases
        logger.info("\n🗄️  STEP 3: Analyzing RDS databases...")
        self.analyze_rds_databases()
        
        # Step 4: Analyze S3 buckets with access patterns
        logger.info("\n🪣 STEP 4: Analyzing S3 buckets and access patterns...")
        self.analyze_s3_buckets()
        
        # Step 5: Generate emergency cost reduction plan
        logger.info("\n🚨 STEP 5: Generating emergency cost reduction plan...")
        self.generate_cost_reduction_plan()
        
        # Step 6: Create executive reports
        logger.info("\n📈 STEP 6: Creating executive reports...")
        self.create_executive_reports()
        
        logger.info("\n✅ Analysis complete!")
        self.print_summary()
    
    def run_inventory_discovery(self):
        """Run multi-account inventory discovery"""
        with open(self.accounts_config, 'r') as f:
            accounts = json.load(f)
        
        # Create inventory scanner
        scanner = MultiAccountInventory(accounts=accounts)
        
        # Discover all resources
        self.inventory = scanner.discover_all_resources()
        
        # Generate summary
        summary = scanner.generate_summary_report()
        
        # Save results
        inventory_file = self.output_dir / f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(inventory_file, 'w') as f:
            json.dump(self.inventory, f, indent=2, default=str)
        
        # Export to Excel
        excel_file = self.output_dir / f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        scanner.export_to_excel(str(excel_file))
        
        logger.info(f"✓ Found {summary['ec2_summary']['total_instances']} EC2 instances")
        logger.info(f"✓ Found {summary['rds_summary']['total_databases']} RDS databases")
        logger.info(f"✓ Found {summary['s3_summary']['total_storage_tb']:.2f} TB in S3")
        logger.info(f"✓ Total monthly cost: ${summary['cost_summary']['total_monthly_cost']:,.2f}")
    
    def analyze_ec2_instances(self):
        """Analyze EC2 instances for optimization opportunities"""
        if not self.inventory:
            logger.error("No inventory data available")
            return
        
        # Create EC2 optimizer with aggressive settings for finding savings
        ec2_optimizer = EC2Optimizer(
            cpu_threshold=5.0,  # Very low threshold as per problem
            lookback_days=30  # 30 days as specified
        )
        
        # Convert inventory to format expected by optimizer
        instances_by_region = {}
        for instance in self.inventory['ec2_instances']:
            if instance['state'] == 'running':
                region = instance['region']
                if region not in instances_by_region:
                    instances_by_region[region] = []
                instances_by_region[region].append(instance)
        
        # Analyze instances
        all_recommendations = []
        for region, instances in instances_by_region.items():
            logger.info(f"  Analyzing {len(instances)} instances in {region}...")
            
            # In real implementation, would call the actual analyze method
            # For now, simulate recommendations based on the problem description
            recommendations = ec2_optimizer.analyze_all_instances(regions=[region])
            all_recommendations.extend(recommendations)
        
        self.recommendations['ec2'] = all_recommendations
        
        # Generate summary
        summary = ec2_optimizer.generate_savings_summary(all_recommendations)
        
        logger.info(f"✓ Generated {len(all_recommendations)} EC2 recommendations")
        logger.info(f"✓ Potential EC2 savings: ${summary['total_monthly_savings']:,.2f}/month")
    
    def analyze_rds_databases(self):
        """Analyze RDS databases for optimization"""
        if not self.inventory:
            return
        
        # Create RDS optimizer
        rds_optimizer = RDSOptimizer(
            lookback_days=7,
            cpu_threshold=25.0
        )
        
        # Get unique regions from inventory
        regions = list(set(db['region'] for db in self.inventory['rds_databases']))
        
        # Analyze databases
        all_recommendations = []
        for region in regions:
            logger.info(f"  Analyzing RDS instances in {region}...")
            recommendations = rds_optimizer.analyze_all_databases(region_name=region)
            all_recommendations.extend(recommendations)
        
        self.recommendations['rds'] = all_recommendations
        
        # Calculate total savings
        total_savings = sum(rec.estimated_monthly_savings for rec in all_recommendations)
        
        logger.info(f"✓ Generated {len(all_recommendations)} RDS recommendations")
        logger.info(f"✓ Potential RDS savings: ${total_savings:,.2f}/month")
    
    def analyze_s3_buckets(self):
        """Analyze S3 buckets including access patterns"""
        if not self.inventory:
            return
        
        # Create S3 optimizer with access analyzer
        s3_optimizer = S3Optimizer()
        
        # Integrate access analyzer for 90+ day detection
        integrate_with_s3_optimizer(s3_optimizer, no_access_days=90)
        
        # Create access analyzer for standalone analysis
        access_analyzer = S3AccessAnalyzer(no_access_days=90)
        
        # Analyze all buckets
        logger.info(f"  Analyzing {len(self.inventory['s3_buckets'])} S3 buckets...")
        
        # Get bucket names
        bucket_names = [b['bucket_name'] for b in self.inventory['s3_buckets']]
        
        # Run S3 optimization analysis
        s3_recommendations = s3_optimizer.analyze_all_buckets()
        
        # Run access analysis
        logger.info("  Checking for buckets with no access in 90+ days...")
        access_results = access_analyzer.analyze_all_buckets(bucket_names[:10])  # Limit for performance
        
        # Generate unused buckets report
        unused_report = access_analyzer.generate_unused_buckets_report(access_results)
        
        self.recommendations['s3'] = s3_recommendations
        self.recommendations['s3_unused'] = unused_report
        
        # Calculate savings
        total_savings = sum(rec.estimated_monthly_savings for rec in s3_recommendations)
        
        logger.info(f"✓ Generated {len(s3_recommendations)} S3 recommendations")
        logger.info(f"✓ Found {unused_report['summary']['unused_buckets_count']} unused buckets")
        logger.info(f"✓ Potential S3 savings: ${total_savings:,.2f}/month")
    
    def generate_cost_reduction_plan(self):
        """Generate emergency cost reduction plan"""
        # Create cost reducer
        reducer = EmergencyCostReducer(target_savings=20000)
        
        # Convert recommendations to format expected by reducer
        formatted_recommendations = {
            'ec2': [self._format_ec2_rec(rec) for rec in self.recommendations.get('ec2', [])],
            'rds': [self._format_rds_rec(rec) for rec in self.recommendations.get('rds', [])],
            's3': [self._format_s3_rec(rec) for rec in self.recommendations.get('s3', [])]
        }
        
        # Generate plan
        plan = reducer.create_emergency_plan(formatted_recommendations)
        
        # Save plan
        plan_file = self.output_dir / "emergency_cost_reduction_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        
        # Export to Excel
        excel_file = self.output_dir / "emergency_cost_reduction_plan.xlsx"
        reducer.export_plan_to_excel(plan, str(excel_file))
        
        # Generate implementation script
        script_file = self.output_dir / "immediate_actions.sh"
        reducer.generate_implementation_script(plan, str(script_file))
        os.chmod(script_file, 0o755)  # Make executable
        
        logger.info(f"✓ Target savings: ${plan['target_savings']:,.2f}")
        logger.info(f"✓ Identified savings: ${plan['total_identified_savings']:,.2f}")
        logger.info(f"✓ Achievement: {plan['savings_achieved_percentage']:.0f}%")
    
    def create_executive_reports(self):
        """Create executive-ready reports"""
        # AWS Cost Explorer style report
        report = {
            'report_date': datetime.now().isoformat(),
            'account_summary': {
                'total_accounts': 4,
                'total_monthly_spend': 47000,
                'identified_waste': 0,
                'optimization_potential': 0
            },
            'service_breakdown': {
                'EC2': {'current_cost': 0, 'optimized_cost': 0, 'savings': 0},
                'RDS': {'current_cost': 0, 'optimized_cost': 0, 'savings': 0},
                'S3': {'current_cost': 0, 'optimized_cost': 0, 'savings': 0}
            },
            'recommendations_summary': [],
            'implementation_roadmap': {
                'week_1': [],
                'week_2': [],
                'week_3_4': []
            }
        }
        
        # Calculate totals from recommendations
        for service in ['ec2', 'rds', 's3']:
            if service in self.recommendations:
                savings = sum(getattr(rec, 'monthly_savings', 0) or 
                            getattr(rec, 'estimated_monthly_savings', 0) 
                            for rec in self.recommendations[service])
                report['account_summary']['identified_waste'] += savings
        
        report['account_summary']['optimization_potential'] = report['account_summary']['identified_waste']
        
        # Create PowerBI/Tableau ready CSV
        metrics_data = []
        
        # Add EC2 metrics
        for rec in self.recommendations.get('ec2', []):
            metrics_data.append({
                'Date': datetime.now().date(),
                'Service': 'EC2',
                'Resource_ID': getattr(rec, 'instance_id', 'Unknown'),
                'Resource_Type': getattr(rec, 'instance_type', 'Unknown'),
                'Current_Cost': getattr(rec, 'current_cost', 0),
                'Potential_Savings': getattr(rec, 'monthly_savings', 0),
                'Action': getattr(rec, 'action', 'Unknown'),
                'Risk_Level': getattr(rec, 'risk_level', 'Unknown')
            })
        
        # Save executive report
        report_file = self.output_dir / "executive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save metrics CSV
        if metrics_data:
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            csv_file = self.output_dir / "cost_optimization_metrics.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info("✓ Created executive report")
        logger.info("✓ Created metrics CSV for BI tools")
    
    def _format_ec2_rec(self, rec):
        """Format EC2 recommendation for cost reducer"""
        return {
            'instance_id': getattr(rec, 'instance_id', 'unknown'),
            'instance_type': getattr(rec, 'instance_type', 'unknown'),
            'action': getattr(rec, 'action', 'unknown'),
            'reason': getattr(rec, 'reason', 'unknown'),
            'monthly_savings': getattr(rec, 'monthly_savings', 0),
            'risk_level': getattr(rec, 'risk_level', 'medium'),
            'environment': getattr(rec, 'tags', {}).get('Environment', 'unknown')
        }
    
    def _format_rds_rec(self, rec):
        """Format RDS recommendation for cost reducer"""
        return {
            'instance_identifier': getattr(rec, 'instance_identifier', 'unknown'),
            'action': getattr(rec, 'action', 'unknown'),
            'reason': getattr(rec, 'reason', 'unknown'),
            'estimated_monthly_savings': getattr(rec, 'estimated_monthly_savings', 0),
            'risk_level': getattr(rec, 'risk_level', 'medium'),
            'environment': 'unknown'  # Would come from tags in real implementation
        }
    
    def _format_s3_rec(self, rec):
        """Format S3 recommendation for cost reducer"""
        return {
            'bucket_name': getattr(rec, 'bucket_name', 'unknown'),
            'action': getattr(rec, 'action', 'unknown'),
            'reason': getattr(rec, 'reason', 'unknown'),
            'estimated_monthly_savings': getattr(rec, 'estimated_monthly_savings', 0),
            'risk_level': getattr(rec, 'risk_level', 'low')
        }
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("TECHSTARTUP COST OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Current Monthly Spend: $47,000")
        print(f"Target Savings: $20,000/month")
        
        total_savings = 0
        for service in ['ec2', 'rds', 's3']:
            if service in self.recommendations:
                savings = sum(getattr(rec, 'monthly_savings', 0) or 
                            getattr(rec, 'estimated_monthly_savings', 0) 
                            for rec in self.recommendations[service])
                total_savings += savings
                print(f"{service.upper()} Savings: ${savings:,.0f}/month")
        
        print(f"\nTotal Identified Savings: ${total_savings:,.0f}/month")
        print(f"Target Achievement: {(total_savings/20000*100):.0f}%")
        
        print(f"\nReports saved to: {self.output_dir}")
        print("\nNext Steps:")
        print("1. Review emergency_cost_reduction_plan.xlsx with CFO")
        print("2. Execute immediate_actions.sh for quick wins")
        print("3. Schedule follow-up for Phase 2 optimizations")
        print("="*60)


@click.command()
@click.option('--accounts-file', default='accounts.json', help='JSON file with AWS accounts')
@click.option('--output-dir', default='optimization_results', help='Output directory')
@click.option('--target-savings', default=20000, help='Target monthly savings')
def main(accounts_file, output_dir, target_savings):
    """Run TechStartup AWS cost optimization analysis"""
    
    # Verify accounts file exists
    if not os.path.exists(accounts_file):
        # Create sample accounts file
        sample_accounts = [
            {
                "account_id": "123456789012",
                "account_name": "TechStartup-Production",
                "role_name": "OrganizationCostOptimizerRole"
            },
            {
                "account_id": "123456789013",
                "account_name": "TechStartup-Development",
                "role_name": "OrganizationCostOptimizerRole"
            },
            {
                "account_id": "123456789014",
                "account_name": "TechStartup-Testing",
                "role_name": "OrganizationCostOptimizerRole"
            },
            {
                "account_id": "123456789015",
                "account_name": "TechStartup-Sandbox",
                "role_name": "OrganizationCostOptimizerRole"
            }
        ]
        
        with open(accounts_file, 'w') as f:
            json.dump(sample_accounts, f, indent=2)
        
        logger.info(f"Created sample accounts file: {accounts_file}")
        logger.info("Please update with your actual AWS account IDs and run again.")
        return
    
    # Create and run optimizer
    optimizer = TechStartupCostOptimizer(accounts_file, output_dir)
    optimizer.run_full_analysis()


if __name__ == '__main__':
    main()