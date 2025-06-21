#!/usr/bin/env python
"""
Main optimization runner script - coordinates all optimization modules
"""
import click
import logging
from datetime import datetime
from aws_cost_optimizer.optimization.s3_optimizer import S3Optimizer
from aws_cost_optimizer.optimization.rds_optimizer import RDSOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--component', type=click.Choice(['all', 's3', 'rds']), default='all',
              help='Which component to optimize')
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
@click.option('--output-dir', default='reports', help='Directory for output reports')
@click.option('--region', default=None, help='AWS region to analyze')
def main(component, dry_run, output_dir, region):
    """Run AWS cost optimization analysis"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if component in ['all', 's3']:
        logger.info("Starting S3 optimization analysis...")
        s3_optimizer = S3Optimizer()
        s3_recommendations = s3_optimizer.analyze_all_buckets(region_name=region)
        
        if s3_recommendations:
            output_file = f"{output_dir}/s3_optimization_{timestamp}.xlsx"
            s3_optimizer.export_recommendations(s3_recommendations, output_file)
            s3_optimizer.generate_cli_commands(s3_recommendations, 
                                              f"{output_dir}/s3_commands_{timestamp}.sh")
            
            total_s3_savings = sum(rec.estimated_monthly_savings for rec in s3_recommendations)
            logger.info(f"S3 Optimization: Found {len(s3_recommendations)} recommendations")
            logger.info(f"Potential S3 savings: ${total_s3_savings:,.2f}/month")
    
    if component in ['all', 'rds']:
        logger.info("Starting RDS optimization analysis...")
        rds_optimizer = RDSOptimizer()
        rds_recommendations = rds_optimizer.analyze_all_databases(region_name=region)
        
        if rds_recommendations:
            output_file = f"{output_dir}/rds_optimization_{timestamp}.xlsx"
            rds_optimizer.export_recommendations(rds_recommendations, output_file)
            
            total_rds_savings = sum(rec.estimated_monthly_savings for rec in rds_recommendations)
            logger.info(f"RDS Optimization: Found {len(rds_recommendations)} recommendations")
            logger.info(f"Potential RDS savings: ${total_rds_savings:,.2f}/month")
    
    if component == 'all':
        total_savings = 0
        if 's3_recommendations' in locals():
            total_savings += sum(rec.estimated_monthly_savings for rec in s3_recommendations)
        if 'rds_recommendations' in locals():
            total_savings += sum(rec.estimated_monthly_savings for rec in rds_recommendations)
        
        logger.info(f"\nTotal potential savings: ${total_savings:,.2f}/month (${total_savings * 12:,.2f}/year)")
        logger.info(f"Reports saved to {output_dir}/")

if __name__ == "__main__":
    main()