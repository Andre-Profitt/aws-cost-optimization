#!/usr/bin/env python3
"""
Example script demonstrating S3 Discovery with Excel reporting
"""
import sys
import os
import boto3
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aws_cost_optimizer.discovery.s3_discovery import S3Discovery
from aws_cost_optimizer.reporting.excel_reporter import ExcelReporter
from aws_cost_optimizer.optimization.s3_optimizer import S3Optimizer


def main():
    """Run S3 discovery and generate Excel report"""
    
    # Initialize AWS session
    session = boto3.Session()
    
    print("Starting S3 Discovery...")
    
    # Step 1: Discover S3 resources
    discovery = S3Discovery(session)
    discovered_resources = discovery.discover_all_resources()
    
    # Get summary
    summary = discovery.get_summary()
    print(f"\nDiscovery Summary:")
    print(f"Total Buckets: {summary['total_buckets']}")
    print(f"Total Size: {summary['total_size_gb']:.2f} GB")
    print(f"Total Objects: {summary['total_objects']:,}")
    print(f"Estimated Monthly Cost: ${summary['total_estimated_monthly_cost']:.2f}")
    
    # Get optimization opportunities
    opportunities = discovery.get_optimization_opportunities()
    print(f"\nFound {len(opportunities)} optimization opportunities")
    
    # Export raw inventory
    discovery.export_inventory('s3_inventory.json')
    
    # Step 2: Run S3 Optimizer for detailed recommendations
    print("\nRunning S3 Optimizer...")
    optimizer = S3Optimizer(session=session)
    recommendations = optimizer.analyze_all_buckets()
    
    print(f"Generated {len(recommendations)} optimization recommendations")
    
    # Step 3: Generate comprehensive Excel report
    print("\nGenerating Excel report...")
    reporter = ExcelReporter()
    
    # Prepare optimization result for the reporter
    optimization_result = {
        'total_monthly_savings': sum(rec.estimated_monthly_savings for rec in recommendations),
        'total_annual_savings': sum(rec.estimated_monthly_savings for rec in recommendations) * 12,
        'recommendations_count': len(recommendations),
        'anomalies_detected': 0,  # Would come from anomaly detector
        'auto_remediation_tasks': 0,  # Would come from auto-remediation engine
        'execution_time': 0,  # Would track actual execution time
        'details': {
            's3': {
                'total_monthly_savings': sum(rec.estimated_monthly_savings for rec in recommendations),
                'recommendations': [
                    {
                        'resource_id': rec.bucket_name,
                        'resource_type': 'S3 Bucket',
                        'action': rec.action,
                        'monthly_savings': rec.estimated_monthly_savings,
                        'risk_level': rec.risk_level,
                        'confidence': rec.confidence,
                        'reason': rec.reason,
                        'current_storage_class': rec.current_storage_class,
                        'recommended_storage_class': rec.recommended_storage_class,
                        'implementation_steps': rec.implementation_steps,
                        'impact': rec.impact
                    }
                    for rec in recommendations
                ],
                'discovery_data': discovered_resources,
                'summary': summary,
                'opportunities': opportunities
            }
        }
    }
    
    # Generate the Excel report
    output_file = f"s3_cost_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    reporter.generate_comprehensive_report(optimization_result, output_file)
    
    print(f"\nReport generated: {output_file}")
    
    # Also generate simple S3-only report
    simple_output = f"s3_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    reporter.generate_simple_report(
        optimization_result['details']['s3']['recommendations'],
        'S3 Storage',
        simple_output
    )
    
    print(f"Simple S3 report generated: {simple_output}")
    
    # Generate CLI commands for implementation
    optimizer.generate_cli_commands(recommendations, 's3_optimization_commands.sh')
    print("CLI commands generated: s3_optimization_commands.sh")
    
    # Print top recommendations
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec.bucket_name}")
        print(f"   Action: {rec.action}")
        print(f"   Monthly Savings: ${rec.estimated_monthly_savings:,.2f}")
        print(f"   Risk: {rec.risk_level}")
        print(f"   Reason: {rec.reason}")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()