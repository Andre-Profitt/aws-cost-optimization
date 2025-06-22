#!/usr/bin/env python3
"""
Example usage of AWS Cost Optimizer

This script demonstrates how to use the AWS Cost Optimizer programmatically
"""

from aws_cost_optimizer.optimization.ec2_optimizer import EC2Optimizer
from aws_cost_optimizer.optimization.s3_optimizer import S3Optimizer
from aws_cost_optimizer.discovery.s3_discovery import S3Discovery
from aws_cost_optimizer.reporting.excel_reporter import ExcelReporter
from aws_cost_optimizer.patterns.pattern_detector import PatternDetector
from aws_cost_optimizer.safety.safety_checker import SafetyChecker
import json
from datetime import datetime


def main():
    # Initialize components
    print("Initializing AWS Cost Optimizer...")
    
    # Create pattern detector and safety checker
    pattern_detector = PatternDetector()
    safety_checker = SafetyChecker()
    
    # Create EC2 optimizer with custom thresholds
    ec2_optimizer = EC2Optimizer(
        cpu_threshold=10.0,      # Consider instances with <10% CPU as idle
        network_threshold=5.0,   # Consider <5 MB/day network as low activity
        observation_days=14      # Look at last 14 days of metrics
    )
    
    # Set the pattern detector and safety checker
    ec2_optimizer.set_pattern_detector(pattern_detector)
    ec2_optimizer.set_safety_checker(safety_checker)
    
    # Analyze specific regions (or None for all regions)
    regions = ['us-east-1', 'us-west-2']
    
    print(f"Analyzing EC2 instances in regions: {', '.join(regions)}")
    recommendations = ec2_optimizer.analyze(regions)
    
    # Process results
    total_monthly_savings = 0
    total_recommendations = 0
    
    for region, recs in recommendations.items():
        if recs:
            print(f"\n=== Region: {region} ===")
            print(f"Found {len(recs)} optimization opportunities:")
            
            for rec in recs:
                total_recommendations += 1
                total_monthly_savings += rec.monthly_savings
                
                print(f"\n- Instance: {rec.instance_id} ({rec.instance_type})")
                print(f"  Action: {rec.action}")
                print(f"  Reason: {rec.reason}")
                print(f"  Monthly Savings: ${rec.monthly_savings:,.2f}")
                print(f"  Risk Level: {rec.risk_level}")
                
                # Show tags
                if rec.tags:
                    print(f"  Tags: {', '.join(f'{k}={v}' for k, v in rec.tags.items())}")
                
                # Show implementation steps
                if rec.implementation_steps:
                    print("  Implementation Steps:")
                    for i, step in enumerate(rec.implementation_steps, 1):
                        print(f"    {step}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total recommendations: {total_recommendations}")
    print(f"Total potential monthly savings: ${total_monthly_savings:,.2f}")
    print(f"Total potential annual savings: ${total_monthly_savings * 12:,.2f}")
    
    # Save recommendations to file
    if recommendations:
        output_file = "ec2_recommendations.json"
        
        # Convert to JSON-serializable format
        json_data = {}
        for region, recs in recommendations.items():
            json_data[region] = [
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
                    'metrics': rec.metrics,
                    'created_at': rec.created_at.isoformat() if rec.created_at else None
                }
                for rec in recs
            ]
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nRecommendations saved to: {output_file}")
    
    # Example: Analyze cost patterns
    print("\n=== Cost Pattern Analysis ===")
    try:
        cost_patterns = pattern_detector.analyze_cost_patterns('current', lookback_days=30)
        
        if cost_patterns:
            print(f"Total spend (30 days): ${cost_patterns.get('total_spend', 0):,.2f}")
            print(f"Average daily spend: ${cost_patterns.get('average_daily_spend', 0):,.2f}")
            
            if cost_patterns.get('anomalies'):
                print(f"\nFound {len(cost_patterns['anomalies'])} cost anomalies:")
                for anomaly in cost_patterns['anomalies'][:5]:  # Show top 5
                    print(f"  - Day {anomaly['day_index']}: ${anomaly['cost']:.2f} "
                          f"(Z-score: {anomaly['z_score']:.2f}, Severity: {anomaly['severity']})")
    except Exception as e:
        print(f"Could not analyze cost patterns: {e}")
    
    # Example: S3 Optimization
    print("\n=== S3 Storage Optimization ===")
    try:
        # Initialize S3 components
        s3_discovery = S3Discovery()
        s3_optimizer = S3Optimizer()
        
        # Discover S3 resources
        print("Discovering S3 buckets...")
        s3_resources = s3_discovery.discover_all_resources()
        summary = s3_discovery.get_summary()
        
        print(f"Found {summary['total_buckets']} buckets")
        print(f"Total storage: {summary['total_size_gb']:.2f} GB")
        print(f"Estimated monthly cost: ${summary['total_estimated_monthly_cost']:.2f}")
        
        # Get optimization recommendations
        s3_recommendations = s3_optimizer.analyze_all_buckets()
        
        if s3_recommendations:
            print(f"\nFound {len(s3_recommendations)} S3 optimization opportunities:")
            s3_monthly_savings = sum(rec.estimated_monthly_savings for rec in s3_recommendations[:5])
            
            for rec in s3_recommendations[:5]:  # Show top 5
                print(f"\n- Bucket: {rec.bucket_name}")
                print(f"  Action: {rec.action}")
                print(f"  Monthly Savings: ${rec.estimated_monthly_savings:,.2f}")
                print(f"  Risk: {rec.risk_level}")
                print(f"  Reason: {rec.reason}")
            
            print(f"\nTotal S3 monthly savings potential: ${s3_monthly_savings:,.2f}")
            
            # Generate Excel report
            print("\n=== Generating Excel Report ===")
            excel_reporter = ExcelReporter()
            
            # Prepare combined optimization results
            optimization_result = {
                'timestamp': datetime.now(),
                'total_monthly_savings': total_monthly_savings + s3_monthly_savings,
                'total_annual_savings': (total_monthly_savings + s3_monthly_savings) * 12,
                'recommendations_count': total_recommendations + len(s3_recommendations),
                'anomalies_detected': len(cost_patterns.get('anomalies', [])) if cost_patterns else 0,
                'auto_remediation_tasks': 0,
                'execution_time': 0,
                'details': {
                    'ec2': {
                        'total_monthly_savings': total_monthly_savings,
                        'recommendations': []  # Would convert EC2 recommendations here
                    },
                    's3': {
                        'total_monthly_savings': s3_monthly_savings,
                        'recommendations': [
                            {
                                'resource_id': rec.bucket_name,
                                'action': rec.action,
                                'monthly_savings': rec.estimated_monthly_savings,
                                'risk_level': rec.risk_level,
                                'confidence': rec.confidence,
                                'reason': rec.reason
                            }
                            for rec in s3_recommendations
                        ]
                    }
                }
            }
            
            # Generate report
            report_filename = f"aws_cost_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_reporter.generate_comprehensive_report(optimization_result, report_filename)
            print(f"Excel report generated: {report_filename}")
            
    except Exception as e:
        print(f"Could not analyze S3 storage: {e}")


if __name__ == "__main__":
    main()