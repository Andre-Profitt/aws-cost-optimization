# Getting Started with AWS Cost Optimizer

## Prerequisites

Before you begin, ensure you have:

1. **Python 3.8+** installed
2. **AWS CLI** configured with appropriate credentials
3. **IAM permissions** for cost optimization (see below)
4. **Docker** (optional, for containerized deployment)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/aws-cost-optimizer.git
cd aws-cost-optimizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Configure the Tool

Create your configuration file:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` with your AWS account details and preferences.

### 2. Run Your First Analysis

```bash
# Analyze all resources
python -m aws_cost_optimizer analyze

# Analyze specific service
python -m aws_cost_optimizer quick-scan --type ec2

# Generate detailed report
python -m aws_cost_optimizer analyze --format excel --output my_report
```

## Required IAM Permissions

Create an IAM policy with the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:Describe*",
                "rds:Describe*",
                "s3:List*",
                "s3:GetBucket*",
                "cloudwatch:GetMetricStatistics",
                "ce:GetCostAndUsage",
                "ce:GetReservationUtilization",
                "ce:GetSavingsPlansPurchaseRecommendation"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:StopInstances",
                "ec2:ModifyInstanceAttribute",
                "rds:StopDBInstance",
                "s3:PutBucketLifecycleConfiguration"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "aws:ResourceTag/OptimizationEnabled": "true"
                }
            }
        }
    ]
}
```

## Configuration Options

### Basic Configuration

```yaml
aws:
  regions:
    - us-east-1
    - us-west-2

optimization:
  ec2:
    cpu_threshold: 10  # CPU % threshold for idle detection
    observation_days: 14

safety:
  dry_run: true  # Always start with dry run!
  require_approval: true
```

### Multi-Account Setup

For organizations with multiple AWS accounts:

```bash
# Set up cross-account roles
./scripts/setup_cross_account_roles.sh
```

## Common Use Cases

### 1. Find and Stop Idle EC2 Instances

```bash
python -m aws_cost_optimizer quick-scan --type ec2 --region us-east-1
```

### 2. Optimize S3 Storage

```bash
python -m aws_cost_optimizer quick-scan --type s3
```

### 3. Get Reserved Instance Recommendations

```bash
python -m aws_cost_optimizer quick-scan --type ri
```

### 4. Detect Cost Anomalies

```bash
python -m aws_cost_optimizer quick-scan --type anomalies
```

## Safety Features

The tool includes multiple safety mechanisms:

1. **Dry Run Mode**: Default mode shows what would be done without making changes
2. **Safety Checks**: Validates resources before modifications
3. **Rollback Capability**: Emergency rollback script included
4. **Approval Workflow**: Requires explicit approval for changes

## Best Practices

1. **Always start with dry run mode**
   ```bash
   python -m aws_cost_optimizer remediate --dry-run
   ```

2. **Tag resources appropriately**
   - Use `OptimizationEnabled: true` for resources that can be optimized
   - Use `DoNotOptimize: true` for critical resources

3. **Review recommendations before applying**
   - Export to Excel for team review
   - Validate with application owners

4. **Monitor after optimization**
   - Set up CloudWatch alarms
   - Use the anomaly detector

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure IAM role has required permissions
   - Check AWS CLI configuration

2. **No Resources Found**
   - Verify regions in configuration
   - Check resource tags

3. **High CPU Usage**
   - Reduce parallel workers in config
   - Process fewer regions at once

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m aws_cost_optimizer analyze
```

## Next Steps

1. Read the [API Reference](api_reference.md)
2. Review [Best Practices](best_practices.md)
3. Set up [Automated Workflows](.github/workflows/daily_discovery.yml)

## Support

- Create an issue on GitHub
- Email: aws-cost-optimizer@your-org.com
- Slack: #aws-cost-optimization
