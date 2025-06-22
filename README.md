# AWS Cost Optimizer

A comprehensive tool for identifying and implementing AWS cost optimization opportunities across your infrastructure.

## Features

- **EC2 Optimization**: Identify idle instances, rightsizing opportunities
- **S3 Storage Analysis**: Optimize storage classes and lifecycle policies
- **EBS Volume Management**: Find unattached and underutilized volumes
- **Reserved Instance Recommendations**: Analyze usage patterns for RI opportunities
- **Cost Anomaly Detection**: Identify unusual spending patterns
- **Multi-Account Support**: Analyze costs across your AWS organization
- **Safety Features**: Built-in safety checks and dry-run mode

## Quick Start

```bash
# Install the tool
pip install -e .

# Run your first analysis
python -m aws_cost_optimizer analyze

# Generate a detailed report
python -m aws_cost_optimizer analyze --format excel --output cost_report
```

See the full documentation in the `docs/` directory for detailed usage instructions.