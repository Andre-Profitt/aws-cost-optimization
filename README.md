# AWS Cost Optimizer

A comprehensive tool for identifying and implementing AWS cost optimization opportunities across your infrastructure.

## Features

### Core Optimization Features
- **EC2 Optimization**: Identify idle instances, rightsizing opportunities
- **S3 Storage Analysis**: Optimize storage classes and lifecycle policies
- **EBS Volume Management**: Find unattached and underutilized volumes
- **Reserved Instance Recommendations**: Analyze usage patterns for RI opportunities
- **Cost Anomaly Detection**: Identify unusual spending patterns
- **Multi-Account Support**: Analyze costs across your AWS organization
- **Safety Features**: Built-in safety checks and dry-run mode

### Enhanced RDS Optimization (NEW)
- **Comprehensive RDS Analysis**: Deep analysis of database utilization patterns
- **Multi-Region Support**: Analyze RDS instances across all AWS regions
- **Advanced Recommendations**:
  - Instance rightsizing based on CPU, memory, and connection metrics
  - Reserved Instance purchase opportunities
  - Aurora migration recommendations for compatible workloads
  - Storage optimization (gp2 to gp3 migration)
  - Development/test database scheduling
  - Backup retention optimization
  - Graviton instance migration opportunities
- **Snapshot Management**: Identify and clean up old manual snapshots
- **Risk Assessment**: Each recommendation includes risk level and rollback plans
- **Implementation Guidance**: Step-by-step CLI commands for each optimization

### Advanced Pattern Detection (NEW)
- **Workload Characterization**: Automatically classify workloads (steady-state, batch, spiky, seasonal)
- **Usage Pattern Analysis**: 
  - Daily, weekly, and monthly usage patterns
  - Business hours vs off-hours analysis
  - Weekend vs weekday usage ratios
- **Predictive Analytics**:
  - Seasonality scoring
  - Variability analysis
  - Predictability scoring for RI suitability
- **Business Context Inference**: Automatically determine environment type and criticality
- **Optimization Scoring**: Cost optimization score for workload groups
- **Architectural Recommendations**: Workload-specific architectural improvements

### Enhanced Utilities (NEW)
- **AWS Helper Functions**: Simplified AWS API interactions with multi-region support
- **Configuration Management**: YAML-based configuration with sensible defaults
- **Advanced Cost Calculations**: Accurate pricing estimates for EC2, RDS, EBS, and S3
- **Tag Analysis**: Smart tag parsing for environment detection and resource grouping
- **Report Generation**: Excel, CSV, and JSON report formats with auto-formatting
- **Caching System**: Performance optimization with TTL-based caching
- **Progress Tracking**: Visual progress bars for long-running operations

## Quick Start

```bash
# Install the tool
pip install -e .

# Run your first analysis
python -m aws_cost_optimizer analyze

# Generate a detailed report
python -m aws_cost_optimizer analyze --format excel --output cost_report
```

### RDS Optimization Examples

```python
from aws_cost_optimizer.optimization import RDSOptimizer

# Initialize the optimizer
optimizer = RDSOptimizer(
    cpu_threshold=20.0,
    lookback_days=14
)

# Analyze all RDS instances
recommendations = optimizer.analyze_all_databases()

# Export recommendations to Excel
optimizer.export_recommendations(recommendations, 'rds_optimization_report.xlsx')

# Generate CLI commands for implementation
optimizer.generate_cli_commands(recommendations, 'rds_commands.sh')
```

### Pattern Detection Examples

```python
from aws_cost_optimizer.analysis import PatternDetector

# Initialize pattern detector
detector = PatternDetector(
    analysis_period_days=90,
    min_data_points=100
)

# Analyze resource patterns
patterns = detector.analyze_all_resources()

# Characterize workloads
workload_characteristics = detector.characterize_workloads(patterns)

# Export analysis
detector.export_pattern_analysis(patterns, workload_characteristics, 'pattern_analysis.xlsx')
```

### Utility Functions Examples

```python
from aws_cost_optimizer.utils import (
    AWSHelper,
    ConfigManager,
    CostCalculator,
    TagUtils
)

# Test AWS connection
helper = AWSHelper()
if helper.test_credentials():
    print("AWS credentials are valid")

# Load configuration
config = ConfigManager.load_config()

# Calculate costs
monthly_cost = CostCalculator.calculate_ec2_monthly_cost('t3.large', 'us-east-1')
print(f"Monthly cost: ${monthly_cost:.2f}")

# Analyze tags
tags = {'Environment': 'Production', 'Application': 'WebApp'}
if TagUtils.is_critical_resource(tags):
    print("This is a critical resource")
```

See the full documentation in the `docs/` directory for detailed usage instructions.