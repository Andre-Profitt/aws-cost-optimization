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

### Enterprise Features (NEW)
- **Dependency Mapping**: Automatically discover and map resource dependencies
- **Change Management**: Integration with ServiceNow and Jira for approval workflows
- **Enhanced Monitoring**: CloudWatch dashboards, anomaly detection, and post-change monitoring
- **Compliance Management**: Enforce tagging policies and regulatory requirements
- **Tag Compliance Rules**: Define required, prohibited, and pattern-based tag rules
- **Data Residency Controls**: Ensure resources comply with geographic restrictions
- **Regulatory Compliance**: Built-in support for HIPAA, PCI, SOX, and GDPR requirements
- **Change Freeze Periods**: Honor blackout windows and change freeze tags
- **Compliance Reporting**: Generate detailed compliance status reports
- **Audit Trail**: Complete audit logging for all optimization activities
- **Event Tracking**: Log all recommendations, approvals, and executions
- **Compliance Filtering**: Automatically filter non-compliant optimization recommendations
- **Real-time Alerts**: SNS integration for compliance violations and critical events
- **Scheduled Optimization**: Run automated optimization on a schedule

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

### Compliance and Audit Examples

```python
from aws_cost_optimizer import ComplianceManager, AuditTrail

# Initialize compliance manager
compliance_manager = ComplianceManager(
    config={
        'required_tags': ['Environment', 'Owner', 'CostCenter'],
        'regulatory_tags': ['HIPAA', 'PCI', 'SOX']
    }
)

# Check resource compliance
result = compliance_manager.check_resource_compliance(
    resource_id='i-1234567890',
    resource_type='ec2',
    region='us-east-1',
    tags={'Environment': 'prod', 'Owner': 'devops'}
)

if result['status'].value == 'non_compliant':
    print(f"Resource has {len(result['violations'])} compliance violations")

# Initialize audit trail
audit_trail = AuditTrail(config={
    'audit_bucket': 'company-audit-logs',
    'audit_table': 'cost-optimizer-events'
})

# Query audit events
from datetime import datetime, timedelta
events = audit_trail.query_audit_trail(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    filters={'event_type': 'CHANGE_EXECUTED'}
)
```

### CLI Compliance Commands

```bash
# Check compliance for a specific resource
python -m aws_cost_optimizer check-compliance -r i-1234567890 -t ec2

# Generate compliance report
python -m aws_cost_optimizer compliance-report --start-date 2024-01-01 --end-date 2024-01-31

# Query audit trail
python -m aws_cost_optimizer audit-trail --start-date 2024-01-01 --user john.doe --format csv

# View all available commands
python -m aws_cost_optimizer --help
```

### Enterprise CLI Commands

```bash
# Run enterprise optimization with full features
python -m aws_cost_optimizer enterprise-analyze -r us-east-1 us-west-2

# Execute approved changes
python -m aws_cost_optimizer execute-changes --dry-run

# Generate enterprise compliance report
python -m aws_cost_optimizer enterprise-report --days 30

# Run scheduled optimization (Mondays at 9 AM)
python examples/enterprise_example.py --schedule --cron "0 9 * * 1"
```

### Multi-Account Optimization Commands

```bash
# Discover resources across multiple AWS accounts
python -m aws_cost_optimizer multi-account-inventory -a config/accounts_config_template.json -o inventory.xlsx

# Generate emergency cost reduction plan
python -m aws_cost_optimizer generate-cost-reduction-plan -i inventory.json -r recommendations.json -t 20000

# Run TechStartup optimization scenario
python -m aws_cost_optimizer techstartup-optimize -a accounts.json -o results/

# Analyze S3 bucket access patterns
python -m aws_cost_optimizer analyze-s3-access -d 90 -o unused_buckets.json
```

### New Advanced Features (v2.0)

#### 🔄 Periodic Resource Detection
```bash
# Detect resources with periodic usage patterns (monthly batch jobs, etc.)
python -m aws_cost_optimizer detect-periodic-resources -d 365 -o periodic_analysis.xlsx

# Analyze specific resources
python -m aws_cost_optimizer detect-periodic-resources -r resources.json -o analysis.xlsx
```

#### 🤖 ML-Based Cost Prediction
```bash
# Train ML models and predict future costs
python -m aws_cost_optimizer predict-costs --train -f 30 -b my-ml-bucket

# Predict using existing models
python -m aws_cost_optimizer predict-costs -f 30 -o predictions.json
```

#### ⚡ Real-time Cost Controls
```bash
# Set up real-time monitoring with circuit breakers
python -m aws_cost_optimizer setup-realtime-controls --setup-eventbridge

# Use custom threshold configuration
python -m aws_cost_optimizer setup-realtime-controls -c config/thresholds.yaml
```

#### 🏷️ Intelligent Tagging
```bash
# Analyze and suggest tags using ML
python -m aws_cost_optimizer intelligent-tagging --train-ml -o tagging_report.json

# Enforce tagging policies
python -m aws_cost_optimizer intelligent-tagging --enforce -r resources.json
```

#### 💰 Savings Tracking
```bash
# Track optimization savings and compare projected vs actual
python -m aws_cost_optimizer track-savings -p 30 --update-actuals

# Generate executive report
python -m aws_cost_optimizer track-savings -p 90 --format excel -o executive_report.xlsx
```

### TechStartup Solution (Practice Scenario)

The project includes a complete solution for the TechStartup acquisition scenario:
- **Problem**: $47K/month AWS spend with no documentation
- **Target**: Find $20K/month in savings
- **Solution**: Multi-account discovery, analysis, and emergency cost reduction

See `docs/TECHSTARTUP_SOLUTION.md` for the complete guide.

See the full documentation in the `docs/` directory for detailed usage instructions.
For enterprise features, see `docs/ENTERPRISE_FEATURES.md`.