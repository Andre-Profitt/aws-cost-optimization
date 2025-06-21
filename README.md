# AWS Cost Optimizer

A comprehensive Python automation tool for safely reducing AWS costs without breaking production. This tool implements systematic approaches for multi-account resource inventory, usage pattern detection, and safe resource optimization.

## 🚀 Features

- **Multi-Account Resource Discovery**: Automated inventory collection across multiple AWS accounts
- **Periodic Usage Pattern Detection**: Identify monthly/quarterly batch jobs before optimization
- **Safety-First Approach**: Comprehensive pre-deletion assessments and rollback capabilities
- **Automated Optimization**: Policy-based resource management using Cloud Custodian
- **Cost Savings Tracking**: Real-time savings calculations and reporting
- **Production-Safe**: Gradual shutdown process with multiple safety checkpoints

### New Features (v2.0)
- **RDS Optimizer**: Identifies idle databases, duplicates, rightsizing opportunities, and Aurora migration candidates
- **S3 Optimizer**: Intelligent-Tiering automation, lifecycle policies, duplicate detection, and access pattern analysis
- **Cloud Custodian Policies**: Pre-built policies for S3 cost optimization with automated enforcement

### New Features (v3.0)
- **Network Optimizer**: NAT Gateway optimization, Elastic IP cleanup, VPC Endpoint recommendations, data transfer analysis
- **Reserved Instance Analyzer**: RI/Savings Plan recommendations based on usage patterns with ROI calculations
- **Cost Anomaly Detector**: Real-time anomaly detection using statistical analysis and machine learning
- **Auto-Remediation Engine**: Automated execution of optimization recommendations with safety controls
- **Unified Orchestrator**: Comprehensive analysis across all optimization components

## 📋 Prerequisites

- Python 3.8+
- AWS CLI configured with appropriate credentials
- Cross-account roles with necessary permissions
- Docker (optional, for containerized deployment)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-org/aws-cost-optimizer.git
cd aws-cost-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🔧 Configuration

1. Copy the example configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

2. Edit `config/config.yaml` with your AWS account details:
```yaml
aws:
  accounts:
    - id: "123456789012"
      name: "production"
      role_name: "OrganizationAccountAccessRole"
    - id: "234567890123"
      name: "development"
      role_name: "OrganizationAccountAccessRole"
  
  regions:
    - us-east-1
    - us-west-2
    - eu-west-1

optimization:
  ec2:
    cpu_threshold: 10
    network_io_threshold: 5
    observation_days: 14
  
  rds:
    connection_threshold: 7
    cpu_threshold: 25
    observation_days: 60
  
  s3:
    intelligent_tiering_min_size: 1000000000  # 1TB in bytes

  # Reserved Instance Analysis
  ri_lookback_days: 90
  ri_minimum_savings: 100  # Minimum monthly savings to recommend

  # Anomaly Detection
  anomaly_lookback_days: 90
  anomaly_threshold: 2.5  # Z-score threshold
  anomaly_alerts_enabled: true
  anomaly_sns_topic: arn:aws:sns:us-east-1:123456789012:cost-anomalies

  # Auto-Remediation
  enable_auto_remediation: true
  remediation_dry_run: true
  max_auto_remediation_savings: 500  # Max monthly savings to auto-approve
  business_hours_only: true
  notification_endpoints:
    - arn:aws:sns:us-east-1:123456789012:cost-optimization

safety:
  dry_run: true
  snapshot_before_deletion: true
  safety_period_days: 30
  require_approval: true
```

## 🚀 Quick Start

### 1. Configure the Tool
```bash
# Interactive configuration setup
python -m aws_cost_optimizer configure
```

### 2. Run Comprehensive Analysis
```bash
# Full cost optimization analysis across all services
python -m aws_cost_optimizer analyze

# Analyze specific regions
python -m aws_cost_optimizer analyze -r us-east-1 -r us-west-2

# Generate HTML executive report
python -m aws_cost_optimizer analyze --format html --output cost_report

# Export detailed Excel report
python -m aws_cost_optimizer analyze --format excel --output detailed_analysis
```

### 3. Quick Scans for Specific Areas
```bash
# EC2 optimization scan
python -m aws_cost_optimizer quick-scan --type ec2

# Network cost analysis
python -m aws_cost_optimizer quick-scan --type network

# Reserved Instance opportunities
python -m aws_cost_optimizer quick-scan --type ri

# Cost anomaly detection
python -m aws_cost_optimizer quick-scan --type anomalies
```

### 4. Auto-Remediation (With Safety Controls)
```bash
# Dry run mode - see what would be changed
python -m aws_cost_optimizer remediate --dry-run

# Execute with auto-approval for low-risk actions
python -m aws_cost_optimizer remediate --auto-approve

# Full execution mode
python -m aws_cost_optimizer remediate --execute
```

### 5. Generate Executive Report
```bash
# Generate comprehensive HTML report
python -m aws_cost_optimizer report --output executive_summary.html
```

## 📊 Example Output

```
🔍 Starting comprehensive cost optimization analysis...
Analyzing [████████████████████████████████] 100%

💰 Total potential monthly savings: $32,847.50
📊 Found 147 optimization opportunities
⚠️  Detected 5 cost anomalies

✅ Executive report saved to cost_report.html

Key Findings:
  💰 Total Monthly Savings: $32,847.50
  📈 Annual Savings: $394,170.00
  🎯 Recommendations: 147
  ⚠️  Anomalies: 5

Top Opportunities:
1. EC2 Optimization: $15,234.00/month
   - 23 idle instances identified
   - 45 rightsizing opportunities
   - 12 instances with low utilization

2. Network Optimization: $8,456.00/month
   - 5 unused NAT Gateways
   - 18 unattached Elastic IPs
   - VPC Endpoint opportunities for S3/DynamoDB

3. Reserved Instance Opportunities: $9,157.50/month
   - 15 RI recommendations
   - 3 Savings Plan opportunities
   - Average ROI: 4.2 months

Cost Anomalies Detected:
- RDS Production: CRITICAL - 250% increase ($1,200 impact)
- Lambda Functions: HIGH - Invocation spike detected
- S3 Transfer: MEDIUM - Inter-region transfer increase
```

## 🏗️ Architecture

```
aws-cost-optimizer/
├── src/
│   └── aws_cost_optimizer/
│       ├── __init__.py
│       ├── cli.py                 # Command-line interface
│       ├── discovery/             # Resource discovery modules
│       │   ├── __init__.py
│       │   ├── multi_account.py
│       │   ├── ec2_discovery.py
│       │   ├── rds_discovery.py
│       │   └── s3_discovery.py
│       ├── analysis/              # Usage analysis modules
│       │   ├── __init__.py
│       │   ├── pattern_detector.py
│       │   ├── cost_calculator.py
│       │   └── anomaly_detection.py
│       ├── optimization/          # Optimization engines
│       │   ├── __init__.py
│       │   ├── ec2_optimizer.py
│       │   ├── rds_optimizer.py
│       │   ├── s3_optimizer.py
│       │   ├── network_optimizer.py
│       │   ├── reserved_instance_analyzer.py
│       │   ├── auto_remediation_engine.py
│       │   └── safety_checks.py
│       ├── orchestrator.py        # Main orchestration engine
│       ├── reporting/             # Report generation
│       │   ├── __init__.py
│       │   ├── excel_reporter.py
│       │   └── dashboard.py
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── aws_helpers.py
│           └── logging_config.py
├── policies/                      # Cloud Custodian policies
│   ├── ec2-policies.yaml
│   ├── rds-policies.yaml
│   └── s3-policies.yaml
├── tests/                         # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── config/                        # Configuration files
│   ├── config.example.yaml
│   └── logging.yaml
├── scripts/                       # Utility scripts
│   ├── setup_cross_account_roles.sh
│   └── emergency_rollback.py
├── docs/                          # Documentation
│   ├── getting_started.md
│   ├── api_reference.md
│   └── best_practices.md
├── .github/                       # GitHub Actions
│   └── workflows/
│       ├── ci.yml
│       ├── daily_discovery.yml
│       └── weekly_optimization.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

## 🔒 Safety Features

1. **Dry Run Mode**: All optimization commands support `--dry-run` to preview changes
2. **Automated Snapshots**: Creates snapshots before any resource deletion
3. **Gradual Shutdown**: 5-phase shutdown process with safety checkpoints
4. **Rollback Capability**: One-command rollback to previous state
5. **Approval Workflow**: Optional approval requirement for production changes
6. **Dependency Checking**: Automated dependency mapping before changes

## 📈 Monitoring & Alerts

The tool integrates with:
- **CloudWatch**: For usage metrics and anomaly detection
- **SNS**: For approval workflows and notifications
- **Cost Explorer**: For savings tracking and reporting

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool modifies AWS resources and can impact production systems. Always:
- Test in non-production environments first
- Use dry-run mode before executing changes
- Maintain proper backups
- Follow your organization's change management procedures

## 🙏 Acknowledgments

- Based on AWS Well-Architected Framework
- Uses Cloud Custodian for policy enforcement
- Inspired by real-world cost optimization case studies

## 📞 Support

- Create an issue for bug reports or feature requests
- Join our Slack channel: #aws-cost-optimization
- Email: aws-cost-optimizer@your-org.com