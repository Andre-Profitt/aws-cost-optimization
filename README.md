# AWS Cost Optimizer

A comprehensive Python automation tool for safely reducing AWS costs without breaking production. This tool implements systematic approaches for multi-account resource inventory, usage pattern detection, and safe resource optimization.

## 🚀 Features

- **Multi-Account Resource Discovery**: Automated inventory collection across multiple AWS accounts
- **Periodic Usage Pattern Detection**: Identify monthly/quarterly batch jobs before optimization
- **Safety-First Approach**: Comprehensive pre-deletion assessments and rollback capabilities
- **Automated Optimization**: Policy-based resource management using Cloud Custodian
- **Cost Savings Tracking**: Real-time savings calculations and reporting
- **Production-Safe**: Gradual shutdown process with multiple safety checkpoints

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

safety:
  dry_run: true
  snapshot_before_deletion: true
  safety_period_days: 30
  require_approval: true
```

## 🚀 Quick Start

### 1. Run Discovery (Safe - Read Only)
```bash
# Discover all resources across accounts
python -m aws_cost_optimizer discover --output-format excel

# Discover with specific resource types
python -m aws_cost_optimizer discover --resource-types ec2,rds,s3
```

### 2. Analyze Usage Patterns
```bash
# Analyze resource usage patterns over 90 days
python -m aws_cost_optimizer analyze --days 90 --detect-periodic

# Generate optimization recommendations
python -m aws_cost_optimizer recommend --savings-target 20000
```

### 3. Execute Optimization (With Safety Checks)
```bash
# Dry run - see what would be changed
python -m aws_cost_optimizer optimize --dry-run

# Execute with approval workflow
python -m aws_cost_optimizer optimize --require-approval

# Execute specific optimizations
python -m aws_cost_optimizer optimize --actions stop-idle-ec2,enable-s3-tiering
```

## 📊 Example Output

```
AWS Cost Optimizer Report
Generated: 2024-01-15 10:30:00

Current Monthly Spend: $47,000
Potential Savings: $22,500 (47.8%)

Quick Wins (Week 1-2):
- Unattached EBS Volumes: $1,200/month
- Unused Elastic IPs: $180/month
- Idle Dev Instances: $3,500/month

Medium-term (Month 1-3):
- EC2 Rightsizing: $8,000/month
- RDS Consolidation: $4,500/month
- S3 Intelligent Tiering: $5,120/month

Total Identified Savings: $22,500/month
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
│       │   └── safety_checks.py
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