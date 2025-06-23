# AWS Cost Optimizer - Enterprise Features

## Overview

The Enterprise Edition of AWS Cost Optimizer provides production-grade features for large organizations, including dependency mapping, change management, enhanced monitoring, and comprehensive compliance capabilities.

## Key Features

### 1. Dependency Mapping
- **Automatic Discovery**: Maps relationships between AWS resources
- **Impact Analysis**: Predicts the impact of changes on dependent resources
- **Risk Assessment**: Evaluates risk levels based on dependency chains
- **Visual Graphs**: Generates dependency visualization graphs

### 2. Change Management
- **Ticketing Integration**: Integrates with ServiceNow and Jira
- **Approval Workflows**: Multi-level approval based on risk and savings
- **Change Tracking**: Complete audit trail of all changes
- **Risk-Based Routing**: Routes high-risk changes to appropriate approvers

### 3. Enhanced Monitoring
- **CloudWatch Integration**: Creates custom dashboards and alarms
- **Post-Change Monitoring**: Monitors resources for 72 hours after changes
- **Anomaly Detection**: Detects unusual behavior after optimizations
- **Multi-System Support**: Integrates with Datadog, New Relic, etc.

### 4. Compliance & Audit
- **Tag Compliance**: Enforces required tags and naming conventions
- **Regulatory Compliance**: Supports HIPAA, PCI, SOX, GDPR requirements
- **Data Residency**: Ensures resources comply with geographic restrictions
- **Complete Audit Trail**: Logs all actions for compliance reporting

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key additional dependencies for enterprise features:
- `networkx` - Dependency graph analysis
- `jira` - Jira integration
- `pysnow` - ServiceNow integration
- `prometheus-client` - Custom metrics
- `apscheduler` - Scheduled optimization runs

### 2. Configure Environment Variables

Create a `.env` file or export these variables:

```bash
# ServiceNow Configuration
export SERVICENOW_URL="https://your-instance.service-now.com"
export SERVICENOW_USER="integration_user"
export SERVICENOW_PASS="secure_password"

# Jira Configuration (if using Jira instead)
export JIRA_URL="https://your-company.atlassian.net"
export JIRA_USER="integration@company.com"
export JIRA_TOKEN="your_api_token"
export JIRA_PROJECT="AWSOPT"

# AWS Configuration
export AWS_PROFILE="cost-optimizer"
export AWS_ACCOUNT_ID="123456789012"

# Optional: Monitoring Integration
export DATADOG_API_KEY="your_datadog_api_key"
export DATADOG_APP_KEY="your_datadog_app_key"
```

### 3. Configure Enterprise Settings

Copy and customize the enterprise configuration:

```bash
cp config/enterprise_config.yaml config/my_enterprise_config.yaml
# Edit the file with your organization's settings
```

## Usage

### Command Line Interface

#### 1. Run Enterprise Analysis

```bash
python -m aws_cost_optimizer enterprise-analyze \
  --regions us-east-1 us-west-2 \
  --services ec2 rds s3 \
  --user john.doe@company.com \
  --output enterprise_analysis
```

#### 2. Execute Approved Changes

```bash
# Dry run (default)
python -m aws_cost_optimizer execute-changes --dry-run

# Execute changes (requires confirmation)
python -m aws_cost_optimizer execute-changes --execute

# Force execution without confirmation
python -m aws_cost_optimizer execute-changes --execute --force
```

#### 3. Generate Compliance Report

```bash
python -m aws_cost_optimizer enterprise-report \
  --days 30 \
  --output compliance_report.json
```

### Python API

#### Basic Enterprise Optimization

```python
from aws_cost_optimizer.enterprise import EnterpriseConfig, EnterpriseOptimizer

# Configure
config = EnterpriseConfig(
    enable_dependency_mapping=True,
    enable_change_management=True,
    enable_monitoring=True,
    enable_compliance=True,
    enable_audit_trail=True,
    ticketing_system="servicenow"
)

# Initialize optimizer
optimizer = EnterpriseOptimizer(config)

# Run optimization
results = optimizer.run_enterprise_optimization(
    regions=['us-east-1', 'us-west-2'],
    services=['ec2', 'rds', 's3'],
    user='automation@company.com'
)

print(f"Total savings: ${results['optimization_result'].total_monthly_savings:,.2f}/month")
print(f"Change requests created: {results['change_requests_created']}")
```

#### Scheduled Optimization

```python
# Run the example script with scheduling
python examples/enterprise_example.py --schedule --cron "0 9 * * 1"
```

This will run optimization every Monday at 9 AM.

## Enterprise Workflow

### 1. Discovery Phase
- Scans all specified regions and services
- Maps dependencies between resources
- Checks compliance status
- Identifies optimization opportunities

### 2. Analysis Phase
- Calculates potential savings
- Assesses risk levels
- Filters non-compliant resources
- Creates change requests

### 3. Approval Phase
- Creates tickets in ServiceNow/Jira
- Routes based on risk and savings
- Auto-approves low-risk changes (if configured)
- Waits for manual approval for high-risk changes

### 4. Execution Phase
- Validates compliance before execution
- Sets up pre-change monitoring
- Executes approved changes
- Monitors for 72 hours post-change

### 5. Reporting Phase
- Generates compliance reports
- Exports audit trails
- Creates executive summaries
- Sends notifications

## Configuration Deep Dive

### Dependency Mapping

```yaml
enterprise:
  dependency_mapping:
    enabled: true
    analyze_vpc_flow_logs: true
    analyze_cloudformation: true
    max_depth: 5
```

### Change Management

```yaml
enterprise:
  change_management:
    enabled: true
    ticketing_system: servicenow
    approval_rules:
      auto_approve_low_risk: false
      auto_approve_max_savings: 100
      require_approval_for_production: true
```

### Compliance Rules

```yaml
enterprise:
  compliance:
    required_tags:
      - Environment
      - Owner
      - CostCenter
    custom_rules:
      - rule_id: "PROD_LOCK_001"
        name: "Production Lock"
        tag_patterns:
          Environment: "^(prod|production)$"
        severity: "critical"
```

## Best Practices

### 1. Start with Dry Run
Always test in dry-run mode first:
```yaml
safety:
  dry_run: true
```

### 2. Implement Gradual Rollout
- Start with non-production environments
- Enable auto-approval only for low-risk changes
- Monitor results before expanding scope

### 3. Configure Business Hours
Ensure changes happen during appropriate times:
```yaml
safety:
  business_hours:
    timezone: "America/New_York"
    start_hour: 9
    end_hour: 17
    work_days: [1, 2, 3, 4, 5]
```

### 4. Set Up Monitoring
- Enable CloudWatch dashboards
- Configure post-change monitoring
- Set up anomaly detection

### 5. Regular Compliance Audits
- Schedule weekly compliance reports
- Review audit trails monthly
- Update compliance rules quarterly

## Troubleshooting

### Common Issues

1. **ServiceNow Connection Failed**
   ```bash
   # Test connection
   curl -u $SERVICENOW_USER:$SERVICENOW_PASS \
     $SERVICENOW_URL/api/now/table/change_request
   ```

2. **Dependency Mapping Timeout**
   - Reduce the number of regions
   - Increase `max_depth` gradually
   - Enable caching

3. **Compliance Blocking All Changes**
   - Review compliance rules
   - Add required tags to resources
   - Use exemption tags for special cases

### Debug Mode

Enable detailed logging:
```yaml
advanced:
  debug:
    enabled: true
    verbose_logging: true
```

## Security Considerations

1. **Credentials**: Never commit credentials to version control
2. **IAM Permissions**: Use least-privilege policies
3. **Encryption**: Enable encryption for audit trails
4. **Access Control**: Implement role-based access
5. **Audit Logs**: Retain for compliance period

## Support

For issues or questions:
1. Check the logs in `aws-cost-optimizer.log`
2. Review audit trail for specific events
3. Consult the API documentation
4. Contact your DevOps team

## Next Steps

1. Review `examples/enterprise_example.py` for a complete workflow
2. Customize `config/enterprise_config.yaml` for your organization
3. Set up environment variables
4. Run a test optimization in dry-run mode
5. Schedule regular optimization runs