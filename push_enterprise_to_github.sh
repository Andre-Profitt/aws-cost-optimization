#!/bin/bash
# Script to push the enterprise integration changes to GitHub

echo "🚀 Pushing Enterprise Integration to GitHub..."

# Add all enterprise-related files
echo "📁 Adding enterprise module files..."
git add src/aws_cost_optimizer/enterprise/
git add src/aws_cost_optimizer/compliance/

# Add updated core files
echo "📁 Adding updated core files..."
git add src/aws_cost_optimizer/__init__.py
git add src/aws_cost_optimizer/orchestrator.py
git add src/aws_cost_optimizer/cli.py

# Add configuration files
echo "📁 Adding configuration files..."
git add config/config.example.yaml
git add config/enterprise_config.yaml

# Add example and documentation
echo "📁 Adding examples and documentation..."
git add examples/enterprise_example.py
git add docs/ENTERPRISE_FEATURES.md

# Add requirements
echo "📁 Adding requirements..."
git add requirements.txt

# Add updated README
echo "📁 Adding README..."
git add README.md

# Show status
echo ""
echo "📋 Git status:"
git status --short

# Create commit
echo ""
echo "💾 Creating commit..."
git commit -m "Add enterprise features: dependency mapping, change management, and enhanced monitoring

- Created enterprise integration module with full workflow orchestration
- Added dependency mapping for safe change execution
- Integrated change management with ServiceNow/Jira support
- Enhanced monitoring with CloudWatch dashboards and anomaly detection
- Extended compliance features with enterprise-grade controls
- Added enterprise CLI commands (enterprise-analyze, execute-changes, enterprise-report)
- Created comprehensive enterprise configuration template
- Added scheduled optimization capability
- Updated requirements with enterprise dependencies
- Added complete enterprise documentation and examples

Enterprise features:
- Automatic resource dependency discovery and impact analysis
- Multi-level approval workflows based on risk and savings
- Post-change monitoring for 72 hours
- Integration with ticketing systems
- Business hours and change freeze period enforcement
- Comprehensive audit trail and compliance reporting

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Enterprise features pushed to GitHub successfully!"
echo ""
echo "Next steps:"
echo "1. Install new dependencies: pip install -r requirements.txt"
echo "2. Configure environment variables for ServiceNow/Jira"
echo "3. Review enterprise documentation: docs/ENTERPRISE_FEATURES.md"
echo "4. Try the example: python examples/enterprise_example.py"