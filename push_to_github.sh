#!/bin/bash
# Script to push the integrated changes to GitHub

# Add all modified and new files
git add README.md
git add src/aws_cost_optimizer/__init__.py
git add src/aws_cost_optimizer/analysis/__init__.py
git add src/aws_cost_optimizer/analysis/pattern_detector.py
git add src/aws_cost_optimizer/cli.py
git add src/aws_cost_optimizer/optimization/__init__.py
git add src/aws_cost_optimizer/optimization/rds_optimizer.py
git add src/aws_cost_optimizer/orchestrator.py
git add src/aws_cost_optimizer/utils/__init__.py
git add src/aws_cost_optimizer/utils/utility_helpers.py

# Create commit
git commit -m "Integrate enhanced RDS optimizer, pattern detector, and utility helpers

- Added comprehensive RDS optimization with multi-region support
- Enhanced pattern detection for workload characterization
- Added utility helpers for AWS operations and cost calculations
- Updated imports and fixed compatibility issues
- Added detailed documentation and usage examples

New features:
- RDS rightsizing, RI recommendations, Aurora migration analysis
- Workload pattern analysis (daily, weekly, monthly)
- Business context inference and optimization scoring
- Configuration management and cost calculations
- Smart tag analysis and report generation"

# Push to GitHub
git push origin main

echo "Changes pushed to GitHub successfully!"