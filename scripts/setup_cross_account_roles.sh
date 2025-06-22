#!/bin/bash

# Setup cross-account roles for AWS Cost Optimizer
# This script creates the necessary IAM roles for multi-account access

ROLE_NAME="OrganizationCostOptimizerRole"
TRUST_POLICY_FILE="/tmp/trust-policy.json"
PERMISSIONS_POLICY_FILE="/tmp/permissions-policy.json"

# Get the main account ID
MAIN_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Setting up cross-account role: $ROLE_NAME"
echo "Main account ID: $MAIN_ACCOUNT_ID"

# Create trust policy
cat > $TRUST_POLICY_FILE << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::${MAIN_ACCOUNT_ID}:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "aws-cost-optimizer-${MAIN_ACCOUNT_ID}"
        }
      }
    }
  ]
}
EOF

# Create permissions policy
cat > $PERMISSIONS_POLICY_FILE << EOF
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
        "cloudwatch:ListMetrics",
        "ce:GetCostAndUsage",
        "ce:GetReservationUtilization",
        "ce:GetSavingsPlansPurchaseRecommendation",
        "ce:GetRightsizingRecommendation",
        "autoscaling:Describe*",
        "elasticloadbalancing:Describe*",
        "tag:GetResources",
        "tag:GetTagKeys",
        "tag:GetTagValues"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Create the role
echo "Creating IAM role..."
aws iam create-role \
  --role-name $ROLE_NAME \
  --assume-role-policy-document file://$TRUST_POLICY_FILE \
  --description "Role for AWS Cost Optimizer cross-account access"

# Attach the policy
echo "Attaching permissions policy..."
aws iam put-role-policy \
  --role-name $ROLE_NAME \
  --policy-name "${ROLE_NAME}Policy" \
  --policy-document file://$PERMISSIONS_POLICY_FILE

# Clean up
rm -f $TRUST_POLICY_FILE $PERMISSIONS_POLICY_FILE

echo "Setup complete!"
echo ""
echo "To use this role in other accounts:"
echo "1. Run this script in each member account"
echo "2. Update the trust policy to include the main account ID: $MAIN_ACCOUNT_ID"
echo "3. Use external ID: aws-cost-optimizer-${MAIN_ACCOUNT_ID}"