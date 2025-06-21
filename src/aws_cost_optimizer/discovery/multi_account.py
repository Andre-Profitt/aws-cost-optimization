"""
Multi-account AWS resource discovery module
"""
import boto3
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

@dataclass
class AWSAccount:
    """Represents an AWS account for discovery"""
    account_id: str
    account_name: str
    role_name: str

class MultiAccountInventory:
    """Handles resource discovery across multiple AWS accounts"""
    
    def __init__(self, accounts: List[AWSAccount], regions: List[str]):
        self.accounts = accounts
        self.regions = regions
        self.resources = []
    
    def assume_role(self, account: AWSAccount) -> boto3.Session:
        """Assume role in target account"""
        sts = boto3.client('sts')
        role_arn = f"arn:aws:iam::{account.account_id}:role/{account.role_name}"
        
        try:
            response = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f"CostOptimizer-{account.account_name}"
            )
            
            return boto3.Session(
                aws_access_key_id=response['Credentials']['AccessKeyId'],
                aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                aws_session_token=response['Credentials']['SessionToken']
            )
        except Exception as e:
            logger.error(f"Failed to assume role in {account.account_name}: {e}")
            raise
    
    def discover_ec2_instances(self, session: boto3.Session, account: AWSAccount, region: str) -> List[Dict[str, Any]]:
        """Discover EC2 instances in a specific region"""
        ec2 = session.client('ec2', region_name=region)
        instances = []
        
        try:
            response = ec2.describe_instances()
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'resource_id': instance['InstanceId'],
                        'resource_type': 'EC2',
                        'account_id': account.account_id,
                        'account_name': account.account_name,
                        'region': region,
                        'state': instance['State']['Name'],
                        'instance_type': instance.get('InstanceType'),
                        'launch_time': str(instance.get('LaunchTime')),
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    })
        except Exception as e:
            logger.error(f"Error discovering EC2 in {account.account_name}/{region}: {e}")
        
        return instances
    
    def discover_rds_instances(self, session: boto3.Session, account: AWSAccount, region: str) -> List[Dict[str, Any]]:
        """Discover RDS instances in a specific region"""
        rds = session.client('rds', region_name=region)
        instances = []
        
        try:
            response = rds.describe_db_instances()
            for db in response['DBInstances']:
                instances.append({
                    'resource_id': db['DBInstanceIdentifier'],
                    'resource_type': 'RDS',
                    'account_id': account.account_id,
                    'account_name': account.account_name,
                    'region': region,
                    'state': db['DBInstanceStatus'],
                    'instance_class': db['DBInstanceClass'],
                    'engine': db['Engine'],
                    'allocated_storage': db['AllocatedStorage']
                })
        except Exception as e:
            logger.error(f"Error discovering RDS in {account.account_name}/{region}: {e}")
        
        return instances
    
    def discover_account_resources(self, account: AWSAccount) -> List[Dict[str, Any]]:
        """Discover all resources in a single account"""
        resources = []
        
        try:
            session = self.assume_role(account)
            
            for region in self.regions:
                # Discover EC2
                resources.extend(self.discover_ec2_instances(session, account, region))
                # Discover RDS
                resources.extend(self.discover_rds_instances(session, account, region))
                # Add more resource types as needed
        
        except Exception as e:
            logger.error(f"Failed to discover resources in {account.account_name}: {e}")
        
        return resources
    
    def collect_all_accounts_inventory(self) -> List[Dict[str, Any]]:
        """Collect inventory from all accounts in parallel"""
        all_resources = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_account = {
                executor.submit(self.discover_account_resources, account): account 
                for account in self.accounts
            }
            
            for future in as_completed(future_to_account):
                account = future_to_account[future]
                try:
                    resources = future.result()
                    all_resources.extend(resources)
                    logger.info(f"Discovered {len(resources)} resources in {account.account_name}")
                except Exception as e:
                    logger.error(f"Failed to process {account.account_name}: {e}")
        
        self.resources = all_resources
        return all_resources
    
    def export_to_excel(self, output_file: str):
        """Export inventory to Excel file"""
        if not self.resources:
            logger.warning("No resources to export")
            return
        
        df = pd.DataFrame(self.resources)
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = df.groupby(['account_name', 'resource_type']).size().reset_index(name='count')
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # EC2 instances
            ec2_df = df[df['resource_type'] == 'EC2']
            if not ec2_df.empty:
                ec2_df.to_excel(writer, sheet_name='EC2_Instances', index=False)
            
            # RDS instances
            rds_df = df[df['resource_type'] == 'RDS']
            if not rds_df.empty:
                rds_df.to_excel(writer, sheet_name='RDS_Instances', index=False)
            
            # All resources
            df.to_excel(writer, sheet_name='All_Resources', index=False)
        
        logger.info(f"Exported {len(self.resources)} resources to {output_file}")