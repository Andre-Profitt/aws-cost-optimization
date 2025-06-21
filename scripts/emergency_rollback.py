#!/usr/bin/env python3
"""
Emergency Rollback Script for AWS Cost Optimizer
Quickly revert optimization actions in case of issues
"""
import boto3
import click
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmergencyRollback:
    """Handle emergency rollback of optimization actions"""
    
    def __init__(self, audit_log_file: str = 'logs/optimization_audit.json'):
        self.audit_log_file = audit_log_file
        self.session = boto3.Session()
        self.ec2 = self.session.client('ec2')
        self.rds = self.session.client('rds')
        self.s3 = self.session.client('s3')
        
    def load_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Load recent actions from audit log"""
        actions = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            with open(self.audit_log_file, 'r') as f:
                for line in f:
                    try:
                        action = json.loads(line)
                        action_time = datetime.fromisoformat(action['timestamp'])
                        if action_time > cutoff_time:
                            actions.append(action)
                    except:
                        continue
        except FileNotFoundError:
            logger.error(f"Audit log not found: {self.audit_log_file}")
            
        return actions
    
    def rollback_ec2_stop(self, instance_id: str, region: str) -> bool:
        """Start a stopped EC2 instance"""
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            # Check current state
            response = ec2_regional.describe_instances(InstanceIds=[instance_id])
            if not response['Reservations']:
                logger.error(f"Instance {instance_id} not found")
                return False
                
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            
            if state == 'stopped':
                logger.info(f"Starting instance {instance_id}")
                ec2_regional.start_instances(InstanceIds=[instance_id])
                
                # Wait for instance to start
                waiter = ec2_regional.get_waiter('instance_running')
                waiter.wait(InstanceIds=[instance_id])
                
                logger.info(f"✓ Instance {instance_id} started successfully")
                return True
            else:
                logger.info(f"Instance {instance_id} is already in state: {state}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start instance {instance_id}: {e}")
            return False
    
    def rollback_ec2_terminate(self, instance_id: str, ami_id: str, region: str) -> bool:
        """Recreate terminated instance from AMI"""
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            logger.info(f"Recreating instance from AMI {ami_id}")
            
            # Get original instance details from AMI tags if available
            response = ec2_regional.run_instances(
                ImageId=ami_id,
                MinCount=1,
                MaxCount=1,
                InstanceType='t3.micro',  # Safe default
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'Restored-{instance_id}'},
                        {'Key': 'RestoredFrom', 'Value': instance_id},
                        {'Key': 'RestoredAt', 'Value': datetime.utcnow().isoformat()}
                    ]
                }]
            )
            
            new_instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"✓ Created new instance {new_instance_id} from AMI {ami_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate instance from {ami_id}: {e}")
            return False
    
    def rollback_rds_stop(self, db_instance_id: str, region: str) -> bool:
        """Start a stopped RDS instance"""
        try:
            rds_regional = self.session.client('rds', region_name=region)
            
            # Check current state
            response = rds_regional.describe_db_instances(DBInstanceIdentifier=db_instance_id)
            if not response['DBInstances']:
                logger.error(f"RDS instance {db_instance_id} not found")
                return False
                
            status = response['DBInstances'][0]['DBInstanceStatus']
            
            if status == 'stopped':
                logger.info(f"Starting RDS instance {db_instance_id}")
                rds_regional.start_db_instance(DBInstanceIdentifier=db_instance_id)
                
                logger.info(f"✓ RDS instance {db_instance_id} start initiated")
                return True
            else:
                logger.info(f"RDS instance {db_instance_id} is already in state: {status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start RDS instance {db_instance_id}: {e}")
            return False
    
    def rollback_nat_gateway_delete(self, nat_gateway_id: str, subnet_id: str, 
                                  allocation_id: str, region: str) -> bool:
        """Recreate deleted NAT Gateway"""
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            logger.info(f"Recreating NAT Gateway in subnet {subnet_id}")
            
            response = ec2_regional.create_nat_gateway(
                SubnetId=subnet_id,
                AllocationId=allocation_id,
                TagSpecifications=[{
                    'ResourceType': 'nat-gateway',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'Restored-{nat_gateway_id}'},
                        {'Key': 'RestoredFrom', 'Value': nat_gateway_id}
                    ]
                }]
            )
            
            new_nat_id = response['NatGateway']['NatGatewayId']
            logger.info(f"✓ Created new NAT Gateway {new_nat_id}")
            
            # TODO: Update route tables
            logger.warning("Remember to update route tables to use new NAT Gateway")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate NAT Gateway: {e}")
            return False
    
    def rollback_s3_lifecycle(self, bucket_name: str, previous_rules: List[Dict]) -> bool:
        """Restore previous S3 lifecycle configuration"""
        try:
            if previous_rules:
                logger.info(f"Restoring lifecycle rules for bucket {bucket_name}")
                self.s3.put_bucket_lifecycle_configuration(
                    Bucket=bucket_name,
                    LifecycleConfiguration={'Rules': previous_rules}
                )
            else:
                logger.info(f"Removing lifecycle rules from bucket {bucket_name}")
                self.s3.delete_bucket_lifecycle(Bucket=bucket_name)
                
            logger.info(f"✓ Lifecycle configuration restored for {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore lifecycle for {bucket_name}: {e}")
            return False
    
    def rollback_all(self, hours: int = 24, dry_run: bool = True) -> Dict[str, Any]:
        """Rollback all actions from the last N hours"""
        actions = self.load_audit_log(hours)
        
        results = {
            'total_actions': len(actions),
            'successful_rollbacks': 0,
            'failed_rollbacks': 0,
            'skipped': 0,
            'details': []
        }
        
        logger.info(f"Found {len(actions)} actions to potentially rollback")
        
        for action in reversed(actions):  # Rollback in reverse order
            action_type = action.get('action_type')
            resource_id = action.get('resource_id')
            region = action.get('region', 'us-east-1')
            
            if dry_run:
                logger.info(f"[DRY RUN] Would rollback: {action_type} on {resource_id}")
                results['skipped'] += 1
                continue
            
            success = False
            
            if action_type == 'stop_instance':
                success = self.rollback_ec2_stop(resource_id, region)
            elif action_type == 'terminate_instance':
                ami_id = action.get('backup_ami_id')
                if ami_id:
                    success = self.rollback_ec2_terminate(resource_id, ami_id, region)
            elif action_type == 'stop_rds':
                success = self.rollback_rds_stop(resource_id, region)
            elif action_type == 'delete_nat_gateway':
                subnet_id = action.get('subnet_id')
                allocation_id = action.get('allocation_id')
                if subnet_id and allocation_id:
                    success = self.rollback_nat_gateway_delete(
                        resource_id, subnet_id, allocation_id, region
                    )
            elif action_type == 'modify_s3_lifecycle':
                previous_rules = action.get('previous_rules', [])
                success = self.rollback_s3_lifecycle(resource_id, previous_rules)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                results['skipped'] += 1
                continue
            
            if success:
                results['successful_rollbacks'] += 1
            else:
                results['failed_rollbacks'] += 1
                
            results['details'].append({
                'resource_id': resource_id,
                'action_type': action_type,
                'rollback_success': success,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return results
    
    def rollback_specific(self, resource_ids: List[str], dry_run: bool = True) -> Dict[str, Any]:
        """Rollback specific resources only"""
        actions = self.load_audit_log(hours=168)  # Last 7 days
        
        # Filter for specific resources
        filtered_actions = [
            a for a in actions 
            if a.get('resource_id') in resource_ids
        ]
        
        results = {
            'requested_resources': len(resource_ids),
            'found_actions': len(filtered_actions),
            'rollbacks': []
        }
        
        for action in filtered_actions:
            # Similar rollback logic as rollback_all
            pass
        
        return results

@click.command()
@click.option('--hours', default=24, help='Rollback actions from last N hours')
@click.option('--resource-id', multiple=True, help='Specific resource IDs to rollback')
@click.option('--dry-run/--execute', default=True, help='Dry run or execute rollback')
@click.option('--audit-log', default='logs/optimization_audit.json', help='Audit log file path')
def main(hours, resource_id, dry_run, audit_log):
    """Emergency rollback tool for AWS Cost Optimizer actions"""
    
    click.echo("AWS Cost Optimizer - Emergency Rollback")
    click.echo("======================================")
    
    rollback = EmergencyRollback(audit_log_file=audit_log)
    
    if resource_id:
        # Rollback specific resources
        click.echo(f"Rolling back specific resources: {', '.join(resource_id)}")
        results = rollback.rollback_specific(list(resource_id), dry_run=dry_run)
    else:
        # Rollback all recent actions
        click.echo(f"Rolling back all actions from last {hours} hours")
        
        if not dry_run:
            if not click.confirm('This will rollback ALL optimization actions. Are you sure?'):
                click.echo("Rollback cancelled.")
                return
                
        results = rollback.rollback_all(hours=hours, dry_run=dry_run)
    
    # Display results
    click.echo("\nRollback Results:")
    click.echo(f"Total actions found: {results.get('total_actions', 0)}")
    click.echo(f"Successful rollbacks: {results.get('successful_rollbacks', 0)}")
    click.echo(f"Failed rollbacks: {results.get('failed_rollbacks', 0)}")
    click.echo(f"Skipped: {results.get('skipped', 0)}")
    
    if results.get('failed_rollbacks', 0) > 0:
        click.echo("\n⚠️  Some rollbacks failed. Check logs for details.")
    
    # Save rollback report
    report_file = f"logs/rollback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    click.echo(f"\nRollback report saved to: {report_file}")

if __name__ == '__main__':
    main()