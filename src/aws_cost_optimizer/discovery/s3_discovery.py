"""
S3 Discovery module for bucket and object inventory
Discovers S3 buckets, their configurations, and usage patterns
"""
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class S3Discovery:
    """Discover S3 resources across regions and accounts"""
    
    def __init__(self, session: Optional[boto3.Session] = None):
        """
        Initialize S3 Discovery
        
        Args:
            session: Boto3 session (optional)
        """
        self.session = session or boto3.Session()
        self.discovered_resources = {
            'buckets': [],
            'bucket_policies': [],
            'lifecycle_rules': [],
            'versioning_configs': [],
            'encryption_configs': [],
            'access_points': [],
            'bucket_metrics': []
        }
        
    def discover_all_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover all S3 resources
        Note: S3 buckets are global, but we need regional clients for some operations
        
        Returns:
            Dictionary of discovered resources by type
        """
        logger.info("Starting S3 discovery")
        
        s3 = self.session.client('s3')
        
        try:
            # Get all buckets
            response = s3.list_buckets()
            buckets = response.get('Buckets', [])
            
            logger.info(f"Found {len(buckets)} S3 buckets")
            
            # Discover detailed information for each bucket
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                for bucket in buckets:
                    futures.append(
                        executor.submit(
                            self._discover_bucket_details,
                            bucket['Name'],
                            bucket.get('CreationDate')
                        )
                    )
                
                for future in as_completed(futures):
                    try:
                        bucket_details = future.result()
                        if bucket_details:
                            self.discovered_resources['buckets'].append(bucket_details)
                    except Exception as e:
                        logger.error(f"Failed to discover bucket details: {e}")
            
            # Discover S3 Access Points
            self._discover_access_points()
            
        except Exception as e:
            logger.error(f"Error discovering S3 resources: {e}")
        
        return self.discovered_resources
    
    def _discover_bucket_details(self, bucket_name: str, creation_date: datetime) -> Dict[str, Any]:
        """Discover detailed information about a specific bucket"""
        try:
            # Get bucket location to determine region
            s3 = self.session.client('s3')
            location = s3.get_bucket_location(Bucket=bucket_name)
            region = location.get('LocationConstraint', 'us-east-1')
            if region is None:
                region = 'us-east-1'
            
            # Create regional client
            s3_regional = self.session.client('s3', region_name=region)
            
            bucket_details = {
                'resource_type': 's3_bucket',
                'resource_id': bucket_name,
                'bucket_name': bucket_name,
                'region': region,
                'creation_date': str(creation_date) if creation_date else '',
                'discovered_at': datetime.utcnow().isoformat()
            }
            
            # Get bucket tags
            try:
                tags_response = s3.get_bucket_tagging(Bucket=bucket_name)
                bucket_details['tags'] = {tag['Key']: tag['Value'] for tag in tags_response.get('TagSet', [])}
            except s3.exceptions.NoSuchTagSet:
                bucket_details['tags'] = {}
            except Exception:
                bucket_details['tags'] = {}
            
            # Get bucket versioning
            try:
                versioning = s3.get_bucket_versioning(Bucket=bucket_name)
                bucket_details['versioning_status'] = versioning.get('Status', 'Disabled')
                bucket_details['mfa_delete'] = versioning.get('MFADelete', 'Disabled')
                
                # Store versioning config separately
                self.discovered_resources['versioning_configs'].append({
                    'bucket_name': bucket_name,
                    'status': versioning.get('Status', 'Disabled'),
                    'mfa_delete': versioning.get('MFADelete', 'Disabled')
                })
            except Exception:
                bucket_details['versioning_status'] = 'Unknown'
                bucket_details['mfa_delete'] = 'Unknown'
            
            # Get bucket encryption
            try:
                encryption = s3.get_bucket_encryption(Bucket=bucket_name)
                rules = encryption.get('ServerSideEncryptionConfiguration', {}).get('Rules', [])
                if rules:
                    encryption_type = rules[0].get('ApplyServerSideEncryptionByDefault', {}).get('SSEAlgorithm', 'None')
                    bucket_details['encryption'] = encryption_type
                    
                    # Store encryption config separately
                    self.discovered_resources['encryption_configs'].append({
                        'bucket_name': bucket_name,
                        'encryption_type': encryption_type,
                        'kms_key_id': rules[0].get('ApplyServerSideEncryptionByDefault', {}).get('KMSMasterKeyID')
                    })
                else:
                    bucket_details['encryption'] = 'None'
            except s3.exceptions.ServerSideEncryptionConfigurationNotFoundError:
                bucket_details['encryption'] = 'None'
            except Exception:
                bucket_details['encryption'] = 'Unknown'
            
            # Get bucket lifecycle configuration
            try:
                lifecycle = s3.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                rules = lifecycle.get('Rules', [])
                bucket_details['lifecycle_rules_count'] = len(rules)
                bucket_details['has_lifecycle_rules'] = len(rules) > 0
                
                # Store lifecycle rules separately
                for rule in rules:
                    self.discovered_resources['lifecycle_rules'].append({
                        'bucket_name': bucket_name,
                        'rule_id': rule.get('ID', 'unnamed'),
                        'status': rule.get('Status', 'Disabled'),
                        'transitions': rule.get('Transitions', []),
                        'expiration': rule.get('Expiration', {}),
                        'noncurrent_transitions': rule.get('NoncurrentVersionTransitions', []),
                        'abort_incomplete_multipart': rule.get('AbortIncompleteMultipartUpload', {})
                    })
                    
                # Check for intelligent tiering
                bucket_details['has_intelligent_tiering'] = any(
                    t.get('StorageClass') == 'INTELLIGENT_TIERING' 
                    for rule in rules 
                    for t in rule.get('Transitions', [])
                )
            except s3.exceptions.NoSuchLifecycleConfiguration:
                bucket_details['lifecycle_rules_count'] = 0
                bucket_details['has_lifecycle_rules'] = False
                bucket_details['has_intelligent_tiering'] = False
            except Exception:
                bucket_details['lifecycle_rules_count'] = 0
                bucket_details['has_lifecycle_rules'] = False
                bucket_details['has_intelligent_tiering'] = False
            
            # Get bucket policy
            try:
                policy_response = s3.get_bucket_policy(Bucket=bucket_name)
                bucket_details['has_bucket_policy'] = True
                
                # Store policy separately
                self.discovered_resources['bucket_policies'].append({
                    'bucket_name': bucket_name,
                    'policy': json.loads(policy_response['Policy'])
                })
            except s3.exceptions.NoSuchBucketPolicy:
                bucket_details['has_bucket_policy'] = False
            except Exception:
                bucket_details['has_bucket_policy'] = False
            
            # Get public access block configuration
            try:
                public_access = s3.get_public_access_block(Bucket=bucket_name)
                config = public_access.get('PublicAccessBlockConfiguration', {})
                bucket_details['public_access_block'] = {
                    'block_public_acls': config.get('BlockPublicAcls', False),
                    'ignore_public_acls': config.get('IgnorePublicAcls', False),
                    'block_public_policy': config.get('BlockPublicPolicy', False),
                    'restrict_public_buckets': config.get('RestrictPublicBuckets', False)
                }
                bucket_details['is_public'] = not all(config.values())
            except s3.exceptions.NoSuchPublicAccessBlockConfiguration:
                bucket_details['public_access_block'] = None
                bucket_details['is_public'] = True  # Assume public if no block config
            except Exception:
                bucket_details['public_access_block'] = None
                bucket_details['is_public'] = 'Unknown'
            
            # Get bucket ACL
            try:
                acl = s3.get_bucket_acl(Bucket=bucket_name)
                bucket_details['owner'] = acl['Owner']['ID']
                bucket_details['grants_count'] = len(acl.get('Grants', []))
            except Exception:
                bucket_details['owner'] = 'Unknown'
                bucket_details['grants_count'] = 0
            
            # Get bucket logging
            try:
                logging_config = s3.get_bucket_logging(Bucket=bucket_name)
                bucket_details['logging_enabled'] = 'LoggingEnabled' in logging_config
                if bucket_details['logging_enabled']:
                    bucket_details['logging_target'] = logging_config['LoggingEnabled'].get('TargetBucket')
            except Exception:
                bucket_details['logging_enabled'] = False
            
            # Get bucket website configuration
            try:
                website = s3.get_bucket_website(Bucket=bucket_name)
                bucket_details['is_website'] = True
                bucket_details['website_endpoint'] = f"{bucket_name}.s3-website-{region}.amazonaws.com"
            except s3.exceptions.NoSuchWebsiteConfiguration:
                bucket_details['is_website'] = False
            except Exception:
                bucket_details['is_website'] = False
            
            # Get bucket CORS configuration
            try:
                cors = s3.get_bucket_cors(Bucket=bucket_name)
                bucket_details['has_cors'] = True
                bucket_details['cors_rules_count'] = len(cors.get('CORSRules', []))
            except s3.exceptions.NoSuchCORSConfiguration:
                bucket_details['has_cors'] = False
                bucket_details['cors_rules_count'] = 0
            except Exception:
                bucket_details['has_cors'] = False
                bucket_details['cors_rules_count'] = 0
            
            # Get bucket replication configuration
            try:
                replication = s3.get_bucket_replication(Bucket=bucket_name)
                bucket_details['has_replication'] = True
                bucket_details['replication_rules_count'] = len(replication.get('ReplicationConfiguration', {}).get('Rules', []))
            except s3.exceptions.ReplicationConfigurationNotFoundError:
                bucket_details['has_replication'] = False
                bucket_details['replication_rules_count'] = 0
            except Exception:
                bucket_details['has_replication'] = False
                bucket_details['replication_rules_count'] = 0
            
            # Get storage metrics from CloudWatch
            bucket_metrics = self._get_bucket_metrics(bucket_name, region)
            bucket_details.update(bucket_metrics)
            
            # Store metrics separately
            if bucket_metrics:
                self.discovered_resources['bucket_metrics'].append({
                    'bucket_name': bucket_name,
                    **bucket_metrics
                })
            
            # Calculate age
            if bucket_details['creation_date']:
                create_time = datetime.fromisoformat(bucket_details['creation_date'].replace('Z', '+00:00').split('+')[0])
                bucket_details['age_days'] = (datetime.utcnow() - create_time).days
            
            # Determine bucket purpose from name/tags
            bucket_details['purpose'] = self._determine_bucket_purpose(bucket_name, bucket_details.get('tags', {}))
            
            return bucket_details
            
        except Exception as e:
            logger.error(f"Error discovering details for bucket {bucket_name}: {e}")
            return None
    
    def _get_bucket_metrics(self, bucket_name: str, region: str) -> Dict[str, Any]:
        """Get bucket metrics from CloudWatch"""
        metrics = {}
        
        try:
            cloudwatch = self.session.client('cloudwatch', region_name=region)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)
            
            # Get bucket size
            size_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if size_response['Datapoints']:
                size_bytes = int(size_response['Datapoints'][0]['Average'])
                metrics['size_bytes'] = size_bytes
                metrics['size_gb'] = round(size_bytes / (1024**3), 2)
            else:
                metrics['size_bytes'] = 0
                metrics['size_gb'] = 0
            
            # Get object count
            count_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='NumberOfObjects',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Average']
            )
            
            if count_response['Datapoints']:
                metrics['object_count'] = int(count_response['Datapoints'][0]['Average'])
            else:
                metrics['object_count'] = 0
            
            # Get storage class distribution
            storage_classes = ['STANDARD', 'STANDARD_IA', 'ONEZONE_IA', 'INTELLIGENT_TIERING', 
                             'GLACIER', 'DEEP_ARCHIVE', 'GLACIER_IR']
            
            metrics['storage_class_distribution'] = {}
            
            for storage_class in storage_classes:
                try:
                    class_response = cloudwatch.get_metric_statistics(
                        Namespace='AWS/S3',
                        MetricName='BucketSizeBytes',
                        Dimensions=[
                            {'Name': 'BucketName', 'Value': bucket_name},
                            {'Name': 'StorageType', 'Value': storage_class}
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=86400,
                        Statistics=['Average']
                    )
                    
                    if class_response['Datapoints']:
                        size = int(class_response['Datapoints'][0]['Average'])
                        if size > 0:
                            metrics['storage_class_distribution'][storage_class] = {
                                'bytes': size,
                                'gb': round(size / (1024**3), 2),
                                'percentage': round((size / metrics['size_bytes']) * 100, 2) if metrics['size_bytes'] > 0 else 0
                            }
                except Exception:
                    pass
            
            # Calculate estimated monthly cost
            metrics['estimated_monthly_cost'] = self._estimate_bucket_cost(metrics)
            
        except Exception as e:
            logger.error(f"Error getting metrics for bucket {bucket_name}: {e}")
        
        return metrics
    
    def _estimate_bucket_cost(self, metrics: Dict[str, Any]) -> float:
        """Estimate monthly cost for a bucket based on storage class distribution"""
        # Simplified pricing per GB/month
        pricing = {
            'STANDARD': 0.023,
            'STANDARD_IA': 0.0125,
            'ONEZONE_IA': 0.01,
            'INTELLIGENT_TIERING': 0.023,  # Base tier
            'GLACIER_IR': 0.004,
            'GLACIER': 0.0036,
            'DEEP_ARCHIVE': 0.00099
        }
        
        total_cost = 0
        
        # If we have storage class distribution, use it
        if 'storage_class_distribution' in metrics:
            for storage_class, data in metrics['storage_class_distribution'].items():
                if storage_class in pricing:
                    total_cost += data['gb'] * pricing[storage_class]
        else:
            # Otherwise assume all STANDARD
            total_cost = metrics.get('size_gb', 0) * pricing['STANDARD']
        
        return round(total_cost, 2)
    
    def _determine_bucket_purpose(self, bucket_name: str, tags: Dict[str, str]) -> str:
        """Determine bucket purpose from name and tags"""
        name_lower = bucket_name.lower()
        
        # Check tags first
        if 'Purpose' in tags:
            return tags['Purpose']
        if 'Type' in tags:
            return tags['Type']
        
        # Check name patterns
        patterns = {
            'logs': ['logs', 'logging', 'log-', '-log'],
            'backup': ['backup', 'bak', 'archive', 'snapshot'],
            'website': ['www', 'static', 'assets', 'public', 'web'],
            'data': ['data', 'analytics', 'datalake', 'warehouse'],
            'temp': ['temp', 'tmp', 'temporary', 'scratch'],
            'config': ['config', 'configuration', 'settings'],
            'terraform': ['tfstate', 'terraform'],
            'cloudtrail': ['cloudtrail', 'trail'],
            'media': ['media', 'images', 'videos', 'assets']
        }
        
        for purpose, keywords in patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                return purpose
        
        return 'general'
    
    def _discover_access_points(self):
        """Discover S3 Access Points across all regions"""
        regions = self._get_all_regions()
        
        for region in regions:
            try:
                s3control = self.session.client('s3control', region_name=region)
                account_id = self.session.client('sts').get_caller_identity()['Account']
                
                paginator = s3control.get_paginator('list_access_points')
                
                for page in paginator.paginate(AccountId=account_id):
                    for ap in page.get('AccessPointList', []):
                        access_point = {
                            'resource_type': 's3_access_point',
                            'resource_id': ap['Name'],
                            'name': ap['Name'],
                            'bucket': ap['Bucket'],
                            'region': region,
                            'network_origin': ap.get('NetworkOrigin', 'Internet'),
                            'vpc_id': ap.get('VpcConfiguration', {}).get('VpcId'),
                            'creation_date': str(ap.get('CreationDate', '')),
                            'discovered_at': datetime.utcnow().isoformat()
                        }
                        
                        self.discovered_resources['access_points'].append(access_point)
                
            except Exception as e:
                logger.error(f"Error discovering access points in {region}: {e}")
    
    def _get_all_regions(self) -> List[str]:
        """Get all enabled regions"""
        try:
            ec2 = self.session.client('ec2')
            response = ec2.describe_regions()
            return [r['RegionName'] for r in response['Regions']]
        except Exception as e:
            logger.error(f"Failed to get regions: {e}")
            return ['us-east-1', 'us-west-2']
    
    def export_inventory(self, output_file: str = 's3_inventory.json'):
        """Export discovered inventory to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.discovered_resources, f, indent=2, default=str)
        
        logger.info(f"Exported S3 inventory to {output_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovered resources"""
        summary = {
            'total_buckets': len(self.discovered_resources['buckets']),
            'total_size_gb': sum(b.get('size_gb', 0) for b in self.discovered_resources['buckets']),
            'total_objects': sum(b.get('object_count', 0) for b in self.discovered_resources['buckets']),
            'total_estimated_monthly_cost': sum(b.get('estimated_monthly_cost', 0) for b in self.discovered_resources['buckets']),
            'buckets_by_region': defaultdict(int),
            'buckets_by_purpose': defaultdict(int),
            'public_buckets': 0,
            'encrypted_buckets': 0,
            'versioned_buckets': 0,
            'lifecycle_enabled_buckets': 0,
            'intelligent_tiering_buckets': 0
        }
        
        for bucket in self.discovered_resources['buckets']:
            summary['buckets_by_region'][bucket['region']] += 1
            summary['buckets_by_purpose'][bucket.get('purpose', 'unknown')] += 1
            
            if bucket.get('is_public'):
                summary['public_buckets'] += 1
            if bucket.get('encryption') not in ['None', 'Unknown']:
                summary['encrypted_buckets'] += 1
            if bucket.get('versioning_status') == 'Enabled':
                summary['versioned_buckets'] += 1
            if bucket.get('has_lifecycle_rules'):
                summary['lifecycle_enabled_buckets'] += 1
            if bucket.get('has_intelligent_tiering'):
                summary['intelligent_tiering_buckets'] += 1
        
        summary['buckets_by_region'] = dict(summary['buckets_by_region'])
        summary['buckets_by_purpose'] = dict(summary['buckets_by_purpose'])
        
        return summary
    
    def get_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify S3 optimization opportunities"""
        opportunities = []
        
        for bucket in self.discovered_resources['buckets']:
            # Large buckets without lifecycle policies
            if bucket.get('size_gb', 0) > 100 and not bucket.get('has_lifecycle_rules'):
                opportunities.append({
                    'type': 'missing_lifecycle_policy',
                    'bucket': bucket['bucket_name'],
                    'size_gb': bucket.get('size_gb', 0),
                    'recommendation': 'Add lifecycle policy to transition old objects',
                    'potential_savings': f"${bucket.get('estimated_monthly_cost', 0) * 0.3:.2f}/month"
                })
            
            # Large buckets without Intelligent-Tiering
            if bucket.get('size_gb', 0) > 1000 and not bucket.get('has_intelligent_tiering'):
                opportunities.append({
                    'type': 'no_intelligent_tiering',
                    'bucket': bucket['bucket_name'],
                    'size_gb': bucket.get('size_gb', 0),
                    'recommendation': 'Enable Intelligent-Tiering for automatic optimization',
                    'potential_savings': f"${bucket.get('estimated_monthly_cost', 0) * 0.2:.2f}/month"
                })
            
            # Unencrypted buckets
            if bucket.get('encryption') == 'None':
                opportunities.append({
                    'type': 'unencrypted_bucket',
                    'bucket': bucket['bucket_name'],
                    'recommendation': 'Enable default encryption',
                    'security_risk': 'high'
                })
            
            # Public buckets
            if bucket.get('is_public'):
                opportunities.append({
                    'type': 'public_bucket',
                    'bucket': bucket['bucket_name'],
                    'recommendation': 'Review and restrict public access if not needed',
                    'security_risk': 'critical'
                })
            
            # Old buckets with no recent access (simplified check)
            if bucket.get('age_days', 0) > 365 and bucket.get('object_count', 0) < 100:
                opportunities.append({
                    'type': 'potentially_unused_bucket',
                    'bucket': bucket['bucket_name'],
                    'age_days': bucket.get('age_days', 0),
                    'object_count': bucket.get('object_count', 0),
                    'recommendation': 'Review for deletion or archival',
                    'potential_savings': f"${bucket.get('estimated_monthly_cost', 0):.2f}/month"
                })
            
            # Logging buckets without lifecycle
            if bucket.get('purpose') == 'logs' and not bucket.get('has_lifecycle_rules'):
                opportunities.append({
                    'type': 'unoptimized_log_bucket',
                    'bucket': bucket['bucket_name'],
                    'recommendation': 'Add aggressive lifecycle policy for logs',
                    'potential_savings': f"${bucket.get('estimated_monthly_cost', 0) * 0.7:.2f}/month"
                })
        
        return opportunities