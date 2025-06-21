"""
S3 Optimizer module for intelligent storage optimization
Targets: Intelligent-Tiering, lifecycle policies, duplicate data, access patterns
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class S3OptimizationRecommendation:
    """Represents an S3 optimization recommendation"""
    bucket_name: str
    current_storage_class: str
    recommended_storage_class: Optional[str]
    action: str
    estimated_monthly_savings: float
    confidence: float
    reason: str
    risk_level: str  # low, medium, high
    implementation_steps: List[str]
    impact: Dict[str, Any]  # size_gb, object_count, etc.

@dataclass
class BucketAnalysis:
    """Analysis results for a single bucket"""
    bucket_name: str
    total_size_bytes: int
    object_count: int
    storage_class_distribution: Dict[str, Dict[str, Any]]
    access_pattern: str  # frequent, infrequent, archive, unknown
    lifecycle_rules_exist: bool
    intelligent_tiering_enabled: bool
    versioning_enabled: bool
    tags: Dict[str, str]
    monthly_cost: float

class S3Optimizer:
    """Comprehensive S3 storage optimization engine"""
    
    # S3 storage pricing per GB/month (simplified)
    STORAGE_PRICING = {
        'STANDARD': 0.023,
        'STANDARD_IA': 0.0125,
        'ONEZONE_IA': 0.01,
        'INTELLIGENT_TIERING': 0.023,  # Base tier
        'GLACIER_IR': 0.004,
        'GLACIER': 0.0036,
        'DEEP_ARCHIVE': 0.00099
    }
    
    # Minimum object age for transitions (days)
    MIN_TRANSITION_AGE = {
        'STANDARD_IA': 30,
        'ONEZONE_IA': 30,
        'INTELLIGENT_TIERING': 0,
        'GLACIER_IR': 0,
        'GLACIER': 90,
        'DEEP_ARCHIVE': 180
    }
    
    def __init__(self, 
                 size_threshold_gb: float = 1024,  # 1TB
                 observation_days: int = 90,
                 session: Optional[boto3.Session] = None):
        """
        Initialize S3 Optimizer
        
        Args:
            size_threshold_gb: Minimum bucket size to recommend Intelligent-Tiering
            observation_days: Days of metrics to analyze
            session: Boto3 session (optional)
        """
        self.size_threshold_gb = size_threshold_gb
        self.observation_days = observation_days
        self.session = session or boto3.Session()
        self.s3 = self.session.client('s3')
        self.cloudwatch = self.session.client('cloudwatch')
        self.s3_analytics = self.session.client('s3')
        
    def analyze_all_buckets(self, region_name: str = None) -> List[S3OptimizationRecommendation]:
        """
        Analyze all S3 buckets in the account
        
        Args:
            region_name: AWS region (uses session default if None)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Get all buckets
        buckets = self._get_all_buckets()
        logger.info(f"Found {len(buckets)} S3 buckets to analyze")
        
        # Analyze each bucket
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_bucket = {
                executor.submit(self._analyze_single_bucket, bucket): bucket 
                for bucket in buckets
            }
            
            for future in as_completed(future_to_bucket):
                bucket = future_to_bucket[future]
                try:
                    bucket_analysis, bucket_recommendations = future.result()
                    recommendations.extend(bucket_recommendations)
                except Exception as e:
                    logger.error(f"Failed to analyze {bucket['Name']}: {e}")
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.estimated_monthly_savings, reverse=True)
        
        return recommendations
    
    def _get_all_buckets(self) -> List[Dict[str, Any]]:
        """Get all S3 buckets"""
        response = self.s3.list_buckets()
        return response.get('Buckets', [])
    
    def _analyze_single_bucket(self, bucket: Dict[str, Any]) -> Tuple[BucketAnalysis, List[S3OptimizationRecommendation]]:
        """Analyze a single bucket for optimization opportunities"""
        bucket_name = bucket['Name']
        recommendations = []
        
        try:
            # Get bucket location to ensure we're in the right region
            location = self.s3.get_bucket_location(Bucket=bucket_name)
            bucket_region = location.get('LocationConstraint') or 'us-east-1'
            
            # Create region-specific client if needed
            if bucket_region != self.session.region_name:
                regional_s3 = self.session.client('s3', region_name=bucket_region)
            else:
                regional_s3 = self.s3
            
            # Perform bucket analysis
            analysis = self._get_bucket_analysis(bucket_name, regional_s3)
            
            # Generate recommendations based on analysis
            
            # 1. Check for Intelligent-Tiering opportunity
            it_rec = self._check_intelligent_tiering(analysis)
            if it_rec:
                recommendations.append(it_rec)
            
            # 2. Check for lifecycle policy opportunities
            lifecycle_rec = self._check_lifecycle_policies(analysis)
            if lifecycle_rec:
                recommendations.append(lifecycle_rec)
            
            # 3. Check for old backup cleanup
            backup_rec = self._check_old_backups(analysis)
            if backup_rec:
                recommendations.append(backup_rec)
            
            # 4. Check for incomplete multipart uploads
            multipart_rec = self._check_incomplete_uploads(bucket_name, regional_s3)
            if multipart_rec:
                recommendations.append(multipart_rec)
            
            # 5. Check for versioning optimization
            version_rec = self._check_versioning_optimization(analysis)
            if version_rec:
                recommendations.append(version_rec)
            
            # 6. Check for duplicate data in non-prod
            duplicate_rec = self._check_duplicate_data(analysis)
            if duplicate_rec:
                recommendations.append(duplicate_rec)
            
            return analysis, recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing bucket {bucket_name}: {e}")
            return None, []
    
    def _get_bucket_analysis(self, bucket_name: str, s3_client: Any) -> BucketAnalysis:
        """Perform comprehensive analysis of a bucket"""
        # Get bucket tags
        try:
            tags_response = s3_client.get_bucket_tagging(Bucket=bucket_name)
            tags = {tag['Key']: tag['Value'] for tag in tags_response.get('TagSet', [])}
        except:
            tags = {}
        
        # Get versioning status
        try:
            versioning = s3_client.get_bucket_versioning(Bucket=bucket_name)
            versioning_enabled = versioning.get('Status') == 'Enabled'
        except:
            versioning_enabled = False
        
        # Check lifecycle rules
        try:
            lifecycle = s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
            lifecycle_rules_exist = len(lifecycle.get('Rules', [])) > 0
            intelligent_tiering_enabled = self._has_intelligent_tiering_rule(lifecycle.get('Rules', []))
        except:
            lifecycle_rules_exist = False
            intelligent_tiering_enabled = False
        
        # Get storage metrics
        storage_metrics = self._get_storage_metrics(bucket_name)
        
        # Calculate storage class distribution
        storage_distribution = self._get_storage_class_distribution(bucket_name, s3_client)
        
        # Analyze access patterns
        access_pattern = self._analyze_access_pattern(bucket_name, storage_metrics)
        
        # Calculate monthly cost
        monthly_cost = self._calculate_bucket_cost(storage_distribution)
        
        return BucketAnalysis(
            bucket_name=bucket_name,
            total_size_bytes=storage_metrics['total_size'],
            object_count=storage_metrics['object_count'],
            storage_class_distribution=storage_distribution,
            access_pattern=access_pattern,
            lifecycle_rules_exist=lifecycle_rules_exist,
            intelligent_tiering_enabled=intelligent_tiering_enabled,
            versioning_enabled=versioning_enabled,
            tags=tags,
            monthly_cost=monthly_cost
        )
    
    def _get_storage_metrics(self, bucket_name: str) -> Dict[str, Any]:
        """Get storage metrics from CloudWatch"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        metrics = {
            'total_size': 0,
            'object_count': 0
        }
        
        # Get bucket size
        try:
            size_response = self.cloudwatch.get_metric_statistics(
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
                metrics['total_size'] = int(size_response['Datapoints'][0]['Average'])
        except Exception as e:
            logger.error(f"Failed to get size metrics for {bucket_name}: {e}")
        
        # Get object count
        try:
            count_response = self.cloudwatch.get_metric_statistics(
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
        except Exception as e:
            logger.error(f"Failed to get object count for {bucket_name}: {e}")
        
        return metrics
    
    def _get_storage_class_distribution(self, bucket_name: str, s3_client: Any) -> Dict[str, Dict[str, Any]]:
        """Get distribution of objects by storage class"""
        distribution = defaultdict(lambda: {'size_bytes': 0, 'object_count': 0})
        
        # Sample objects to get distribution (full scan would be expensive)
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            
            sample_size = 1000  # Sample first 1000 objects
            objects_sampled = 0
            
            for page in paginator.paginate(Bucket=bucket_name):
                for obj in page.get('Contents', []):
                    storage_class = obj.get('StorageClass', 'STANDARD')
                    distribution[storage_class]['size_bytes'] += obj['Size']
                    distribution[storage_class]['object_count'] += 1
                    
                    objects_sampled += 1
                    if objects_sampled >= sample_size:
                        break
                
                if objects_sampled >= sample_size:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to get storage distribution for {bucket_name}: {e}")
        
        return dict(distribution)
    
    def _analyze_access_pattern(self, bucket_name: str, storage_metrics: Dict[str, Any]) -> str:
        """Analyze bucket access patterns based on CloudWatch metrics"""
        # This is simplified - in production you'd use S3 Storage Class Analysis
        # or CloudTrail logs for more accurate patterns
        
        # Get request metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        
        try:
            # Get GET requests
            get_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='GetRequests',
                Dimensions=[{'Name': 'BucketName', 'Value': bucket_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,
                Statistics=['Sum']
            )
            
            if get_response['Datapoints']:
                total_requests = sum(dp['Sum'] for dp in get_response['Datapoints'])
                avg_daily_requests = total_requests / 30
                
                # Classify based on requests per GB
                size_gb = storage_metrics['total_size'] / (1024**3)
                if size_gb > 0:
                    requests_per_gb = avg_daily_requests / size_gb
                    
                    if requests_per_gb > 10:
                        return 'frequent'
                    elif requests_per_gb > 1:
                        return 'infrequent'
                    else:
                        return 'archive'
                        
        except Exception as e:
            logger.error(f"Failed to analyze access pattern for {bucket_name}: {e}")
        
        return 'unknown'
    
    def _has_intelligent_tiering_rule(self, rules: List[Dict[str, Any]]) -> bool:
        """Check if bucket has Intelligent-Tiering lifecycle rule"""
        for rule in rules:
            if rule.get('Status') == 'Enabled':
                for transition in rule.get('Transitions', []):
                    if transition.get('StorageClass') == 'INTELLIGENT_TIERING':
                        return True
        return False
    
    def _check_intelligent_tiering(self, analysis: BucketAnalysis) -> Optional[S3OptimizationRecommendation]:
        """Check if bucket should enable Intelligent-Tiering"""
        size_gb = analysis.total_size_bytes / (1024**3)
        
        # Check if bucket is large enough and doesn't have IT enabled
        if (size_gb >= self.size_threshold_gb and 
            not analysis.intelligent_tiering_enabled and
            analysis.access_pattern in ['frequent', 'infrequent', 'unknown']):
            
            # Calculate potential savings (30-50% for infrequent access)
            current_standard_cost = (analysis.storage_class_distribution.get('STANDARD', {}).get('size_bytes', 0) / (1024**3)) * self.STORAGE_PRICING['STANDARD']
            
            # IT can save 30-50% on infrequently accessed data
            estimated_savings = current_standard_cost * 0.3
            
            return S3OptimizationRecommendation(
                bucket_name=analysis.bucket_name,
                current_storage_class='STANDARD',
                recommended_storage_class='INTELLIGENT_TIERING',
                action='enable_intelligent_tiering',
                estimated_monthly_savings=estimated_savings,
                confidence=0.9,
                reason=f"Large bucket ({size_gb:.1f} GB) without Intelligent-Tiering",
                risk_level='low',
                implementation_steps=[
                    f"aws s3api put-bucket-lifecycle-configuration --bucket {analysis.bucket_name} --lifecycle-configuration file://it-policy.json",
                    "Monitor cost savings in S3 Storage Lens",
                    "No application changes required"
                ],
                impact={
                    'size_gb': size_gb,
                    'object_count': analysis.object_count,
                    'current_cost': analysis.monthly_cost
                }
            )
        
        return None
    
    def _check_lifecycle_policies(self, analysis: BucketAnalysis) -> Optional[S3OptimizationRecommendation]:
        """Check if bucket needs lifecycle policies"""
        size_gb = analysis.total_size_bytes / (1024**3)
        
        # Check if bucket has no lifecycle rules and is large enough
        if not analysis.lifecycle_rules_exist and size_gb > 10:
            # Estimate savings based on typical aging pattern
            # Assume 30% moves to IA, 20% to Glacier eventually
            standard_gb = size_gb
            
            ia_savings = (standard_gb * 0.3) * (self.STORAGE_PRICING['STANDARD'] - self.STORAGE_PRICING['STANDARD_IA'])
            glacier_savings = (standard_gb * 0.2) * (self.STORAGE_PRICING['STANDARD'] - self.STORAGE_PRICING['GLACIER_IR'])
            
            total_savings = ia_savings + glacier_savings
            
            return S3OptimizationRecommendation(
                bucket_name=analysis.bucket_name,
                current_storage_class='STANDARD',
                recommended_storage_class='TIERED',
                action='add_lifecycle_policy',
                estimated_monthly_savings=total_savings,
                confidence=0.8,
                reason=f"No lifecycle policy on {size_gb:.1f} GB bucket",
                risk_level='low',
                implementation_steps=[
                    "1. Create lifecycle policy JSON file",
                    f"2. Apply policy: aws s3api put-bucket-lifecycle-configuration --bucket {analysis.bucket_name} --lifecycle-configuration file://lifecycle.json",
                    "3. Policy transitions: Standard -> Standard-IA (30d) -> Glacier IR (90d)",
                    "4. Monitor transitions in S3 console"
                ],
                impact={
                    'size_gb': size_gb,
                    'estimated_ia_gb': size_gb * 0.3,
                    'estimated_glacier_gb': size_gb * 0.2
                }
            )
        
        return None
    
    def _check_old_backups(self, analysis: BucketAnalysis) -> Optional[S3OptimizationRecommendation]:
        """Check for old backup data that can be archived or deleted"""
        bucket_name = analysis.bucket_name.lower()
        
        # Check if this looks like a backup bucket
        is_backup = any(keyword in bucket_name for keyword in ['backup', 'bak', 'archive', 'snapshot'])
        is_backup = is_backup or analysis.tags.get('Type', '').lower() == 'backup'
        
        if is_backup and analysis.total_size_bytes > 1024**3:  # > 1GB
            size_gb = analysis.total_size_bytes / (1024**3)
            
            # Check if mostly STANDARD storage (not already optimized)
            standard_pct = (analysis.storage_class_distribution.get('STANDARD', {}).get('size_bytes', 0) / 
                          analysis.total_size_bytes if analysis.total_size_bytes > 0 else 0)
            
            if standard_pct > 0.5:
                # Recommend aggressive archival for backups
                glacier_savings = size_gb * standard_pct * (self.STORAGE_PRICING['STANDARD'] - self.STORAGE_PRICING['GLACIER'])
                
                return S3OptimizationRecommendation(
                    bucket_name=analysis.bucket_name,
                    current_storage_class='STANDARD',
                    recommended_storage_class='GLACIER',
                    action='archive_old_backups',
                    estimated_monthly_savings=glacier_savings,
                    confidence=0.85,
                    reason=f"Backup bucket with {size_gb:.1f} GB in STANDARD storage",
                    risk_level='medium',
                    implementation_steps=[
                        "1. Verify backup retention requirements with data owners",
                        "2. Create aggressive lifecycle policy for backups",
                        "3. Transition: Standard -> Glacier (1d) -> Deep Archive (30d)",
                        "4. Set expiration based on retention policy"
                    ],
                    impact={
                        'size_gb': size_gb,
                        'standard_storage_gb': size_gb * standard_pct
                    }
                )
        
        return None
    
    def _check_incomplete_uploads(self, bucket_name: str, s3_client: Any) -> Optional[S3OptimizationRecommendation]:
        """Check for incomplete multipart uploads"""
        try:
            # List multipart uploads
            response = s3_client.list_multipart_uploads(Bucket=bucket_name)
            uploads = response.get('Uploads', [])
            
            if uploads:
                # Calculate wasted space (estimate)
                wasted_gb = len(uploads) * 0.1  # Assume 100MB average per incomplete upload
                wasted_cost = wasted_gb * self.STORAGE_PRICING['STANDARD']
                
                return S3OptimizationRecommendation(
                    bucket_name=bucket_name,
                    current_storage_class='STANDARD',
                    recommended_storage_class=None,
                    action='cleanup_multipart_uploads',
                    estimated_monthly_savings=wasted_cost,
                    confidence=0.95,
                    reason=f"Found {len(uploads)} incomplete multipart uploads",
                    risk_level='low',
                    implementation_steps=[
                        f"1. Review uploads: aws s3api list-multipart-uploads --bucket {bucket_name}",
                        f"2. Abort old uploads: aws s3api abort-multipart-upload --bucket {bucket_name} --key KEY --upload-id ID",
                        "3. Add lifecycle rule to auto-cleanup after 7 days"
                    ],
                    impact={
                        'incomplete_uploads': len(uploads),
                        'estimated_wasted_gb': wasted_gb
                    }
                )
        except Exception as e:
            logger.error(f"Failed to check multipart uploads for {bucket_name}: {e}")
        
        return None
    
    def _check_versioning_optimization(self, analysis: BucketAnalysis) -> Optional[S3OptimizationRecommendation]:
        """Check if versioned bucket needs optimization"""
        if analysis.versioning_enabled:
            # Check if there's a noncurrent version expiration policy
            # This is simplified - would need to check actual lifecycle rules
            
            size_gb = analysis.total_size_bytes / (1024**3)
            # Assume 20% of storage is old versions
            version_overhead = size_gb * 0.2
            potential_savings = version_overhead * self.STORAGE_PRICING['STANDARD']
            
            return S3OptimizationRecommendation(
                bucket_name=analysis.bucket_name,
                current_storage_class='STANDARD',
                recommended_storage_class='TIERED',
                action='optimize_versioning',
                estimated_monthly_savings=potential_savings,
                confidence=0.7,
                reason=f"Versioned bucket without version lifecycle policy",
                risk_level='medium',
                implementation_steps=[
                    "1. Analyze version usage patterns",
                    "2. Add noncurrent version transitions",
                    "3. Transition old versions: Standard -> IA (30d) -> Glacier (90d)",
                    "4. Set noncurrent version expiration (365d)"
                ],
                impact={
                    'size_gb': size_gb,
                    'estimated_version_gb': version_overhead
                }
            )
        
        return None
    
    def _check_duplicate_data(self, analysis: BucketAnalysis) -> Optional[S3OptimizationRecommendation]:
        """Check for potential duplicate data in non-production environments"""
        env = analysis.tags.get('Environment', '').lower()
        
        if env in ['dev', 'test', 'staging'] and analysis.total_size_bytes > 100 * (1024**3):  # > 100GB
            size_gb = analysis.total_size_bytes / (1024**3)
            
            # Non-prod environments often have duplicate data
            estimated_duplicates = size_gb * 0.3  # Assume 30% duplicates
            potential_savings = estimated_duplicates * self.STORAGE_PRICING['STANDARD']
            
            return S3OptimizationRecommendation(
                bucket_name=analysis.bucket_name,
                current_storage_class='STANDARD',
                recommended_storage_class=None,
                action='deduplicate_data',
                estimated_monthly_savings=potential_savings,
                confidence=0.6,
                reason=f"Large non-prod bucket ({env}) likely has duplicate data",
                risk_level='medium',
                implementation_steps=[
                    "1. Run S3 Inventory to catalog all objects",
                    "2. Analyze for duplicate content using checksums",
                    "3. Implement deduplication strategy",
                    "4. Use S3 Batch Operations for cleanup"
                ],
                impact={
                    'size_gb': size_gb,
                    'estimated_duplicate_gb': estimated_duplicates,
                    'environment': env
                }
            )
        
        return None
    
    def _calculate_bucket_cost(self, storage_distribution: Dict[str, Dict[str, Any]]) -> float:
        """Calculate monthly cost for a bucket based on storage distribution"""
        total_cost = 0
        
        for storage_class, info in storage_distribution.items():
            size_gb = info['size_bytes'] / (1024**3)
            price_per_gb = self.STORAGE_PRICING.get(storage_class, self.STORAGE_PRICING['STANDARD'])
            total_cost += size_gb * price_per_gb
        
        return total_cost
    
    def generate_optimization_report(self, recommendations: List[S3OptimizationRecommendation]) -> pd.DataFrame:
        """Generate a detailed optimization report"""
        data = []
        
        for rec in recommendations:
            data.append({
                'Bucket': rec.bucket_name,
                'Current Storage': rec.current_storage_class,
                'Recommended Storage': rec.recommended_storage_class or 'N/A',
                'Action': rec.action,
                'Monthly Savings': f"${rec.estimated_monthly_savings:,.2f}",
                'Annual Savings': f"${rec.estimated_monthly_savings * 12:,.2f}",
                'Confidence': f"{rec.confidence:.0%}",
                'Risk Level': rec.risk_level,
                'Reason': rec.reason,
                'Size (GB)': rec.impact.get('size_gb', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Add summary
        total_monthly = sum(rec.estimated_monthly_savings for rec in recommendations)
        total_annual = total_monthly * 12
        
        print(f"\nS3 Optimization Summary:")
        print(f"Total Recommendations: {len(recommendations)}")
        print(f"Total Monthly Savings: ${total_monthly:,.2f}")
        print(f"Total Annual Savings: ${total_annual:,.2f}")
        print(f"\nTop 5 Opportunities:")
        print(df.head())
        
        return df
    
    def export_recommendations(self, 
                             recommendations: List[S3OptimizationRecommendation],
                             output_file: str = 's3_optimization_report.xlsx'):
        """Export recommendations to Excel with multiple sheets"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self.generate_optimization_report(recommendations)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed implementation
            impl_data = []
            for rec in recommendations:
                impl_data.append({
                    'Bucket': rec.bucket_name,
                    'Action': rec.action,
                    'Monthly Savings': rec.estimated_monthly_savings,
                    'Implementation Steps': '\n'.join(rec.implementation_steps),
                    'Size Impact (GB)': rec.impact.get('size_gb', 0),
                    'Object Count': rec.impact.get('object_count', 0)
                })
            
            impl_df = pd.DataFrame(impl_data)
            impl_df.to_excel(writer, sheet_name='Implementation Guide', index=False)
            
            # Action priority matrix
            action_data = []
            for action in set(rec.action for rec in recommendations):
                action_recs = [r for r in recommendations if r.action == action]
                action_data.append({
                    'Action Type': action.replace('_', ' ').title(),
                    'Count': len(action_recs),
                    'Total Monthly Savings': sum(r.estimated_monthly_savings for r in action_recs),
                    'Average Confidence': np.mean([r.confidence for r in action_recs])
                })
            
            action_df = pd.DataFrame(action_data)
            action_df.sort_values('Total Monthly Savings', ascending=False, inplace=True)
            action_df.to_excel(writer, sheet_name='Actions by Impact', index=False)
            
        logger.info(f"Exported S3 optimization report to {output_file}")
    
    def generate_cli_commands(self, recommendations: List[S3OptimizationRecommendation], 
                            output_file: str = 's3_optimization_commands.sh'):
        """Generate CLI commands for implementing recommendations"""
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# S3 Optimization Implementation Commands\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            
            # Group by action type
            by_action = defaultdict(list)
            for rec in recommendations:
                by_action[rec.action].append(rec)
            
            # Generate commands for each action type
            for action, recs in by_action.items():
                f.write(f"\n# {action.replace('_', ' ').upper()}\n")
                f.write(f"# Potential savings: ${sum(r.estimated_monthly_savings for r in recs):,.2f}/month\n\n")
                
                if action == 'enable_intelligent_tiering':
                    f.write("# Create Intelligent-Tiering lifecycle policy\n")
                    f.write("cat > it-policy.json << 'EOF'\n")
                    f.write(self._generate_it_policy())
                    f.write("\nEOF\n\n")
                    
                    for rec in recs:
                        f.write(f"# Enable for {rec.bucket_name} ({rec.impact.get('size_gb', 0):.1f} GB)\n")
                        f.write(f"aws s3api put-bucket-lifecycle-configuration --bucket {rec.bucket_name} --lifecycle-configuration file://it-policy.json\n\n")
                
                elif action == 'add_lifecycle_policy':
                    f.write("# Create standard lifecycle policy\n")
                    f.write("cat > lifecycle-policy.json << 'EOF'\n")
                    f.write(self._generate_standard_lifecycle_policy())
                    f.write("\nEOF\n\n")
                    
                    for rec in recs:
                        f.write(f"# Apply to {rec.bucket_name}\n")
                        f.write(f"aws s3api put-bucket-lifecycle-configuration --bucket {rec.bucket_name} --lifecycle-configuration file://lifecycle-policy.json\n\n")
                
                elif action == 'cleanup_multipart_uploads':
                    for rec in recs:
                        f.write(f"# Cleanup incomplete uploads in {rec.bucket_name}\n")
                        f.write(f"# List uploads first:\n")
                        f.write(f"aws s3api list-multipart-uploads --bucket {rec.bucket_name}\n")
                        f.write(f"# Then abort old uploads (example):\n")
                        f.write(f"# aws s3api abort-multipart-upload --bucket {rec.bucket_name} --key <KEY> --upload-id <ID>\n\n")
        
        logger.info(f"Generated CLI commands in {output_file}")
    
    def _generate_it_policy(self) -> str:
        """Generate Intelligent-Tiering lifecycle policy JSON"""
        policy = {
            "Rules": [{
                "ID": "IntelligentTieringRule",
                "Status": "Enabled",
                "Transitions": [{
                    "Days": 0,
                    "StorageClass": "INTELLIGENT_TIERING"
                }]
            }]
        }
        return json.dumps(policy, indent=2)
    
    def _generate_standard_lifecycle_policy(self) -> str:
        """Generate standard lifecycle policy JSON"""
        policy = {
            "Rules": [{
                "ID": "StandardLifecyclePolicy",
                "Status": "Enabled",
                "Transitions": [
                    {"Days": 30, "StorageClass": "STANDARD_IA"},
                    {"Days": 90, "StorageClass": "GLACIER_IR"},
                    {"Days": 180, "StorageClass": "DEEP_ARCHIVE"}
                ],
                "NoncurrentVersionTransitions": [
                    {"Days": 7, "StorageClass": "STANDARD_IA"},
                    {"Days": 30, "StorageClass": "GLACIER"}
                ],
                "AbortIncompleteMultipartUpload": {
                    "DaysAfterInitiation": 7
                }
            }]
        }
        return json.dumps(policy, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = S3Optimizer(
        size_threshold_gb=1024,  # 1TB
        observation_days=90
    )
    
    # Analyze all buckets
    recommendations = optimizer.analyze_all_buckets()
    
    # Generate reports
    optimizer.export_recommendations(recommendations, 's3_optimization_report.xlsx')
    optimizer.generate_cli_commands(recommendations, 's3_optimization_commands.sh')