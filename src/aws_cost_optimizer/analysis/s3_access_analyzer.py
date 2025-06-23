"""
S3 Access Analyzer - Detects buckets with no access in 90+ days
Integrates with the existing S3 optimizer
"""
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)

class S3AccessAnalyzer:
    """Analyzes S3 bucket access patterns to identify unused buckets"""
    
    def __init__(self, session: Optional[boto3.Session] = None, 
                 no_access_days: int = 90):
        self.session = session or boto3.Session()
        self.no_access_days = no_access_days
        self.s3 = self.session.client('s3')
        self.cloudtrail = self.session.client('cloudtrail')
        self.s3_control = self.session.client('s3control')
        
    def analyze_bucket_access(self, bucket_name: str, 
                            check_cloudtrail: bool = True,
                            check_access_logs: bool = True) -> Dict[str, Any]:
        """Analyze access patterns for a single bucket"""
        logger.info(f"Analyzing access for bucket: {bucket_name}")
        
        access_info = {
            'bucket_name': bucket_name,
            'last_access_date': None,
            'days_since_access': None,
            'access_frequency': 'unknown',
            'data_sources_checked': [],
            'is_unused': False,
            'confidence_score': 0.0,
            'access_details': {}
        }
        
        try:
            # Method 1: Check CloudTrail for S3 access events
            if check_cloudtrail:
                cloudtrail_access = self._check_cloudtrail_access(bucket_name)
                if cloudtrail_access:
                    access_info['data_sources_checked'].append('cloudtrail')
                    access_info['access_details']['cloudtrail'] = cloudtrail_access
                    if cloudtrail_access['last_access']:
                        access_info['last_access_date'] = cloudtrail_access['last_access']
            
            # Method 2: Check S3 access logs if enabled
            if check_access_logs:
                log_access = self._check_access_logs(bucket_name)
                if log_access:
                    access_info['data_sources_checked'].append('access_logs')
                    access_info['access_details']['access_logs'] = log_access
                    if log_access['last_access'] and (not access_info['last_access_date'] or 
                        log_access['last_access'] > access_info['last_access_date']):
                        access_info['last_access_date'] = log_access['last_access']
            
            # Method 3: Check S3 Inventory if available
            inventory_access = self._check_s3_inventory(bucket_name)
            if inventory_access:
                access_info['data_sources_checked'].append('inventory')
                access_info['access_details']['inventory'] = inventory_access
            
            # Method 4: Check CloudWatch metrics
            metrics_access = self._check_cloudwatch_metrics(bucket_name)
            if metrics_access:
                access_info['data_sources_checked'].append('cloudwatch')
                access_info['access_details']['cloudwatch'] = metrics_access
            
            # Calculate days since last access
            if access_info['last_access_date']:
                days_since = (datetime.utcnow() - access_info['last_access_date']).days
                access_info['days_since_access'] = days_since
                access_info['is_unused'] = days_since > self.no_access_days
                
                # Determine access frequency
                if days_since < 7:
                    access_info['access_frequency'] = 'frequent'
                elif days_since < 30:
                    access_info['access_frequency'] = 'regular'
                elif days_since < 90:
                    access_info['access_frequency'] = 'occasional'
                else:
                    access_info['access_frequency'] = 'rare'
            else:
                # No access found in any data source
                access_info['is_unused'] = True
                access_info['days_since_access'] = float('inf')
                access_info['access_frequency'] = 'never'
            
            # Calculate confidence score
            access_info['confidence_score'] = self._calculate_confidence(access_info)
            
        except Exception as e:
            logger.error(f"Error analyzing bucket {bucket_name}: {e}")
            access_info['error'] = str(e)
        
        return access_info
    
    def _check_cloudtrail_access(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Check CloudTrail for S3 access events"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=min(self.no_access_days + 30, 90))
            
            events = []
            last_access = None
            
            # Look for S3 data events
            paginator = self.cloudtrail.get_paginator('lookup_events')
            
            for page in paginator.paginate(
                LookupAttributes=[
                    {
                        'AttributeKey': 'ResourceName',
                        'AttributeValue': f'arn:aws:s3:::{bucket_name}/'
                    }
                ],
                StartTime=start_time,
                EndTime=end_time
            ):
                for event in page.get('Events', []):
                    event_name = event.get('EventName', '')
                    
                    # Filter for actual access events (not just management)
                    if any(action in event_name for action in 
                          ['GetObject', 'PutObject', 'DeleteObject', 'ListObjects']):
                        events.append({
                            'event_name': event_name,
                            'event_time': event['EventTime'],
                            'user': event.get('Username', 'Unknown')
                        })
                        
                        if not last_access or event['EventTime'] > last_access:
                            last_access = event['EventTime']
            
            return {
                'last_access': last_access,
                'access_count': len(events),
                'recent_events': events[:5]  # Last 5 events
            }
            
        except Exception as e:
            logger.debug(f"CloudTrail check failed for {bucket_name}: {e}")
            return None
    
    def _check_access_logs(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Check S3 server access logs if enabled"""
        try:
            # Check if logging is enabled
            logging_config = self.s3.get_bucket_logging(Bucket=bucket_name)
            
            if 'LoggingEnabled' not in logging_config:
                return None
            
            target_bucket = logging_config['LoggingEnabled']['TargetBucket']
            target_prefix = logging_config['LoggingEnabled'].get('TargetPrefix', '')
            
            # Look for recent log files
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Check last week of logs
            
            last_access = None
            access_count = 0
            
            # List log objects
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=target_bucket,
                Prefix=f"{target_prefix}{bucket_name}"
            ):
                for obj in page.get('Contents', []):
                    # Parse log file timestamp from key
                    if obj['LastModified'] >= start_date:
                        # In production, would download and parse log file
                        # For now, use LastModified as proxy
                        if not last_access or obj['LastModified'] > last_access:
                            last_access = obj['LastModified']
                        access_count += 1
            
            return {
                'last_access': last_access,
                'log_files_found': access_count,
                'logging_enabled': True
            }
            
        except Exception as e:
            logger.debug(f"Access log check failed for {bucket_name}: {e}")
            return None
    
    def _check_s3_inventory(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Check S3 Inventory for object metadata"""
        try:
            # Check if inventory is configured
            account_id = self.session.client('sts').get_caller_identity()['Account']
            
            response = self.s3.list_bucket_inventory_configurations(Bucket=bucket_name)
            
            if 'InventoryConfigurationList' in response:
                return {
                    'inventory_configured': True,
                    'configurations': len(response['InventoryConfigurationList'])
                }
                
        except Exception as e:
            logger.debug(f"Inventory check failed for {bucket_name}: {e}")
            
        return None
    
    def _check_cloudwatch_metrics(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Check CloudWatch metrics for bucket activity"""
        try:
            cloudwatch = self.session.client('cloudwatch')
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=self.no_access_days)
            
            # Check request metrics
            metrics_to_check = [
                ('AllRequests', 'Count'),
                ('GetRequests', 'Count'),
                ('PutRequests', 'Count')
            ]
            
            total_requests = 0
            last_activity = None
            
            for metric_name, stat in metrics_to_check:
                response = cloudwatch.get_metric_statistics(
                    Namespace='AWS/S3',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'BucketName', 'Value': bucket_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # Daily
                    Statistics=['Sum']
                )
                
                for datapoint in response.get('Datapoints', []):
                    if datapoint['Sum'] > 0:
                        total_requests += datapoint['Sum']
                        if not last_activity or datapoint['Timestamp'] > last_activity:
                            last_activity = datapoint['Timestamp']
            
            if total_requests > 0:
                return {
                    'last_activity': last_activity,
                    'total_requests': total_requests,
                    'requests_per_day': total_requests / self.no_access_days
                }
                
        except Exception as e:
            logger.debug(f"CloudWatch check failed for {bucket_name}: {e}")
            
        return None
    
    def _calculate_confidence(self, access_info: Dict[str, Any]) -> float:
        """Calculate confidence score for access analysis"""
        confidence = 0.0
        
        # Base confidence on data sources checked
        if 'cloudtrail' in access_info['data_sources_checked']:
            confidence += 0.4
        if 'access_logs' in access_info['data_sources_checked']:
            confidence += 0.3
        if 'cloudwatch' in access_info['data_sources_checked']:
            confidence += 0.2
        if 'inventory' in access_info['data_sources_checked']:
            confidence += 0.1
        
        # Adjust based on findings
        if access_info['last_access_date']:
            confidence = min(confidence + 0.2, 1.0)
        
        return confidence
    
    def analyze_all_buckets(self, bucket_names: List[str] = None,
                          parallel: bool = True) -> List[Dict[str, Any]]:
        """Analyze access patterns for multiple buckets"""
        if not bucket_names:
            # Get all buckets
            response = self.s3.list_buckets()
            bucket_names = [b['Name'] for b in response['Buckets']]
        
        logger.info(f"Analyzing access for {len(bucket_names)} buckets")
        
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_bucket = {
                    executor.submit(self.analyze_bucket_access, bucket): bucket
                    for bucket in bucket_names
                }
                
                for future in as_completed(future_to_bucket):
                    bucket = future_to_bucket[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing {bucket}: {e}")
                        results.append({
                            'bucket_name': bucket,
                            'error': str(e),
                            'is_unused': None
                        })
        else:
            for bucket in bucket_names:
                results.append(self.analyze_bucket_access(bucket))
        
        # Sort by days since access (unused buckets first)
        results.sort(key=lambda x: x.get('days_since_access', 0), reverse=True)
        
        return results
    
    def generate_unused_buckets_report(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report of unused buckets"""
        unused_buckets = [r for r in analysis_results if r.get('is_unused', False)]
        
        report = {
            'summary': {
                'total_buckets_analyzed': len(analysis_results),
                'unused_buckets_count': len(unused_buckets),
                'unused_percentage': (len(unused_buckets) / len(analysis_results) * 100) if analysis_results else 0,
                'confidence_levels': {
                    'high': len([b for b in unused_buckets if b.get('confidence_score', 0) > 0.7]),
                    'medium': len([b for b in unused_buckets if 0.3 < b.get('confidence_score', 0) <= 0.7]),
                    'low': len([b for b in unused_buckets if b.get('confidence_score', 0) <= 0.3])
                }
            },
            'unused_buckets': unused_buckets,
            'recommendations': []
        }
        
        # Generate recommendations
        for bucket in unused_buckets:
            if bucket.get('days_since_access', 0) > 365:
                recommendation = {
                    'bucket_name': bucket['bucket_name'],
                    'action': 'archive_and_delete',
                    'reason': f"No access in {bucket.get('days_since_access', 'unknown')} days",
                    'risk_level': 'medium',
                    'steps': [
                        f"1. Create final backup of {bucket['bucket_name']} to Glacier",
                        "2. Notify stakeholders with 30-day warning",
                        "3. Delete bucket after confirmation"
                    ]
                }
            elif bucket.get('days_since_access', 0) > 180:
                recommendation = {
                    'bucket_name': bucket['bucket_name'],
                    'action': 'move_to_glacier',
                    'reason': f"No access in {bucket.get('days_since_access', 'unknown')} days",
                    'risk_level': 'low',
                    'steps': [
                        f"1. Apply lifecycle policy to transition to Glacier",
                        "2. Set up retrieval process documentation",
                        "3. Monitor for any access attempts"
                    ]
                }
            else:
                recommendation = {
                    'bucket_name': bucket['bucket_name'],
                    'action': 'apply_ia_lifecycle',
                    'reason': f"No access in {bucket.get('days_since_access', 'unknown')} days",
                    'risk_level': 'low',
                    'steps': [
                        f"1. Transition objects to Infrequent Access storage",
                        "2. Monitor access patterns",
                        "3. Consider Glacier after 6 months"
                    ]
                }
            
            report['recommendations'].append(recommendation)
        
        return report
    
    def enable_access_monitoring(self, bucket_name: str) -> Dict[str, bool]:
        """Enable various access monitoring methods for a bucket"""
        results = {
            'cloudtrail_data_events': False,
            'access_logging': False,
            'cloudwatch_metrics': False,
            'inventory': False
        }
        
        try:
            # Enable S3 access logging
            log_bucket = f"{bucket_name}-logs"
            
            # First create log bucket if it doesn't exist
            try:
                self.s3.create_bucket(Bucket=log_bucket)
            except:
                pass  # Bucket might already exist
            
            # Enable logging
            self.s3.put_bucket_logging(
                Bucket=bucket_name,
                BucketLoggingStatus={
                    'LoggingEnabled': {
                        'TargetBucket': log_bucket,
                        'TargetPrefix': f"{bucket_name}/"
                    }
                }
            )
            results['access_logging'] = True
            
        except Exception as e:
            logger.error(f"Failed to enable access logging: {e}")
        
        # CloudWatch request metrics are enabled by default for all buckets
        results['cloudwatch_metrics'] = True
        
        return results


def integrate_with_s3_optimizer(s3_optimizer_instance, no_access_days: int = 90):
    """
    Integration function to add access analysis to existing S3 optimizer
    
    Usage:
    from aws_cost_optimizer.optimization.s3_optimizer import S3Optimizer
    from aws_cost_optimizer.analysis.s3_access_analyzer import integrate_with_s3_optimizer
    
    s3_opt = S3Optimizer()
    integrate_with_s3_optimizer(s3_opt, no_access_days=90)
    
    # Now s3_opt has access analysis capabilities
    recommendations = s3_opt.analyze_all_buckets()
    """
    
    # Create access analyzer
    analyzer = S3AccessAnalyzer(
        session=s3_optimizer_instance.session,
        no_access_days=no_access_days
    )
    
    # Add access analyzer to S3 optimizer
    s3_optimizer_instance.access_analyzer = analyzer
    
    # Extend the analyze_bucket method
    original_analyze = s3_optimizer_instance.analyze_bucket
    
    def enhanced_analyze_bucket(bucket_name: str, bucket_info: Dict[str, Any]) -> Optional[Any]:
        # First get original recommendations
        recommendation = original_analyze(bucket_name, bucket_info)
        
        # Add access analysis
        access_info = analyzer.analyze_bucket_access(bucket_name)
        
        # If bucket is unused, create or update recommendation
        if access_info.get('is_unused', False):
            if not recommendation:
                # Create new recommendation for unused bucket
                from aws_cost_optimizer.optimization.models import S3OptimizationRecommendation
                
                recommendation = S3OptimizationRecommendation(
                    bucket_name=bucket_name,
                    current_storage_class='STANDARD',
                    recommended_storage_class='GLACIER',
                    action='archive_unused',
                    reason=f"No access detected in {access_info.get('days_since_access', 'unknown')} days",
                    estimated_monthly_savings=bucket_info.get('monthly_cost', 0) * 0.9,
                    confidence=access_info.get('confidence_score', 0.5),
                    risk_level='medium',
                    implementation_steps=[
                        "1. Verify bucket is truly unused with stakeholders",
                        "2. Create final backup if needed",
                        "3. Apply Glacier lifecycle policy or delete bucket"
                    ],
                    impact={
                        'size_gb': bucket_info.get('size_gb', 0),
                        'days_since_access': access_info.get('days_since_access'),
                        'last_access': str(access_info.get('last_access_date')) if access_info.get('last_access_date') else None
                    }
                )
            else:
                # Update existing recommendation with access info
                if hasattr(recommendation, 'impact'):
                    recommendation.impact.update({
                        'days_since_access': access_info.get('days_since_access'),
                        'last_access': str(access_info.get('last_access_date')) if access_info.get('last_access_date') else None,
                        'access_frequency': access_info.get('access_frequency')
                    })
        
        return recommendation
    
    # Replace the method
    s3_optimizer_instance.analyze_bucket = enhanced_analyze_bucket
    
    logger.info("S3 Access Analyzer integrated with S3 Optimizer")