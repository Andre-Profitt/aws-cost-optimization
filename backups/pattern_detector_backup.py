"""
Pattern detection module for analyzing AWS resource usage patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import boto3
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PatternDetector:
    """Detects usage patterns and anomalies in AWS resources"""
    
    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.cloudwatch = boto3.client('cloudwatch')
    
    def get_metric_statistics(self, 
                            namespace: str,
                            metric_name: str,
                            dimensions: List[Dict[str, str]],
                            stat: str = 'Average',
                            period: int = 3600) -> pd.DataFrame:
        """Fetch metric statistics from CloudWatch"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.lookback_days)
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[stat]
            )
            
            # Convert to DataFrame
            if response['Datapoints']:
                df = pd.DataFrame(response['Datapoints'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.sort_values('Timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return pd.DataFrame()
    
    def analyze_ec2_patterns(self, instance_id: str) -> Dict[str, Any]:
        """Analyze EC2 instance usage patterns"""
        dimensions = [{'Name': 'InstanceId', 'Value': instance_id}]
        
        # Get CPU utilization
        cpu_df = self.get_metric_statistics(
            'AWS/EC2', 'CPUUtilization', dimensions
        )
        
        # Get Network In
        network_in_df = self.get_metric_statistics(
            'AWS/EC2', 'NetworkIn', dimensions, stat='Sum'
        )
        
        # Get Network Out
        network_out_df = self.get_metric_statistics(
            'AWS/EC2', 'NetworkOut', dimensions, stat='Sum'
        )
        
        analysis = {
            'instance_id': instance_id,
            'usage_metrics': {},
            'patterns': {},
            'recommendations': []
        }
        
        # Analyze CPU patterns
        if not cpu_df.empty:
            cpu_stats = self._analyze_time_series(cpu_df['Average'].values)
            analysis['usage_metrics']['cpu_utilization_avg'] = cpu_stats['mean']
            analysis['usage_metrics']['cpu_utilization_p95'] = cpu_stats['p95']
            analysis['patterns']['cpu_pattern'] = cpu_stats['pattern']
            
            # Check for idle instance
            if cpu_stats['mean'] < 10:
                analysis['recommendations'].append({
                    'type': 'idle_instance',
                    'severity': 'high',
                    'message': f"Instance has average CPU of {cpu_stats['mean']:.1f}%"
                })
        
        # Analyze Network patterns
        if not network_in_df.empty:
            network_in_total = network_in_df['Sum'].sum()
            analysis['usage_metrics']['network_in_total'] = network_in_total
            
            # Low network usage check
            if network_in_total < 5 * 1024 * 1024:  # Less than 5MB
                analysis['recommendations'].append({
                    'type': 'low_network',
                    'severity': 'medium',
                    'message': f"Very low network usage: {network_in_total/1024/1024:.2f}MB total"
                })
        
        return analysis
    
    def analyze_rds_patterns(self, db_identifier: str) -> Dict[str, Any]:
        """Analyze RDS instance usage patterns"""
        dimensions = [{'Name': 'DBInstanceIdentifier', 'Value': db_identifier}]
        
        # Get CPU utilization
        cpu_df = self.get_metric_statistics(
            'AWS/RDS', 'CPUUtilization', dimensions
        )
        
        # Get Database connections
        connections_df = self.get_metric_statistics(
            'AWS/RDS', 'DatabaseConnections', dimensions
        )
        
        analysis = {
            'db_identifier': db_identifier,
            'usage_metrics': {},
            'patterns': {},
            'recommendations': []
        }
        
        # Analyze patterns
        if not cpu_df.empty:
            cpu_stats = self._analyze_time_series(cpu_df['Average'].values)
            analysis['usage_metrics']['cpu_utilization_avg'] = cpu_stats['mean']
            
            if cpu_stats['mean'] < 5:
                analysis['recommendations'].append({
                    'type': 'underutilized_rds',
                    'severity': 'high',
                    'message': f"RDS instance has very low CPU usage: {cpu_stats['mean']:.1f}%"
                })
        
        if not connections_df.empty:
            conn_stats = self._analyze_time_series(connections_df['Average'].values)
            analysis['usage_metrics']['avg_connections'] = conn_stats['mean']
            
            if conn_stats['mean'] < 1:
                analysis['recommendations'].append({
                    'type': 'unused_rds',
                    'severity': 'high',
                    'message': "RDS instance has almost no connections"
                })
        
        return analysis
    
    def _analyze_time_series(self, values: np.ndarray) -> Dict[str, Any]:
        """Analyze time series data for patterns"""
        if len(values) == 0:
            return {
                'mean': 0,
                'std': 0,
                'p95': 0,
                'pattern': 'no_data'
            }
        
        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'min': np.min(values),
            'max': np.max(values)
        }
        
        # Detect patterns
        if stats['std'] < 0.1 * stats['mean']:
            stats['pattern'] = 'steady'
        elif stats['max'] > 3 * stats['mean']:
            stats['pattern'] = 'spiky'
        else:
            stats['pattern'] = 'variable'
        
        return stats
    
    def detect_anomalies(self, values: np.ndarray, threshold: float = 3.0) -> List[int]:
        """Detect anomalies using z-score method"""
        if len(values) < 10:
            return []
        
        z_scores = np.abs(stats.zscore(values))
        return np.where(z_scores > threshold)[0].tolist()
    
    def generate_optimization_report(self, resources: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate optimization report for all resources"""
        recommendations = []
        
        for resource in resources:
            if resource['resource_type'] == 'EC2':
                analysis = self.analyze_ec2_patterns(resource['resource_id'])
                for rec in analysis['recommendations']:
                    recommendations.append({
                        'resource_id': resource['resource_id'],
                        'resource_type': 'EC2',
                        'account_name': resource['account_name'],
                        'region': resource['region'],
                        'recommendation_type': rec['type'],
                        'severity': rec['severity'],
                        'message': rec['message']
                    })
            
            elif resource['resource_type'] == 'RDS':
                analysis = self.analyze_rds_patterns(resource['resource_id'])
                for rec in analysis['recommendations']:
                    recommendations.append({
                        'resource_id': resource['resource_id'],
                        'resource_type': 'RDS',
                        'account_name': resource['account_name'],
                        'region': resource['region'],
                        'recommendation_type': rec['type'],
                        'severity': rec['severity'],
                        'message': rec['message']
                    })
        
        return pd.DataFrame(recommendations)