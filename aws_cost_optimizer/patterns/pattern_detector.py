import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
import numpy as np
from scipy import stats
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class PatternDetector:
    def __init__(self, lookback_days: int = 14):
        self.lookback_days = lookback_days
    
    def analyze_ec2_patterns(self, instance_id: str, region: str) -> Dict[str, Any]:
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.lookback_days)
        
        patterns = {}
        
        try:
            # Get CPU utilization data
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour granularity
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
                values = [dp['Average'] for dp in datapoints]
                timestamps = [dp['Timestamp'] for dp in datapoints]
                
                # Analyze patterns
                patterns = {
                    'patterns': {
                        'cpu_pattern': self._detect_pattern_type(values, timestamps),
                        'has_spikes': self._has_significant_spikes(values),
                        'is_periodic': self._is_periodic(values),
                        'daily_pattern': self._analyze_daily_pattern(datapoints),
                        'weekly_pattern': self._analyze_weekly_pattern(datapoints)
                    },
                    'statistics': {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    }
                }
        
        except ClientError as e:
            logger.error(f"Error analyzing patterns for instance {instance_id}: {e}")
            patterns = {'patterns': {}, 'statistics': {}}
        
        return patterns
    
    def _detect_pattern_type(self, values: List[float], timestamps: List[datetime]) -> str:
        if len(values) < 24:  # Need at least 24 hours of data
            return 'insufficient_data'
        
        # Check for constant low usage
        if max(values) < 5:
            return 'idle'
        
        # Check for periodic patterns
        if self._is_periodic(values):
            return 'periodic'
        
        # Check for steady usage
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        if cv < 0.3:  # Low coefficient of variation
            return 'steady'
        
        # Check for business hours pattern
        if self._has_business_hours_pattern(values, timestamps):
            return 'business_hours'
        
        return 'variable'
    
    def _has_significant_spikes(self, values: List[float]) -> bool:
        if len(values) < 10:
            return False
        
        mean = np.mean(values)
        std = np.std(values)
        
        # Check for values more than 3 standard deviations from mean
        spikes = [v for v in values if v > mean + 3 * std]
        
        return len(spikes) > 0
    
    def _is_periodic(self, values: List[float]) -> bool:
        if len(values) < 48:  # Need at least 2 days of hourly data
            return False
        
        # Try to detect daily periodicity (24-hour cycle)
        try:
            # Reshape data into days (if we have complete days)
            days = len(values) // 24
            if days >= 2:
                daily_data = np.array(values[:days*24]).reshape(days, 24)
                
                # Calculate correlation between days
                correlations = []
                for i in range(days - 1):
                    corr, _ = stats.pearsonr(daily_data[i], daily_data[i+1])
                    correlations.append(corr)
                
                # High correlation indicates periodicity
                avg_correlation = np.mean(correlations)
                return avg_correlation > 0.7
        
        except Exception as e:
            logger.debug(f"Error detecting periodicity: {e}")
        
        return False
    
    def _analyze_daily_pattern(self, datapoints: List[Dict[str, Any]]) -> Dict[str, float]:
        hourly_averages = {}
        
        for dp in datapoints:
            hour = dp['Timestamp'].hour
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(dp['Average'])
        
        # Calculate average for each hour
        hourly_pattern = {}
        for hour, values in hourly_averages.items():
            hourly_pattern[f'hour_{hour}'] = np.mean(values)
        
        return hourly_pattern
    
    def _analyze_weekly_pattern(self, datapoints: List[Dict[str, Any]]) -> Dict[str, float]:
        daily_averages = {}
        
        for dp in datapoints:
            day = dp['Timestamp'].strftime('%A')
            if day not in daily_averages:
                daily_averages[day] = []
            daily_averages[day].append(dp['Average'])
        
        # Calculate average for each day
        weekly_pattern = {}
        for day, values in daily_averages.items():
            weekly_pattern[day] = np.mean(values)
        
        return weekly_pattern
    
    def _has_business_hours_pattern(self, values: List[float], timestamps: List[datetime]) -> bool:
        if len(values) < 168:  # Need at least a week of hourly data
            return False
        
        business_hours_usage = []
        off_hours_usage = []
        
        for value, timestamp in zip(values, timestamps):
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            # Business hours: Monday-Friday, 8 AM - 6 PM
            if weekday < 5 and 8 <= hour < 18:
                business_hours_usage.append(value)
            else:
                off_hours_usage.append(value)
        
        if not business_hours_usage or not off_hours_usage:
            return False
        
        # Check if business hours usage is significantly higher
        business_avg = np.mean(business_hours_usage)
        off_hours_avg = np.mean(off_hours_usage)
        
        # Business hours should be at least 2x higher than off hours
        return business_avg > 2 * off_hours_avg and business_avg > 20
    
    def analyze_cost_patterns(self, account_id: str, lookback_days: int = 30) -> Dict[str, Any]:
        ce = boto3.client('ce')
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # Get daily cost data
            response = ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            # Analyze cost trends
            daily_costs = []
            service_costs = {}
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                total_cost = 0
                
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    
                    if service not in service_costs:
                        service_costs[service] = []
                    service_costs[service].append(cost)
                    total_cost += cost
                
                daily_costs.append(total_cost)
            
            # Detect anomalies
            anomalies = self._detect_cost_anomalies(daily_costs)
            
            # Analyze service trends
            service_trends = {}
            for service, costs in service_costs.items():
                if sum(costs) > 100:  # Only analyze services with significant spend
                    service_trends[service] = self._analyze_cost_trend(costs)
            
            return {
                'total_spend': sum(daily_costs),
                'average_daily_spend': np.mean(daily_costs),
                'cost_trend': self._analyze_cost_trend(daily_costs),
                'anomalies': anomalies,
                'service_trends': service_trends
            }
        
        except ClientError as e:
            logger.error(f"Error analyzing cost patterns: {e}")
            return {}
    
    def _detect_cost_anomalies(self, costs: List[float]) -> List[Dict[str, Any]]:
        if len(costs) < 7:
            return []
        
        anomalies = []
        mean = np.mean(costs)
        std = np.std(costs)
        
        for i, cost in enumerate(costs):
            z_score = (cost - mean) / std if std > 0 else 0
            
            if abs(z_score) > 2:  # More than 2 standard deviations
                anomalies.append({
                    'day_index': i,
                    'cost': cost,
                    'z_score': z_score,
                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                })
        
        return anomalies
    
    def _analyze_cost_trend(self, costs: List[float]) -> Dict[str, Any]:
        if len(costs) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate linear regression
        x = np.arange(len(costs))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, costs)
        
        # Determine trend
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate percentage change
        pct_change = ((costs[-1] - costs[0]) / costs[0] * 100) if costs[0] > 0 else 0
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'percentage_change': pct_change,
            'projected_30_days': intercept + slope * (len(costs) + 30)
        }