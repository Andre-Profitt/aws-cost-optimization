"""
Cost Anomaly Detector - Real-time detection of unusual AWS spending patterns
Uses statistical analysis and machine learning to identify cost spikes and anomalies
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass 
class CostAnomaly:
    """Represents a detected cost anomaly"""
    anomaly_id: str
    detection_date: datetime
    service: str
    region: Optional[str]
    anomaly_type: str  # 'spike', 'gradual_increase', 'new_service', 'unusual_pattern'
    severity: str  # 'low', 'medium', 'high', 'critical'
    current_daily_cost: float
    expected_daily_cost: float
    cost_impact: float  # Additional cost
    percentage_increase: float
    confidence_score: float
    probable_causes: List[str]
    affected_resources: List[str]
    recommended_actions: List[str]
    alert_sent: bool = False

@dataclass
class CostTrend:
    """Represents a cost trend analysis"""
    service: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 score
    weekly_change_rate: float
    monthly_change_rate: float
    forecast_next_month: float
    confidence_interval: Tuple[float, float]

class CostAnomalyDetector:
    """Advanced cost anomaly detection system"""
    
    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        'low': {'percentage': 20, 'absolute': 50},      # 20% or $50
        'medium': {'percentage': 50, 'absolute': 200},   # 50% or $200
        'high': {'percentage': 100, 'absolute': 500},    # 100% or $500
        'critical': {'percentage': 200, 'absolute': 1000} # 200% or $1000
    }
    
    def __init__(self,
                 lookback_days: int = 90,
                 anomaly_threshold: float = 2.5,  # Z-score threshold
                 min_daily_spend: float = 10,     # Minimum spend to analyze
                 session: Optional[boto3.Session] = None):
        """
        Initialize Cost Anomaly Detector
        
        Args:
            lookback_days: Days of historical data for baseline
            anomaly_threshold: Statistical threshold for anomaly detection
            min_daily_spend: Minimum daily spend to consider
            session: Boto3 session
        """
        self.lookback_days = lookback_days
        self.anomaly_threshold = anomaly_threshold
        self.min_daily_spend = min_daily_spend
        self.session = session or boto3.Session()
        self.ce = self.session.client('ce')
        self.sns = self.session.client('sns')
        self.cloudwatch = self.session.client('cloudwatch')
        
        # Initialize ML model for pattern detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, 
                        real_time: bool = True,
                        services_filter: List[str] = None) -> List[CostAnomaly]:
        """
        Detect cost anomalies across all services
        
        Args:
            real_time: Whether to check today's costs (real-time)
            services_filter: List of services to analyze (None = all)
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        logger.info("Starting anomaly detection...")
        
        # Get historical cost data
        cost_data = self._get_cost_data(services_filter)
        
        # Detect anomalies by service
        for service in cost_data['services']:
            service_anomalies = self._analyze_service_costs(service, cost_data['data'])
            anomalies.extend(service_anomalies)
        
        # Detect cross-service anomalies
        cross_service_anomalies = self._detect_cross_service_anomalies(cost_data['data'])
        anomalies.extend(cross_service_anomalies)
        
        # Check for new services
        new_service_anomalies = self._detect_new_services(cost_data['data'])
        anomalies.extend(new_service_anomalies)
        
        # Real-time detection for today
        if real_time:
            today_anomalies = self._detect_realtime_anomalies()
            anomalies.extend(today_anomalies)
        
        # Sort by severity and impact
        anomalies.sort(key=lambda x: (
            self._severity_rank(x.severity),
            x.cost_impact
        ), reverse=True)
        
        return anomalies
    
    def _get_cost_data(self, services_filter: List[str] = None) -> Dict[str, Any]:
        """Get historical cost data from Cost Explorer"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        try:
            # Build filter
            filter_dict = {}
            if services_filter:
                filter_dict = {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': services_filter
                    }
                }
            
            # Get daily costs by service
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ],
                Filter=filter_dict if filter_dict else None
            )
            
            # Parse response into DataFrame
            data = []
            services = set()
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                
                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    region = group['Keys'][1] if len(group['Keys']) > 1 else 'global'
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    usage = float(group['Metrics']['UsageQuantity']['Amount'])
                    
                    services.add(service)
                    
                    data.append({
                        'date': pd.to_datetime(date),
                        'service': service,
                        'region': region,
                        'cost': cost,
                        'usage': usage
                    })
            
            df = pd.DataFrame(data)
            
            # Also get cost by resource tags for better insights
            tag_response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': (end_date - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'TAG', 'Key': 'Environment'},
                    {'Type': 'TAG', 'Key': 'Application'}
                ]
            )
            
            return {
                'data': df,
                'services': list(services),
                'tag_data': tag_response
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost data: {e}")
            return {'data': pd.DataFrame(), 'services': [], 'tag_data': {}}
    
    def _analyze_service_costs(self, service: str, df: pd.DataFrame) -> List[CostAnomaly]:
        """Analyze costs for a specific service"""
        anomalies = []
        
        # Filter data for this service
        service_data = df[df['service'] == service].copy()
        
        if service_data.empty:
            return anomalies
        
        # Aggregate daily costs
        daily_costs = service_data.groupby('date')['cost'].sum().reset_index()
        daily_costs = daily_costs.sort_values('date')
        
        # Skip if spend is too low
        if daily_costs['cost'].mean() < self.min_daily_spend:
            return anomalies
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(
            daily_costs, service
        )
        anomalies.extend(statistical_anomalies)
        
        # ML-based pattern anomaly detection
        if len(daily_costs) > 30:  # Need enough data
            pattern_anomalies = self._detect_pattern_anomalies(
                daily_costs, service
            )
            anomalies.extend(pattern_anomalies)
        
        # Detect gradual increases
        trend_anomalies = self._detect_trend_anomalies(
            daily_costs, service
        )
        anomalies.extend(trend_anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, 
                                    daily_costs: pd.DataFrame,
                                    service: str) -> List[CostAnomaly]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        # Calculate rolling statistics
        daily_costs['rolling_mean'] = daily_costs['cost'].rolling(window=7, min_periods=1).mean()
        daily_costs['rolling_std'] = daily_costs['cost'].rolling(window=7, min_periods=1).std()
        
        # Z-score calculation
        daily_costs['z_score'] = (daily_costs['cost'] - daily_costs['rolling_mean']) / (daily_costs['rolling_std'] + 1e-10)
        
        # Detect anomalies
        anomaly_mask = abs(daily_costs['z_score']) > self.anomaly_threshold
        
        for idx, row in daily_costs[anomaly_mask].iterrows():
            # Skip if it's the most recent incomplete day
            if row['date'] == daily_costs['date'].max() and datetime.utcnow().hour < 20:
                continue
                
            cost_impact = row['cost'] - row['rolling_mean']
            percentage_increase = (cost_impact / row['rolling_mean']) * 100
            
            # Determine severity
            severity = self._determine_severity(percentage_increase, cost_impact)
            
            # Identify probable causes
            probable_causes = self._identify_probable_causes(
                service, row['date'], cost_impact
            )
            
            # Get affected resources
            affected_resources = self._get_affected_resources(
                service, row['date']
            )
            
            anomaly = CostAnomaly(
                anomaly_id=f"{service}-{row['date'].strftime('%Y%m%d')}-stat",
                detection_date=datetime.utcnow(),
                service=service,
                region=None,
                anomaly_type='spike',
                severity=severity,
                current_daily_cost=row['cost'],
                expected_daily_cost=row['rolling_mean'],
                cost_impact=cost_impact,
                percentage_increase=percentage_increase,
                confidence_score=min(abs(row['z_score']) / 5, 1.0),  # Normalize to 0-1
                probable_causes=probable_causes,
                affected_resources=affected_resources,
                recommended_actions=self._generate_recommendations(
                    service, 'spike', probable_causes
                )
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self,
                                 daily_costs: pd.DataFrame,
                                 service: str) -> List[CostAnomaly]:
        """Detect anomalies using ML pattern recognition"""
        anomalies = []
        
        try:
            # Prepare features for ML model
            features = self._prepare_ml_features(daily_costs)
            
            if features.shape[0] < 10:  # Not enough data
                return anomalies
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train and predict
            self.isolation_forest.fit(features_scaled[:-7])  # Train on all but last week
            predictions = self.isolation_forest.predict(features_scaled)
            anomaly_scores = self.isolation_forest.score_samples(features_scaled)
            
            # Check last 7 days for anomalies
            for i in range(max(0, len(predictions) - 7), len(predictions)):
                if predictions[i] == -1:  # Anomaly detected
                    row = daily_costs.iloc[i]
                    
                    # Calculate baseline from similar days
                    day_of_week = row['date'].dayofweek
                    similar_days = daily_costs[
                        daily_costs['date'].dt.dayofweek == day_of_week
                    ]['cost'].iloc[:-1]  # Exclude current
                    
                    if len(similar_days) > 0:
                        expected_cost = similar_days.mean()
                        cost_impact = row['cost'] - expected_cost
                        percentage_increase = (cost_impact / expected_cost) * 100
                        
                        if abs(percentage_increase) > 15:  # Significant enough
                            anomaly = CostAnomaly(
                                anomaly_id=f"{service}-{row['date'].strftime('%Y%m%d')}-pattern",
                                detection_date=datetime.utcnow(),
                                service=service,
                                region=None,
                                anomaly_type='unusual_pattern',
                                severity=self._determine_severity(percentage_increase, cost_impact),
                                current_daily_cost=row['cost'],
                                expected_daily_cost=expected_cost,
                                cost_impact=cost_impact,
                                percentage_increase=percentage_increase,
                                confidence_score=abs(anomaly_scores[i]),
                                probable_causes=['Unusual usage pattern detected by ML model'],
                                affected_resources=[],
                                recommended_actions=[
                                    'Review service usage patterns',
                                    'Check for configuration changes',
                                    'Verify auto-scaling settings'
                                ]
                            )
                            anomalies.append(anomaly)
                            
        except Exception as e:
            logger.error(f"ML pattern detection failed: {e}")
            
        return anomalies
    
    def _prepare_ml_features(self, daily_costs: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model"""
        df = daily_costs.copy()
        
        # Add time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add rolling features
        for window in [3, 7, 14]:
            df[f'rolling_mean_{window}'] = df['cost'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['cost'].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Add lag features
        for lag in [1, 7]:
            df[f'cost_lag_{lag}'] = df['cost'].shift(lag).fillna(df['cost'].mean())
        
        # Select features
        feature_cols = [
            'cost', 'day_of_week', 'day_of_month', 'is_weekend',
            'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14',
            'rolling_std_3', 'rolling_std_7', 'rolling_std_14',
            'cost_lag_1', 'cost_lag_7'
        ]
        
        return df[feature_cols].fillna(0).values
    
    def _detect_trend_anomalies(self,
                               daily_costs: pd.DataFrame,
                               service: str) -> List[CostAnomaly]:
        """Detect gradual cost increases"""
        anomalies = []
        
        if len(daily_costs) < 14:  # Need at least 2 weeks
            return anomalies
        
        # Calculate weekly averages
        daily_costs['week'] = daily_costs['date'].dt.isocalendar().week
        weekly_avg = daily_costs.groupby('week')['cost'].mean()
        
        if len(weekly_avg) < 4:  # Need at least 4 weeks
            return anomalies
        
        # Calculate trend
        weeks = np.arange(len(weekly_avg))
        slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, weekly_avg.values)
        
        # Check if significant upward trend
        if slope > 0 and p_value < 0.05:
            # Calculate percentage increase per week
            avg_cost = weekly_avg.mean()
            weekly_increase_pct = (slope / avg_cost) * 100
            
            # If increasing more than 5% per week, it's concerning
            if weekly_increase_pct > 5:
                current_cost = weekly_avg.iloc[-1]
                initial_cost = weekly_avg.iloc[0]
                total_increase = current_cost - initial_cost
                total_increase_pct = (total_increase / initial_cost) * 100
                
                anomaly = CostAnomaly(
                    anomaly_id=f"{service}-trend-{datetime.utcnow().strftime('%Y%m%d')}",
                    detection_date=datetime.utcnow(),
                    service=service,
                    region=None,
                    anomaly_type='gradual_increase',
                    severity='medium' if weekly_increase_pct < 10 else 'high',
                    current_daily_cost=daily_costs['cost'].iloc[-1],
                    expected_daily_cost=initial_cost,
                    cost_impact=total_increase,
                    percentage_increase=total_increase_pct,
                    confidence_score=abs(r_value),  # R-squared as confidence
                    probable_causes=[
                        f'Steady cost increase of {weekly_increase_pct:.1f}% per week',
                        'Possible resource leak or scaling issue',
                        'May indicate growing usage without optimization'
                    ],
                    affected_resources=[],
                    recommended_actions=[
                        'Review auto-scaling policies',
                        'Check for unused resources accumulating',
                        'Analyze usage growth vs business growth',
                        'Implement cost optimization measures'
                    ]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_cross_service_anomalies(self, df: pd.DataFrame) -> List[CostAnomaly]:
        """Detect anomalies across multiple services"""
        anomalies = []
        
        # Check total daily costs
        total_daily = df.groupby('date')['cost'].sum().reset_index()
        
        if len(total_daily) > 7:
            # Use same statistical approach on total
            total_anomalies = self._detect_statistical_anomalies(
                total_daily, 'TOTAL_ACCOUNT'
            )
            
            # Adjust anomaly details for cross-service
            for anomaly in total_anomalies:
                anomaly.anomaly_type = 'cross_service'
                anomaly.probable_causes = [
                    'Multiple services showing increased costs',
                    'Possible account-wide issue or event',
                    'Check for large-scale deployments or migrations'
                ]
                
            anomalies.extend(total_anomalies)
        
        return anomalies
    
    def _detect_new_services(self, df: pd.DataFrame) -> List[CostAnomaly]:
        """Detect when new services start incurring costs"""
        anomalies = []
        
        # Group by service and find first appearance
        service_first_date = df.groupby('service')['date'].min()
        
        # Check last 7 days for new services
        recent_date = datetime.utcnow().date() - timedelta(days=7)
        new_services = service_first_date[service_first_date > pd.Timestamp(recent_date)]
        
        for service, first_date in new_services.items():
            # Get costs for this new service
            service_costs = df[df['service'] == service]['cost'].sum()
            daily_avg = service_costs / ((datetime.utcnow().date() - first_date.date()).days + 1)
            
            if daily_avg > self.min_daily_spend:
                anomaly = CostAnomaly(
                    anomaly_id=f"new-service-{service}-{first_date.strftime('%Y%m%d')}",
                    detection_date=datetime.utcnow(),
                    service=service,
                    region=None,
                    anomaly_type='new_service',
                    severity='low' if daily_avg < 100 else 'medium',
                    current_daily_cost=daily_avg,
                    expected_daily_cost=0,
                    cost_impact=daily_avg,
                    percentage_increase=100,  # New service
                    confidence_score=1.0,
                    probable_causes=[
                        f'New service {service} started on {first_date.strftime("%Y-%m-%d")}',
                        'Could be intentional or accidental enablement'
                    ],
                    affected_resources=[],
                    recommended_actions=[
                        f'Verify if {service} was intentionally enabled',
                        'Review service configuration and usage',
                        'Set up cost alerts for this service',
                        'Consider setting budget limits'
                    ]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_realtime_anomalies(self) -> List[CostAnomaly]:
        """Detect anomalies in today's costs (real-time)"""
        anomalies = []
        
        try:
            # Get today's costs so far
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': today.strftime('%Y-%m-%d'),
                    'End': (today + timedelta(days=1)).strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            # Get yesterday's costs for comparison
            yesterday_response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': yesterday.strftime('%Y-%m-%d'),
                    'End': today.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            # Compare costs
            if response['ResultsByTime'] and yesterday_response['ResultsByTime']:
                today_data = response['ResultsByTime'][0]
                yesterday_data = yesterday_response['ResultsByTime'][0]
                
                # Adjust for time of day
                hours_elapsed = datetime.utcnow().hour
                time_factor = hours_elapsed / 24
                
                for group in today_data.get('Groups', []):
                    service = group['Keys'][0]
                    today_cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    projected_cost = today_cost / time_factor if time_factor > 0 else today_cost
                    
                    # Find yesterday's cost for this service
                    yesterday_cost = 0
                    for y_group in yesterday_data.get('Groups', []):
                        if y_group['Keys'][0] == service:
                            yesterday_cost = float(y_group['Metrics']['UnblendedCost']['Amount'])
                            break
                    
                    # Check if projection is anomalous
                    if yesterday_cost > 0:
                        increase_pct = ((projected_cost - yesterday_cost) / yesterday_cost) * 100
                        
                        if increase_pct > 50:  # 50% increase threshold
                            anomaly = CostAnomaly(
                                anomaly_id=f"realtime-{service}-{today.strftime('%Y%m%d')}",
                                detection_date=datetime.utcnow(),
                                service=service,
                                region=None,
                                anomaly_type='spike',
                                severity=self._determine_severity(increase_pct, projected_cost - yesterday_cost),
                                current_daily_cost=projected_cost,
                                expected_daily_cost=yesterday_cost,
                                cost_impact=projected_cost - yesterday_cost,
                                percentage_increase=increase_pct,
                                confidence_score=min(time_factor, 0.8),  # Lower confidence early in day
                                probable_causes=[
                                    'Real-time cost spike detected',
                                    'Unusual activity in last few hours'
                                ],
                                affected_resources=[],
                                recommended_actions=[
                                    'URGENT: Check service dashboard immediately',
                                    'Look for runaway processes or misconfigurations',
                                    'Consider implementing emergency cost controls'
                                ]
                            )
                            anomalies.append(anomaly)
                            
        except Exception as e:
            logger.error(f"Real-time detection failed: {e}")
            
        return anomalies
    
    def _determine_severity(self, percentage_increase: float, absolute_increase: float) -> str:
        """Determine anomaly severity based on percentage and absolute increase"""
        for severity in ['critical', 'high', 'medium', 'low']:
            thresholds = self.SEVERITY_THRESHOLDS[severity]
            if (abs(percentage_increase) >= thresholds['percentage'] or 
                abs(absolute_increase) >= thresholds['absolute']):
                return severity
        return 'low'
    
    def _severity_rank(self, severity: str) -> int:
        """Convert severity to numeric rank for sorting"""
        ranks = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return ranks.get(severity, 0)
    
    def _identify_probable_causes(self, 
                                service: str,
                                date: pd.Timestamp,
                                cost_impact: float) -> List[str]:
        """Identify probable causes for cost anomaly"""
        causes = []
        
        # Service-specific causes
        service_causes = {
            'Amazon Elastic Compute Cloud - Compute': [
                'New EC2 instances launched',
                'Instance type changes',
                'Spot instance price spikes',
                'Auto-scaling triggered'
            ],
            'Amazon Simple Storage Service': [
                'Large data upload/download',
                'S3 request spike',
                'Data transfer costs',
                'Storage class transitions'
            ],
            'Amazon Relational Database Service': [
                'Database scaling',
                'Backup storage increase',
                'Multi-AZ deployment',
                'Read replica addition'
            ],
            'AWS Lambda': [
                'Function invocation spike',
                'Increased execution duration',
                'Memory configuration changes',
                'New function deployments'
            ]
        }
        
        if service in service_causes:
            causes.extend(service_causes[service][:2])  # Top 2 causes
        else:
            causes.append(f'Unusual activity in {service}')
        
        # Add date-specific context
        if date.weekday() == 0:  # Monday
            causes.append('Monday traffic spike')
        elif date.weekday() in [5, 6]:  # Weekend
            causes.append('Unusual weekend activity')
        
        # Add cost-specific context
        if cost_impact > 1000:
            causes.append('Large-scale resource deployment possible')
        
        return causes
    
    def _get_affected_resources(self, service: str, date: pd.Timestamp) -> List[str]:
        """Get specific resources contributing to anomaly"""
        # This would query CloudTrail or Resource tagging API
        # Placeholder implementation
        return [f"{service}-resource-1", f"{service}-resource-2"]
    
    def _generate_recommendations(self,
                                service: str,
                                anomaly_type: str,
                                probable_causes: List[str]) -> List[str]:
        """Generate actionable recommendations based on anomaly"""
        recommendations = []
        
        # General recommendations
        recommendations.extend([
            f'Review {service} dashboard and metrics',
            'Check recent deployments and changes',
            'Set up CloudWatch alarms for this metric'
        ])
        
        # Type-specific recommendations
        if anomaly_type == 'spike':
            recommendations.extend([
                'Look for runaway processes or loops',
                'Check auto-scaling settings',
                'Review recent configuration changes'
            ])
        elif anomaly_type == 'gradual_increase':
            recommendations.extend([
                'Analyze long-term usage trends',
                'Implement resource cleanup policies',
                'Consider Reserved Instances or Savings Plans'
            ])
        elif anomaly_type == 'new_service':
            recommendations.extend([
                'Verify service was intentionally enabled',
                'Review IAM permissions',
                'Set up cost budgets and alerts'
            ])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def send_alerts(self, anomalies: List[CostAnomaly], sns_topic_arn: str = None):
        """Send alerts for detected anomalies"""
        if not sns_topic_arn:
            logger.warning("No SNS topic ARN provided for alerts")
            return
            
        for anomaly in anomalies:
            if anomaly.alert_sent:
                continue
                
            # Only alert for medium+ severity
            if anomaly.severity in ['medium', 'high', 'critical']:
                try:
                    message = self._format_alert_message(anomaly)
                    
                    self.sns.publish(
                        TopicArn=sns_topic_arn,
                        Subject=f"AWS Cost Anomaly Detected - {anomaly.severity.upper()}",
                        Message=message,
                        MessageAttributes={
                            'severity': {'DataType': 'String', 'StringValue': anomaly.severity},
                            'service': {'DataType': 'String', 'StringValue': anomaly.service},
                            'cost_impact': {'DataType': 'Number', 'StringValue': str(anomaly.cost_impact)}
                        }
                    )
                    
                    anomaly.alert_sent = True
                    logger.info(f"Alert sent for anomaly: {anomaly.anomaly_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
    
    def _format_alert_message(self, anomaly: CostAnomaly) -> str:
        """Format anomaly alert message"""
        return f"""
AWS COST ANOMALY DETECTED

Severity: {anomaly.severity.upper()}
Service: {anomaly.service}
Type: {anomaly.anomaly_type.replace('_', ' ').title()}

Current Daily Cost: ${anomaly.current_daily_cost:,.2f}
Expected Daily Cost: ${anomaly.expected_daily_cost:,.2f}
Cost Impact: ${anomaly.cost_impact:,.2f} ({anomaly.percentage_increase:+.1f}%)

Probable Causes:
{chr(10).join(f'• {cause}' for cause in anomaly.probable_causes)}

Recommended Actions:
{chr(10).join(f'{i+1}. {action}' for i, action in enumerate(anomaly.recommended_actions))}

Detection Time: {anomaly.detection_date.strftime('%Y-%m-%d %H:%M:%S UTC')}
Confidence Score: {anomaly.confidence_score:.0%}

---
This is an automated alert from AWS Cost Anomaly Detector
"""
    
    def create_cost_dashboard(self, anomalies: List[CostAnomaly]):
        """Create CloudWatch dashboard for cost anomalies"""
        try:
            widgets = []
            
            # Summary widget
            summary_widget = {
                'type': 'metric',
                'properties': {
                    'metrics': [
                        ['AWS/Billing', 'EstimatedCharges', {'stat': 'Maximum'}]
                    ],
                    'period': 300,
                    'stat': 'Average',
                    'region': 'us-east-1',
                    'title': 'Total AWS Costs'
                }
            }
            widgets.append(summary_widget)
            
            # Add widgets for each anomaly
            for anomaly in anomalies[:5]:  # Top 5
                anomaly_widget = {
                    'type': 'metric',
                    'properties': {
                        'metrics': [
                            ['AWS/Billing', 'EstimatedCharges', 
                             {'Service': anomaly.service}, 
                             {'stat': 'Maximum'}]
                        ],
                        'period': 3600,
                        'stat': 'Average', 
                        'region': 'us-east-1',
                        'title': f'{anomaly.service} - Anomaly Detected',
                        'annotations': {
                            'horizontal': [{
                                'label': 'Expected',
                                'value': anomaly.expected_daily_cost
                            }]
                        }
                    }
                }
                widgets.append(anomaly_widget)
            
            dashboard_body = {
                'widgets': widgets
            }
            
            self.cloudwatch.put_dashboard(
                DashboardName='CostAnomalyDetector',
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info("Cost anomaly dashboard created/updated")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def export_anomaly_report(self,
                            anomalies: List[CostAnomaly],
                            output_file: str = 'cost_anomaly_report.xlsx'):
        """Export detailed anomaly report"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for anomaly in anomalies:
                summary_data.append({
                    'Detection Date': anomaly.detection_date.strftime('%Y-%m-%d %H:%M'),
                    'Service': anomaly.service,
                    'Type': anomaly.anomaly_type.replace('_', ' ').title(),
                    'Severity': anomaly.severity.upper(),
                    'Current Cost': f"${anomaly.current_daily_cost:,.2f}",
                    'Expected Cost': f"${anomaly.expected_daily_cost:,.2f}",
                    'Impact': f"${anomaly.cost_impact:,.2f}",
                    'Increase %': f"{anomaly.percentage_increase:+.1f}%",
                    'Confidence': f"{anomaly.confidence_score:.0%}"
                })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Anomaly Summary', index=False)
            
            # Detailed analysis by severity
            for severity in ['critical', 'high', 'medium', 'low']:
                severity_anomalies = [a for a in anomalies if a.severity == severity]
                
                if severity_anomalies:
                    detail_data = []
                    for anomaly in severity_anomalies:
                        detail_data.append({
                            'Service': anomaly.service,
                            'Daily Impact': f"${anomaly.cost_impact:.2f}",
                            'Monthly Impact': f"${anomaly.cost_impact * 30:,.2f}",
                            'Probable Causes': '\n'.join(anomaly.probable_causes),
                            'Recommendations': '\n'.join(anomaly.recommended_actions)
                        })
                    
                    pd.DataFrame(detail_data).to_excel(
                        writer, 
                        sheet_name=f'{severity.title()} Severity',
                        index=False
                    )
            
        logger.info(f"Anomaly report exported to {output_file}")