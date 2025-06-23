"""
ML-based Cost Prediction System

Uses machine learning models to predict future costs and detect anomalies
before they occur, enabling proactive cost optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import boto3
import pickle
import logging
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of cost predictions"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SERVICE_LEVEL = "service_level"
    RESOURCE_LEVEL = "resource_level"


class AnomalyType(Enum):
    """Types of cost anomalies"""
    SPIKE = "spike"
    GRADUAL_INCREASE = "gradual_increase"
    UNUSUAL_PATTERN = "unusual_pattern"
    NEW_RESOURCE = "new_resource"
    SERVICE_SURGE = "service_surge"


@dataclass
class CostPrediction:
    """Represents a cost prediction"""
    prediction_date: datetime
    prediction_type: PredictionType
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    service_breakdown: Optional[Dict[str, float]] = None
    drivers: List[str] = field(default_factory=list)
    
    
@dataclass
class PredictedAnomaly:
    """Represents a predicted cost anomaly"""
    anomaly_date: datetime
    anomaly_type: AnomalyType
    service: str
    predicted_impact: float
    probability: float
    description: str
    recommended_actions: List[str]
    alert_priority: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: str
    last_trained: datetime
    mse: float
    rmse: float
    mape: float
    r2_score: float
    training_samples: int
    feature_importance: Dict[str, float]


class CostPredictor:
    """ML-based cost prediction and anomaly detection"""
    
    def __init__(self,
                 lookback_days: int = 90,
                 forecast_days: int = 30,
                 anomaly_threshold: float = 0.95,
                 ce_client=None,
                 s3_client=None):
        """
        Initialize cost predictor
        
        Args:
            lookback_days: Days of historical data to use
            forecast_days: Days to forecast into future
            anomaly_threshold: Threshold for anomaly detection (0-1)
            ce_client: Boto3 Cost Explorer client
            s3_client: Boto3 S3 client for model storage
        """
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.anomaly_threshold = anomaly_threshold
        self.ce_client = ce_client or boto3.client('ce')
        self.s3_client = s3_client or boto3.client('s3')
        
        # ML models
        self.daily_predictor = None
        self.service_predictor = None
        self.anomaly_detector = None
        self.scalers = {}
        
        # Model metadata
        self.model_performance = {}
        self.feature_columns = []
        
    def train_models(self, 
                    historical_data: Optional[pd.DataFrame] = None,
                    model_bucket: Optional[str] = None) -> Dict[str, ModelPerformance]:
        """
        Train all prediction models
        
        Args:
            historical_data: Optional pre-fetched historical data
            model_bucket: S3 bucket to save trained models
            
        Returns:
            Dictionary of model performance metrics
        """
        logger.info("Training cost prediction models...")
        
        # Fetch historical data if not provided
        if historical_data is None:
            historical_data = self._fetch_historical_costs()
        
        # Prepare features
        features_df = self._engineer_features(historical_data)
        
        # Train daily cost predictor
        daily_performance = self._train_daily_predictor(features_df)
        self.model_performance['daily'] = daily_performance
        
        # Train service-level predictor
        service_performance = self._train_service_predictor(features_df)
        self.model_performance['service'] = service_performance
        
        # Train anomaly detector
        anomaly_performance = self._train_anomaly_detector(features_df)
        self.model_performance['anomaly'] = anomaly_performance
        
        # Save models if bucket specified
        if model_bucket:
            self._save_models(model_bucket)
        
        logger.info(f"Model training complete. Daily MAPE: {daily_performance.mape:.2%}")
        
        return self.model_performance
    
    def predict_costs(self,
                     prediction_type: PredictionType = PredictionType.DAILY,
                     include_breakdown: bool = True) -> List[CostPrediction]:
        """
        Predict future costs
        
        Args:
            prediction_type: Type of prediction to generate
            include_breakdown: Include service-level breakdown
            
        Returns:
            List of cost predictions
        """
        if not self.daily_predictor:
            raise ValueError("Models not trained. Call train_models() first.")
        
        predictions = []
        
        # Get recent data for context
        recent_data = self._fetch_recent_costs(days=30)
        recent_features = self._engineer_features(recent_data)
        
        # Generate predictions based on type
        if prediction_type == PredictionType.DAILY:
            predictions = self._predict_daily_costs(recent_features, include_breakdown)
        elif prediction_type == PredictionType.WEEKLY:
            predictions = self._predict_weekly_costs(recent_features, include_breakdown)
        elif prediction_type == PredictionType.MONTHLY:
            predictions = self._predict_monthly_costs(recent_features, include_breakdown)
        elif prediction_type == PredictionType.SERVICE_LEVEL:
            predictions = self._predict_service_costs(recent_features)
        
        return predictions
    
    def detect_future_anomalies(self,
                               forecast_days: Optional[int] = None) -> List[PredictedAnomaly]:
        """
        Detect potential future cost anomalies
        
        Args:
            forecast_days: Days to look ahead (default: self.forecast_days)
            
        Returns:
            List of predicted anomalies
        """
        if not self.anomaly_detector:
            raise ValueError("Anomaly detector not trained. Call train_models() first.")
        
        forecast_days = forecast_days or self.forecast_days
        anomalies = []
        
        # Get predictions
        predictions = self.predict_costs(PredictionType.DAILY, include_breakdown=True)
        
        # Analyze each prediction for anomalies
        for pred in predictions[:forecast_days]:
            # Check for cost spikes
            if pred.predicted_cost > pred.confidence_interval[1]:
                anomalies.append(self._create_spike_anomaly(pred))
            
            # Check service-level anomalies
            if pred.service_breakdown:
                service_anomalies = self._detect_service_anomalies(pred)
                anomalies.extend(service_anomalies)
        
        # Detect gradual increases
        gradual_anomalies = self._detect_gradual_increases(predictions)
        anomalies.extend(gradual_anomalies)
        
        # Sort by priority and date
        anomalies.sort(key=lambda x: (x.alert_priority, x.anomaly_date))
        
        return anomalies
    
    def _fetch_historical_costs(self) -> pd.DataFrame:
        """Fetch historical cost data from Cost Explorer"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Fetch daily costs by service
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        )
        
        # Parse response into DataFrame
        data = []
        for result in response['ResultsByTime']:
            date = pd.to_datetime(result['TimePeriod']['Start'])
            
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                usage = float(group['Metrics']['UsageQuantity']['Amount'] or 0)
                
                data.append({
                    'date': date,
                    'service': service,
                    'cost': cost,
                    'usage': usage
                })
        
        df = pd.DataFrame(data)
        
        # Pivot to have services as columns
        cost_pivot = df.pivot_table(
            index='date',
            columns='service',
            values='cost',
            fill_value=0
        )
        
        return cost_pivot
    
    def _fetch_recent_costs(self, days: int = 30) -> pd.DataFrame:
        """Fetch recent cost data"""
        return self._fetch_historical_costs()  # Reuse the method but with fewer days
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        features = data.copy()
        
        # Add time-based features
        features['day_of_week'] = features.index.dayofweek
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_month_end'] = (features.index.day >= 25).astype(int)
        
        # Add rolling statistics
        for col in data.columns:
            if col not in ['date']:
                features[f'{col}_rolling_7d_mean'] = data[col].rolling(7).mean()
                features[f'{col}_rolling_7d_std'] = data[col].rolling(7).std()
                features[f'{col}_rolling_30d_mean'] = data[col].rolling(30).mean()
                features[f'{col}_pct_change'] = data[col].pct_change()
        
        # Add total cost
        service_columns = [col for col in data.columns if col not in ['date']]
        features['total_cost'] = data[service_columns].sum(axis=1)
        
        # Add lagged features
        for lag in [1, 7, 30]:
            features[f'total_cost_lag_{lag}'] = features['total_cost'].shift(lag)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def _train_daily_predictor(self, features_df: pd.DataFrame) -> ModelPerformance:
        """Train model to predict daily costs"""
        # Prepare data
        target = features_df['total_cost']
        feature_cols = [col for col in features_df.columns if col not in ['total_cost']]
        X = features_df[feature_cols]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['daily'] = scaler
        
        # Train Random Forest model
        self.daily_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.daily_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.daily_predictor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            feature_cols,
            self.daily_predictor.feature_importances_
        ))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return ModelPerformance(
            model_type='RandomForest',
            last_trained=datetime.utcnow(),
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2_score=self.daily_predictor.score(X_test_scaled, y_test),
            training_samples=len(X_train),
            feature_importance=feature_importance
        )
    
    def _train_service_predictor(self, features_df: pd.DataFrame) -> ModelPerformance:
        """Train models for service-level predictions"""
        # For simplicity, using the same model architecture
        # In production, would train separate models per service
        
        service_cols = [col for col in features_df.columns 
                       if not any(x in col for x in ['rolling', 'lag', 'day_', 'month', 'quarter', 'is_', 'total_', 'pct_'])]
        
        if not service_cols:
            return ModelPerformance(
                model_type='ServicePredictor',
                last_trained=datetime.utcnow(),
                mse=0, rmse=0, mape=0, r2_score=0,
                training_samples=0,
                feature_importance={}
            )
        
        # Train a model for the top service
        top_service = max(service_cols, key=lambda x: features_df[x].sum())
        
        target = features_df[top_service]
        feature_cols = [col for col in features_df.columns if col != top_service and col not in service_cols]
        X = features_df[feature_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['service'] = scaler
        
        self.service_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        self.service_predictor.fit(X_train_scaled, y_train)
        
        y_pred = self.service_predictor.predict(X_test_scaled)
        
        return ModelPerformance(
            model_type='ServicePredictor',
            last_trained=datetime.utcnow(),
            mse=mean_squared_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mape=mean_absolute_percentage_error(y_test, y_pred),
            r2_score=self.service_predictor.score(X_test_scaled, y_test),
            training_samples=len(X_train),
            feature_importance={}
        )
    
    def _train_anomaly_detector(self, features_df: pd.DataFrame) -> ModelPerformance:
        """Train anomaly detection model"""
        # Use Isolation Forest for anomaly detection
        feature_cols = [col for col in features_df.columns if 'total_cost' not in col]
        X = features_df[feature_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['anomaly'] = scaler
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector.fit(X_scaled)
        
        # Evaluate
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomaly_predictions = self.anomaly_detector.predict(X_scaled)
        
        return ModelPerformance(
            model_type='IsolationForest',
            last_trained=datetime.utcnow(),
            mse=0,  # Not applicable for anomaly detection
            rmse=0,
            mape=0,
            r2_score=0,
            training_samples=len(X),
            feature_importance={'anomaly_contamination': 0.1}
        )
    
    def _predict_daily_costs(self, 
                           recent_features: pd.DataFrame,
                           include_breakdown: bool) -> List[CostPrediction]:
        """Generate daily cost predictions"""
        predictions = []
        last_date = recent_features.index[-1]
        
        # Prepare feature template
        last_row = recent_features.iloc[-1].copy()
        
        for day in range(1, self.forecast_days + 1):
            pred_date = last_date + timedelta(days=day)
            
            # Update time features
            last_row['day_of_week'] = pred_date.dayofweek
            last_row['day_of_month'] = pred_date.day
            last_row['month'] = pred_date.month
            last_row['quarter'] = (pred_date.month - 1) // 3 + 1
            last_row['is_weekend'] = int(pred_date.dayofweek in [5, 6])
            last_row['is_month_end'] = int(pred_date.day >= 25)
            
            # Prepare features
            feature_cols = [col for col in self.feature_columns if col != 'total_cost']
            X_pred = last_row[feature_cols].values.reshape(1, -1)
            X_pred_scaled = self.scalers['daily'].transform(X_pred)
            
            # Make prediction
            pred_cost = self.daily_predictor.predict(X_pred_scaled)[0]
            
            # Calculate confidence interval (using forest predictions)
            tree_predictions = [tree.predict(X_pred_scaled)[0] 
                              for tree in self.daily_predictor.estimators_]
            ci_lower = np.percentile(tree_predictions, 5)
            ci_upper = np.percentile(tree_predictions, 95)
            confidence = 1 - (ci_upper - ci_lower) / (pred_cost + 1e-6)
            
            # Service breakdown (simplified)
            service_breakdown = None
            if include_breakdown:
                service_breakdown = self._predict_service_breakdown(last_row, pred_cost)
            
            # Identify cost drivers
            drivers = self._identify_cost_drivers(last_row, pred_cost)
            
            predictions.append(CostPrediction(
                prediction_date=pred_date,
                prediction_type=PredictionType.DAILY,
                predicted_cost=pred_cost,
                confidence_interval=(ci_lower, ci_upper),
                confidence_score=confidence,
                service_breakdown=service_breakdown,
                drivers=drivers
            ))
            
            # Update lagged features for next prediction
            last_row['total_cost_lag_1'] = pred_cost
            if day >= 7:
                last_row['total_cost_lag_7'] = predictions[-7].predicted_cost
            if day >= 30:
                last_row['total_cost_lag_30'] = predictions[-30].predicted_cost
        
        return predictions
    
    def _predict_weekly_costs(self,
                            recent_features: pd.DataFrame,
                            include_breakdown: bool) -> List[CostPrediction]:
        """Generate weekly cost predictions"""
        daily_predictions = self._predict_daily_costs(recent_features, include_breakdown)
        
        weekly_predictions = []
        for week in range(self.forecast_days // 7):
            week_start = week * 7
            week_end = min((week + 1) * 7, len(daily_predictions))
            week_preds = daily_predictions[week_start:week_end]
            
            if week_preds:
                total_cost = sum(p.predicted_cost for p in week_preds)
                ci_lower = sum(p.confidence_interval[0] for p in week_preds)
                ci_upper = sum(p.confidence_interval[1] for p in week_preds)
                avg_confidence = np.mean([p.confidence_score for p in week_preds])
                
                # Aggregate service breakdowns
                service_breakdown = None
                if include_breakdown and week_preds[0].service_breakdown:
                    service_breakdown = {}
                    for pred in week_preds:
                        for service, cost in pred.service_breakdown.items():
                            service_breakdown[service] = service_breakdown.get(service, 0) + cost
                
                weekly_predictions.append(CostPrediction(
                    prediction_date=week_preds[-1].prediction_date,
                    prediction_type=PredictionType.WEEKLY,
                    predicted_cost=total_cost,
                    confidence_interval=(ci_lower, ci_upper),
                    confidence_score=avg_confidence,
                    service_breakdown=service_breakdown,
                    drivers=week_preds[0].drivers  # Use first day's drivers
                ))
        
        return weekly_predictions
    
    def _predict_monthly_costs(self,
                             recent_features: pd.DataFrame,
                             include_breakdown: bool) -> List[CostPrediction]:
        """Generate monthly cost prediction"""
        daily_predictions = self._predict_daily_costs(recent_features, include_breakdown)
        
        if daily_predictions:
            total_cost = sum(p.predicted_cost for p in daily_predictions)
            ci_lower = sum(p.confidence_interval[0] for p in daily_predictions)
            ci_upper = sum(p.confidence_interval[1] for p in daily_predictions)
            avg_confidence = np.mean([p.confidence_score for p in daily_predictions])
            
            # Aggregate service breakdowns
            service_breakdown = None
            if include_breakdown and daily_predictions[0].service_breakdown:
                service_breakdown = {}
                for pred in daily_predictions:
                    for service, cost in pred.service_breakdown.items():
                        service_breakdown[service] = service_breakdown.get(service, 0) + cost
            
            return [CostPrediction(
                prediction_date=daily_predictions[-1].prediction_date,
                prediction_type=PredictionType.MONTHLY,
                predicted_cost=total_cost,
                confidence_interval=(ci_lower, ci_upper),
                confidence_score=avg_confidence,
                service_breakdown=service_breakdown,
                drivers=['Monthly aggregate of daily predictions']
            )]
        
        return []
    
    def _predict_service_costs(self, recent_features: pd.DataFrame) -> List[CostPrediction]:
        """Generate service-level predictions"""
        # Simplified implementation
        # In production, would have separate models per service
        return self._predict_daily_costs(recent_features, include_breakdown=True)
    
    def _predict_service_breakdown(self, features: pd.Series, total_cost: float) -> Dict[str, float]:
        """Predict service-level cost breakdown"""
        # Simplified: use historical proportions
        # In production, would use service-specific models
        
        service_cols = [col for col in features.index 
                       if not any(x in col for x in ['rolling', 'lag', 'day_', 'month', 'quarter', 'is_', 'total_', 'pct_'])]
        
        if not service_cols:
            return {'Unknown': total_cost}
        
        # Use recent proportions
        service_costs = {}
        recent_total = sum(features[col] for col in service_cols if col in features)
        
        if recent_total > 0:
            for service in service_cols[:5]:  # Top 5 services
                if service in features:
                    proportion = features[service] / recent_total
                    service_costs[service] = total_cost * proportion
        else:
            service_costs['Unknown'] = total_cost
        
        return service_costs
    
    def _identify_cost_drivers(self, features: pd.Series, predicted_cost: float) -> List[str]:
        """Identify main drivers of the prediction"""
        drivers = []
        
        # Check for weekend/weekday pattern
        if features['is_weekend'] == 1:
            drivers.append("Weekend pattern detected")
        
        # Check for month-end
        if features['is_month_end'] == 1:
            drivers.append("Month-end processing expected")
        
        # Check for trend
        if 'total_cost_lag_7' in features and features['total_cost_lag_7'] > 0:
            weekly_change = (predicted_cost - features['total_cost_lag_7']) / features['total_cost_lag_7']
            if weekly_change > 0.1:
                drivers.append(f"Upward trend: +{weekly_change:.1%} vs last week")
            elif weekly_change < -0.1:
                drivers.append(f"Downward trend: {weekly_change:.1%} vs last week")
        
        if not drivers:
            drivers.append("Normal expected variation")
        
        return drivers
    
    def _create_spike_anomaly(self, prediction: CostPrediction) -> PredictedAnomaly:
        """Create anomaly for cost spike"""
        baseline = prediction.confidence_interval[0]
        spike_amount = prediction.predicted_cost - baseline
        spike_pct = spike_amount / baseline if baseline > 0 else 0
        
        # Determine priority based on spike size
        if spike_pct > 0.5:
            priority = 'critical'
        elif spike_pct > 0.3:
            priority = 'high'
        elif spike_pct > 0.2:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Service-specific description
        top_service = 'Unknown'
        if prediction.service_breakdown:
            top_service = max(prediction.service_breakdown.items(), key=lambda x: x[1])[0]
        
        return PredictedAnomaly(
            anomaly_date=prediction.prediction_date,
            anomaly_type=AnomalyType.SPIKE,
            service=top_service,
            predicted_impact=spike_amount,
            probability=0.9,
            description=f"Predicted {spike_pct:.0%} cost spike on {prediction.prediction_date.strftime('%Y-%m-%d')}",
            recommended_actions=[
                f"Review {top_service} scaling policies",
                "Check for scheduled batch jobs",
                "Verify no unintended resource provisioning",
                "Set up cost alerts for this date"
            ],
            alert_priority=priority
        )
    
    def _detect_service_anomalies(self, prediction: CostPrediction) -> List[PredictedAnomaly]:
        """Detect anomalies at service level"""
        anomalies = []
        
        if not prediction.service_breakdown:
            return anomalies
        
        # Check each service for unusual patterns
        for service, cost in prediction.service_breakdown.items():
            # Simple threshold-based detection
            # In production, would use service-specific models
            if cost > 1000:  # Arbitrary threshold
                anomalies.append(PredictedAnomaly(
                    anomaly_date=prediction.prediction_date,
                    anomaly_type=AnomalyType.SERVICE_SURGE,
                    service=service,
                    predicted_impact=cost,
                    probability=0.7,
                    description=f"{service} showing unusual activity",
                    recommended_actions=[
                        f"Review {service} resource usage",
                        f"Check {service} auto-scaling settings",
                        f"Verify {service} configurations"
                    ],
                    alert_priority='medium'
                ))
        
        return anomalies
    
    def _detect_gradual_increases(self, predictions: List[CostPrediction]) -> List[PredictedAnomaly]:
        """Detect gradual cost increases over time"""
        anomalies = []
        
        if len(predictions) < 7:
            return anomalies
        
        # Calculate moving averages
        costs = [p.predicted_cost for p in predictions]
        week1_avg = np.mean(costs[:7])
        week2_avg = np.mean(costs[7:14]) if len(costs) >= 14 else None
        week3_avg = np.mean(costs[14:21]) if len(costs) >= 21 else None
        week4_avg = np.mean(costs[21:28]) if len(costs) >= 28 else None
        
        # Check for consistent increases
        if week2_avg and week2_avg > week1_avg * 1.1:
            if week3_avg and week3_avg > week2_avg * 1.05:
                trend_pct = (week3_avg - week1_avg) / week1_avg
                
                anomalies.append(PredictedAnomaly(
                    anomaly_date=predictions[20].prediction_date if len(predictions) > 20 else predictions[-1].prediction_date,
                    anomaly_type=AnomalyType.GRADUAL_INCREASE,
                    service='All Services',
                    predicted_impact=(week3_avg - week1_avg) * 7,  # Weekly impact
                    probability=0.8,
                    description=f"Costs trending up {trend_pct:.0%} over 3 weeks",
                    recommended_actions=[
                        "Review recent infrastructure changes",
                        "Check for resource leaks",
                        "Analyze service growth patterns",
                        "Consider implementing cost controls"
                    ],
                    alert_priority='high'
                ))
        
        return anomalies
    
    def _save_models(self, bucket: str):
        """Save trained models to S3"""
        models = {
            'daily_predictor': self.daily_predictor,
            'service_predictor': self.service_predictor,
            'anomaly_detector': self.anomaly_detector,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'model_performance': self.model_performance
        }
        
        # Serialize models
        model_data = pickle.dumps(models)
        
        # Save to S3
        key = f"cost-optimizer/models/cost_predictor_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=model_data
        )
        
        logger.info(f"Models saved to s3://{bucket}/{key}")
    
    def load_models(self, bucket: str, model_key: str):
        """Load trained models from S3"""
        response = self.s3_client.get_object(Bucket=bucket, Key=model_key)
        model_data = response['Body'].read()
        
        models = pickle.loads(model_data)
        
        self.daily_predictor = models['daily_predictor']
        self.service_predictor = models['service_predictor']
        self.anomaly_detector = models['anomaly_detector']
        self.scalers = models['scalers']
        self.feature_columns = models['feature_columns']
        self.model_performance = models['model_performance']
        
        logger.info(f"Models loaded from s3://{bucket}/{model_key}")
    
    def generate_prediction_report(self,
                                 predictions: List[CostPrediction],
                                 anomalies: List[PredictedAnomaly],
                                 output_file: str = 'cost_predictions.json'):
        """Generate comprehensive prediction report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'prediction_summary': {
                'forecast_days': self.forecast_days,
                'total_predicted_cost': sum(p.predicted_cost for p in predictions),
                'average_daily_cost': np.mean([p.predicted_cost for p in predictions if p.prediction_type == PredictionType.DAILY]),
                'confidence_score': np.mean([p.confidence_score for p in predictions])
            },
            'predictions': [
                {
                    'date': p.prediction_date.isoformat(),
                    'type': p.prediction_type.value,
                    'predicted_cost': p.predicted_cost,
                    'confidence_interval': p.confidence_interval,
                    'confidence_score': p.confidence_score,
                    'service_breakdown': p.service_breakdown,
                    'drivers': p.drivers
                }
                for p in predictions
            ],
            'anomalies': [
                {
                    'date': a.anomaly_date.isoformat(),
                    'type': a.anomaly_type.value,
                    'service': a.service,
                    'predicted_impact': a.predicted_impact,
                    'probability': a.probability,
                    'description': a.description,
                    'recommended_actions': a.recommended_actions,
                    'priority': a.alert_priority
                }
                for a in anomalies
            ],
            'model_performance': {
                model_name: {
                    'mape': perf.mape,
                    'rmse': perf.rmse,
                    'r2_score': perf.r2_score,
                    'last_trained': perf.last_trained.isoformat()
                }
                for model_name, perf in self.model_performance.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Prediction report saved to {output_file}")
        
        return report