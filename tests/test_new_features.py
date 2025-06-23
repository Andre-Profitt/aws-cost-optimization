"""
Integration tests for new AWS Cost Optimizer features
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

from aws_cost_optimizer.analysis.periodic_detector import (
    PeriodicResourceDetector, PeriodType, PeriodicPattern, ResourcePeriodicity
)
from aws_cost_optimizer.ml.cost_predictor import (
    CostPredictor, PredictionType, AnomalyType
)
from aws_cost_optimizer.realtime.cost_controller import (
    RealtimeCostController, CostThreshold, ThresholdType, ControlAction
)
from aws_cost_optimizer.tagging.intelligent_tagger import (
    IntelligentTagger, TagCategory
)
from aws_cost_optimizer.tracking.savings_tracker import (
    SavingsTracker, OptimizationRecord, OptimizationType, SavingsStatus
)


class TestPeriodicDetector(unittest.TestCase):
    """Test periodic resource detection"""
    
    def setUp(self):
        self.detector = PeriodicResourceDetector(lookback_days=90)
        
    def test_detect_monthly_pattern(self):
        """Test detection of monthly patterns"""
        # Create mock time series with monthly peaks
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        values = np.zeros(90)
        
        # Add monthly peaks on day 28
        for i in range(3):
            peak_day = 28 + (i * 30)
            if peak_day < 90:
                values[peak_day-3:peak_day+3] = 100  # 6-day peak
        
        time_series = pd.Series(values, index=dates)
        
        # Mock CloudWatch data
        with patch.object(self.detector, '_fetch_metric_data', return_value=time_series):
            result = self.detector.analyze_resource('i-12345', 'ec2')
        
        # Verify periodic pattern detected
        self.assertEqual(result.usage_classification, 'periodic')
        self.assertTrue(any(p.period_type == PeriodType.MONTHLY for p in result.patterns))
        self.assertGreater(result.risk_score, 0.7)
        
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Mock empty time series
        empty_series = pd.Series()
        
        with patch.object(self.detector, '_fetch_metric_data', return_value=empty_series):
            result = self.detector.analyze_resource('i-12345', 'ec2')
        
        self.assertEqual(result.usage_classification, 'unknown')
        self.assertEqual(len(result.patterns), 0)


class TestCostPredictor(unittest.TestCase):
    """Test ML-based cost prediction"""
    
    def setUp(self):
        self.predictor = CostPredictor(lookback_days=90, forecast_days=30)
        
    def test_model_training(self):
        """Test model training process"""
        # Create mock historical data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        costs = pd.DataFrame({
            'Amazon EC2': np.random.normal(1000, 100, 90),
            'Amazon RDS': np.random.normal(500, 50, 90),
            'Amazon S3': np.random.normal(200, 20, 90)
        }, index=dates)
        
        with patch.object(self.predictor, '_fetch_historical_costs', return_value=costs):
            performance = self.predictor.train_models()
        
        # Verify models trained
        self.assertIsNotNone(self.predictor.daily_predictor)
        self.assertIsNotNone(self.predictor.anomaly_detector)
        self.assertIn('daily', performance)
        self.assertGreater(performance['daily'].training_samples, 0)
        
    def test_cost_prediction(self):
        """Test cost prediction generation"""
        # Train mock models first
        self.predictor.daily_predictor = MagicMock()
        self.predictor.daily_predictor.predict.return_value = [1500]
        self.predictor.daily_predictor.estimators_ = [MagicMock() for _ in range(10)]
        for estimator in self.predictor.daily_predictor.estimators_:
            estimator.predict.return_value = [1500 + np.random.normal(0, 50)]
        
        self.predictor.scalers = {'daily': MagicMock()}
        self.predictor.feature_columns = ['col1', 'col2', 'total_cost']
        
        # Mock recent data
        recent_data = pd.DataFrame({
            'col1': [1], 'col2': [2], 'total_cost': [1000]
        }, index=[datetime.now()])
        
        with patch.object(self.predictor, '_fetch_recent_costs', return_value=recent_data):
            with patch.object(self.predictor, '_engineer_features', return_value=recent_data):
                predictions = self.predictor.predict_costs(PredictionType.DAILY)
        
        # Verify predictions
        self.assertEqual(len(predictions), 30)  # 30 days forecast
        self.assertTrue(all(p.predicted_cost > 0 for p in predictions))
        self.assertTrue(all(p.confidence_score > 0 for p in predictions))
        
    def test_anomaly_detection(self):
        """Test anomaly detection in predictions"""
        # Setup predictor with mock predictions that include anomaly
        self.predictor.anomaly_detector = MagicMock()
        self.predictor.daily_predictor = MagicMock()
        self.predictor.scalers = {'daily': MagicMock()}
        self.predictor.feature_columns = ['col1']
        
        # Mock predict_costs to return spike
        normal_cost = 1000
        spike_cost = 2000  # 100% spike
        
        mock_predictions = []
        for i in range(30):
            if i == 15:  # Spike on day 15
                cost = spike_cost
            else:
                cost = normal_cost
            
            mock_prediction = MagicMock()
            mock_prediction.predicted_cost = cost
            mock_prediction.confidence_interval = (normal_cost * 0.9, normal_cost * 1.1)
            mock_prediction.prediction_date = datetime.now() + timedelta(days=i)
            mock_prediction.service_breakdown = {'EC2': cost}
            mock_predictions.append(mock_prediction)
        
        with patch.object(self.predictor, 'predict_costs', return_value=mock_predictions):
            anomalies = self.predictor.detect_future_anomalies()
        
        # Verify anomaly detected
        self.assertGreater(len(anomalies), 0)
        spike_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.SPIKE]
        self.assertGreater(len(spike_anomalies), 0)


class TestRealtimeController(unittest.TestCase):
    """Test real-time cost controls"""
    
    def setUp(self):
        self.thresholds = [
            CostThreshold(
                threshold_id='test_daily',
                threshold_type=ThresholdType.DAILY,
                value=1000,
                action=ControlAction.ALERT
            ),
            CostThreshold(
                threshold_id='test_ec2',
                threshold_type=ThresholdType.SERVICE,
                target='Amazon EC2',
                value=500,
                action=ControlAction.THROTTLE
            )
        ]
        self.controller = RealtimeCostController(thresholds=self.thresholds)
        
    def test_threshold_checking(self):
        """Test cost threshold checking"""
        current_costs = {
            'Amazon EC2': 600,  # Exceeds threshold
            'Amazon RDS': 300,
            'Amazon S3': 200
        }
        
        events = self.controller.check_thresholds(current_costs)
        
        # Verify threshold breaches detected
        self.assertEqual(len(events), 2)  # Daily total and EC2 service
        
        # Check daily threshold breach
        daily_breach = next(e for e in events if e.threshold_breached.threshold_id == 'test_daily')
        self.assertEqual(daily_breach.current_cost, 1100)  # 600 + 300 + 200
        
        # Check EC2 threshold breach
        ec2_breach = next(e for e in events if e.threshold_breached.threshold_id == 'test_ec2')
        self.assertEqual(ec2_breach.current_cost, 600)
        
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        service = 'Amazon EC2'
        
        # Normal spend - breaker should not trip
        breaker = self.controller.update_circuit_breaker(service, 5000)
        self.assertIsNone(breaker)
        
        # Exceed threshold - breaker should trip
        breaker = self.controller.update_circuit_breaker(service, 15000)
        self.assertIsNotNone(breaker)
        self.assertTrue(breaker.is_open)
        self.assertEqual(breaker.service, service)
        
    def test_eventbridge_rule_creation(self):
        """Test EventBridge rule setup"""
        with patch.object(self.controller.eventbridge, 'put_rule', return_value={'RuleArn': 'test-arn'}):
            with patch.object(self.controller.eventbridge, 'put_targets'):
                rules = self.controller.setup_eventbridge_rules()
        
        self.assertEqual(len(rules), 5)  # 5 default rules


class TestIntelligentTagger(unittest.TestCase):
    """Test intelligent tagging system"""
    
    def setUp(self):
        self.tagger = IntelligentTagger(
            required_tags=['Environment', 'Owner', 'CostCenter']
        )
        
    def test_name_based_suggestions(self):
        """Test tag suggestions from resource names"""
        suggestions = self.tagger.analyze_resource(
            resource_id='i-12345',
            resource_type='ec2',
            resource_name='webapp-prod-api-01',
            current_tags={}
        )
        
        # Should suggest Environment=production
        env_suggestions = [s for s in suggestions if s.key == 'Environment']
        self.assertGreater(len(env_suggestions), 0)
        self.assertEqual(env_suggestions[0].value, 'production')
        self.assertGreater(env_suggestions[0].confidence, 0.8)
        
    def test_compliance_checking(self):
        """Test tag compliance validation"""
        result = self.tagger.check_compliance(
            resource_id='i-12345',
            resource_type='ec2',
            current_tags={'Environment': 'prod'}
        )
        
        # Should be non-compliant (missing Owner and CostCenter)
        self.assertFalse(result.compliant)
        self.assertEqual(len(result.missing_required), 2)
        self.assertIn('Owner', result.missing_required)
        self.assertIn('CostCenter', result.missing_required)
        
    def test_ml_training(self):
        """Test ML model training"""
        training_data = [
            {
                'name': 'api-prod-01',
                'tags': {'Environment': 'production', 'Application': 'api'}
            },
            {
                'name': 'web-dev-02',
                'tags': {'Environment': 'development', 'Application': 'web'}
            },
            {
                'name': 'db-prod-03',
                'tags': {'Environment': 'production', 'Application': 'database'}
            }
        ]
        
        self.tagger.train_ml_models(training_data)
        
        # Verify models trained
        self.assertIsNotNone(self.tagger.tag_classifier)
        self.assertIsNotNone(self.tagger.name_vectorizer)


class TestSavingsTracker(unittest.TestCase):
    """Test savings tracking functionality"""
    
    def setUp(self):
        self.tracker = SavingsTracker()
        
    def test_record_optimization(self):
        """Test recording optimization actions"""
        optimization = OptimizationRecord(
            optimization_id='opt-001',
            resource_id='i-12345',
            resource_type='ec2',
            optimization_type=OptimizationType.RIGHTSIZING,
            description='Downsize t3.xlarge to t3.large',
            implemented_date=datetime.now(),
            projected_monthly_savings=150.0
        )
        
        opt_id = self.tracker.record_optimization(optimization)
        
        self.assertEqual(opt_id, 'opt-001')
        self.assertIn(opt_id, self.tracker.optimization_cache)
        
    def test_update_actual_savings(self):
        """Test updating actual savings"""
        # First record an optimization
        optimization = OptimizationRecord(
            optimization_id='opt-002',
            resource_id='i-67890',
            resource_type='ec2',
            optimization_type=OptimizationType.TERMINATION,
            description='Terminate idle instance',
            implemented_date=datetime.now(),
            projected_monthly_savings=200.0
        )
        self.tracker.record_optimization(optimization)
        
        # Update with actual savings
        updated = self.tracker.update_actual_savings('opt-002', 180.0)
        
        self.assertEqual(updated.actual_monthly_savings, 180.0)
        self.assertEqual(updated.status, SavingsStatus.REALIZED)
        
    def test_savings_metrics_calculation(self):
        """Test savings metrics calculation"""
        # Add test optimizations
        now = datetime.now()
        optimizations = [
            OptimizationRecord(
                optimization_id=f'opt-{i}',
                resource_id=f'i-{i}',
                resource_type='ec2',
                optimization_type=OptimizationType.RIGHTSIZING,
                description=f'Optimization {i}',
                implemented_date=now - timedelta(days=i),
                projected_monthly_savings=100.0 * i,
                actual_monthly_savings=90.0 * i if i % 2 == 0 else None,
                status=SavingsStatus.REALIZED if i % 2 == 0 else SavingsStatus.PROJECTED
            )
            for i in range(1, 5)
        ]
        
        for opt in optimizations:
            self.tracker.record_optimization(opt)
        
        # Calculate metrics
        summary = self.tracker.calculate_savings_metrics(
            now - timedelta(days=30),
            now
        )
        
        self.assertGreater(summary.total_projected_savings, 0)
        self.assertGreater(summary.total_realized_savings, 0)
        self.assertGreater(len(summary.by_type), 0)
        
    def test_projection_vs_actual_comparison(self):
        """Test comparison of projected vs actual savings"""
        # Add optimizations with both projected and actual
        now = datetime.now()
        optimizations = [
            OptimizationRecord(
                optimization_id='opt-comp-1',
                resource_id='i-comp-1',
                resource_type='ec2',
                optimization_type=OptimizationType.RIGHTSIZING,
                description='Rightsize instance',
                implemented_date=now - timedelta(days=15),
                projected_monthly_savings=100.0,
                actual_monthly_savings=110.0,  # 10% over
                status=SavingsStatus.REALIZED
            ),
            OptimizationRecord(
                optimization_id='opt-comp-2',
                resource_id='i-comp-2',
                resource_type='rds',
                optimization_type=OptimizationType.RIGHTSIZING,
                description='Rightsize database',
                implemented_date=now - timedelta(days=10),
                projected_monthly_savings=200.0,
                actual_monthly_savings=150.0,  # 25% under
                status=SavingsStatus.PARTIAL
            )
        ]
        
        for opt in optimizations:
            self.tracker.record_optimization(opt)
        
        # Compare
        comparison = self.tracker.compare_projected_vs_actual(
            optimization_ids=['opt-comp-1', 'opt-comp-2']
        )
        
        self.assertEqual(comparison['optimization_count'], 2)
        self.assertEqual(comparison['total_projected'], 300.0)
        self.assertEqual(comparison['total_actual'], 260.0)
        self.assertAlmostEqual(comparison['overall_accuracy'], 260.0/300.0, places=2)


class TestCLICommands(unittest.TestCase):
    """Test new CLI commands"""
    
    def test_cli_imports(self):
        """Test that CLI commands can import new modules"""
        from aws_cost_optimizer.cli import cli
        
        # Get all commands
        commands = cli.commands
        
        # Verify new commands exist
        self.assertIn('detect-periodic-resources', commands)
        self.assertIn('predict-costs', commands)
        self.assertIn('setup-realtime-controls', commands)
        self.assertIn('intelligent-tagging', commands)
        self.assertIn('track-savings', commands)
        self.assertIn('version', commands)


class TestIntegration(unittest.TestCase):
    """Integration tests across modules"""
    
    def test_periodic_detection_to_tagging(self):
        """Test integration between periodic detection and tagging"""
        # Detect periodic resource
        detector = PeriodicResourceDetector()
        tagger = IntelligentTagger()
        
        # Mock periodic detection result
        periodic_result = ResourcePeriodicity(
            resource_id='i-12345',
            resource_type='ec2',
            patterns=[
                PeriodicPattern(
                    period_type=PeriodType.MONTHLY,
                    period_days=30,
                    confidence=0.9,
                    peak_times=[],
                    description='Monthly batch job'
                )
            ],
            usage_classification='periodic',
            recommendation='Do not terminate',
            risk_score=0.9,
            analysis_period_days=365,
            last_analyzed=datetime.now()
        )
        
        # Tag suggestion should include periodic indicator
        suggestions = tagger.analyze_resource(
            resource_id=periodic_result.resource_id,
            resource_type=periodic_result.resource_type,
            usage_metrics={
                'periodic_pattern': periodic_result.patterns[0].period_type.value,
                'risk_score': periodic_result.risk_score
            }
        )
        
        # Should suggest workload type tag
        workload_suggestions = [s for s in suggestions if s.key == 'WorkloadType']
        self.assertGreater(len(workload_suggestions), 0)
        
    def test_prediction_to_realtime_control(self):
        """Test integration between prediction and real-time controls"""
        predictor = CostPredictor()
        
        # Mock anomaly prediction
        from aws_cost_optimizer.ml.cost_predictor import PredictedAnomaly
        anomaly = PredictedAnomaly(
            anomaly_date=datetime.now() + timedelta(days=3),
            anomaly_type=AnomalyType.SPIKE,
            service='Amazon EC2',
            predicted_impact=5000,
            probability=0.9,
            description='Predicted 200% spike',
            recommended_actions=['Review scaling policies'],
            alert_priority='critical'
        )
        
        # Create threshold based on prediction
        threshold = CostThreshold(
            threshold_id=f'predicted_{anomaly.anomaly_date.strftime("%Y%m%d")}',
            threshold_type=ThresholdType.SERVICE,
            target=anomaly.service,
            value=anomaly.predicted_impact * 0.8,  # Set threshold at 80% of predicted spike
            action=ControlAction.ALERT
        )
        
        controller = RealtimeCostController(thresholds=[threshold])
        
        # Verify threshold created
        self.assertEqual(len(controller.thresholds), 1)
        self.assertEqual(
            controller.thresholds[threshold.threshold_id].value,
            anomaly.predicted_impact * 0.8
        )


if __name__ == '__main__':
    unittest.main()