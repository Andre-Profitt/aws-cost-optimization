"""
Tests for EC2 Optimizer
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import boto3
from moto import mock_ec2, mock_cloudwatch

from aws_cost_optimizer.optimization.ec2_optimizer import (
    EC2Optimizer, EC2OptimizationRecommendation
)

@pytest.fixture
def ec2_optimizer():
    """Create EC2 optimizer instance"""
    return EC2Optimizer(
        cpu_threshold=10.0,
        memory_threshold=20.0,
        network_threshold=5.0,
        observation_days=14
    )

@pytest.fixture
def sample_instance():
    """Sample EC2 instance data"""
    return {
        'InstanceId': 'i-1234567890abcdef0',
        'InstanceType': 't3.large',
        'State': {'Name': 'running'},
        'LaunchTime': datetime.utcnow() - timedelta(days=30),
        'Tags': [
            {'Key': 'Name', 'Value': 'test-instance'},
            {'Key': 'Environment', 'Value': 'development'}
        ]
    }

@mock_ec2
@mock_cloudwatch
def test_analyze_idle_instance(ec2_optimizer, sample_instance):
    """Test detection of idle instances"""
    # Set up mock EC2
    ec2 = boto3.client('ec2', region_name='us-east-1')
    
    # Create mock instance
    ec2.run_instances(
        ImageId='ami-12345678',
        MinCount=1,
        MaxCount=1,
        InstanceType='t3.large',
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [
                {'Key': 'Name', 'Value': 'test-instance'},
                {'Key': 'Environment', 'Value': 'development'}
            ]
        }]
    )
    
    # Mock CloudWatch metrics
    with patch.object(ec2_optimizer, '_get_instance_metrics') as mock_metrics:
        mock_metrics.return_value = {
            'cpu_avg': 5.0,  # Below threshold
            'cpu_max': 8.0,
            'cpu_p95': 7.0,
            'network_in_total_mb': 10.0,
            'network_in_daily_avg_mb': 0.7,  # Below threshold
            'network_out_total_mb': 15.0,
            'network_out_daily_avg_mb': 1.0
        }
        
        # Mock pattern detector
        with patch.object(ec2_optimizer.pattern_detector, 'analyze_ec2_patterns') as mock_patterns:
            mock_patterns.return_value = {
                'patterns': {'cpu_pattern': 'steady'}  # Not periodic
            }
            
            # Mock safety checker
            with patch.object(ec2_optimizer.safety_checker, 'check_instance_safety') as mock_safety:
                mock_safety.return_value = {
                    'safe_to_modify': True,
                    'warnings': [],
                    'blockers': []
                }
                
                # Run analysis
                recommendations = ec2_optimizer._analyze_instance(sample_instance, 'us-east-1')
                
                # Verify recommendations
                assert len(recommendations) > 0
                
                idle_rec = next((r for r in recommendations if r.action == 'stop'), None)
                assert idle_rec is not None
                assert idle_rec.reason.startswith("Instance is idle")
                assert idle_rec.monthly_savings > 0
                assert idle_rec.risk_level == 'low'

def test_check_rightsizing(ec2_optimizer, sample_instance):
    """Test rightsizing recommendations"""
    metrics = {
        'cpu_avg': 15.0,  # Low but not idle
        'cpu_max': 45.0,
        'cpu_p95': 35.0,  # Below 60% threshold
        'network_in_daily_avg_mb': 100.0,
        'network_out_daily_avg_mb': 150.0
    }
    
    with patch.object(ec2_optimizer, '_get_instance_metrics', return_value=metrics):
        with patch.object(ec2_optimizer, '_get_recommended_instance_type', return_value='t3.medium'):
            rec = ec2_optimizer._check_rightsizing(
                sample_instance,
                metrics,
                {'Environment': 'development'},
                'us-east-1'
            )
            
            assert rec is not None
            assert rec.action == 'rightsize'
            assert rec.instance_type == 't3.large'
            assert 't3.medium' in rec.implementation_steps[2]  # New type in steps

def test_calculate_instance_cost(ec2_optimizer):
    """Test instance cost calculation"""
    cost = ec2_optimizer._calculate_instance_cost('t3.large')
    expected = 0.0832 * 24 * 30  # $0.0832/hour * 24 hours * 30 days
    assert abs(cost - expected) < 0.01

def test_get_recommended_instance_type(ec2_optimizer):
    """Test instance type recommendation logic"""
    # Test downsizing from t3.large
    recommended = ec2_optimizer._get_recommended_instance_type(
        't3.large',
        cpu_avg=8.0,
        cpu_p95=15.0
    )
    assert recommended == 't3.small'  # Down 2 sizes
    
    # Test minimal downsizing
    recommended = ec2_optimizer._get_recommended_instance_type(
        't3.xlarge',
        cpu_avg=18.0,
        cpu_p95=35.0
    )
    assert recommended == 't3.large'  # Down 1 size
    
    # Test no downsizing needed
    recommended = ec2_optimizer._get_recommended_instance_type(
        't3.medium',
        cpu_avg=45.0,
        cpu_p95=70.0
    )
    assert recommended is None

@mock_ec2
def test_analyze_ebs_volumes(ec2_optimizer):
    """Test EBS volume optimization"""
    # Create mock volumes
    ec2 = boto3.client('ec2', region_name='us-east-1')
    
    # Create unattached volume
    response = ec2.create_volume(
        AvailabilityZone='us-east-1a',
        Size=100,
        VolumeType='gp2'
    )
    volume_id = response['VolumeId']
    
    # Mock volume age
    with patch.object(ec2_optimizer, '_is_old_volume', return_value=True):
        with patch.object(ec2_optimizer, '_has_recent_snapshot', return_value=False):
            recommendations = ec2_optimizer._analyze_ebs_volumes('us-east-1')
            
            assert len(recommendations) > 0
            
            # Find our volume
            vol_rec = next((r for r in recommendations if r.volume_id == volume_id), None)
            assert vol_rec is not None
            assert vol_rec.action == 'delete'
            assert vol_rec.monthly_savings == 10.0  # 100GB * $0.10

def test_safety_integration(ec2_optimizer, sample_instance):
    """Test safety checks integration"""
    # Test with blocked instance
    with patch.object(ec2_optimizer.safety_checker, 'check_instance_safety') as mock_safety:
        mock_safety.return_value = {
            'safe_to_modify': False,
            'warnings': [],
            'blockers': ['Instance is in production']
        }
        
        recommendations = ec2_optimizer._analyze_instance(sample_instance, 'us-east-1')
        
        # Should not generate recommendations for unsafe instances
        assert len(recommendations) == 0

def test_pattern_detection_integration(ec2_optimizer, sample_instance):
    """Test pattern detection prevents stopping periodic workloads"""
    metrics = {
        'cpu_avg': 5.0,  # Idle on average
        'cpu_max': 80.0,  # But has spikes
        'cpu_p95': 10.0,
        'network_in_daily_avg_mb': 2.0,
        'network_out_daily_avg_mb': 3.0
    }
    
    with patch.object(ec2_optimizer, '_get_instance_metrics', return_value=metrics):
        with patch.object(ec2_optimizer.pattern_detector, 'analyze_ec2_patterns') as mock_patterns:
            mock_patterns.return_value = {
                'patterns': {'cpu_pattern': 'periodic'}  # Periodic workload
            }
            
            with patch.object(ec2_optimizer.safety_checker, 'check_instance_safety') as mock_safety:
                mock_safety.return_value = {'safe_to_modify': True}
                
                recommendations = ec2_optimizer._analyze_instance(sample_instance, 'us-east-1')
                
                # Should not recommend stopping periodic workloads
                idle_rec = next((r for r in recommendations if r.action == 'stop'), None)
                assert idle_rec is None