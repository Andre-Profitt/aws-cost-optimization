"""
Utility Helper Functions - Common operations and utilities for AWS cost optimization
"""
import boto3
import logging
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import yaml
import os
import re
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)

class AWSHelper:
    """Helper class for common AWS operations"""
    
    def __init__(self, profile: str = None):
        """Initialize AWS helper with optional profile"""
        self.profile = profile
        if profile:
            self.session = boto3.Session(profile_name=profile)
        else:
            self.session = boto3.Session()
    
    def get_client(self, service: str, region: str = None):
        """Get boto3 client for a service"""
        return self.session.client(service, region_name=region)
    
    def get_all_regions(self, service: str = 'ec2') -> List[str]:
        """Get list of all available AWS regions"""
        try:
            client = self.get_client(service)
            response = client.describe_regions()
            return [region['RegionName'] for region in response['Regions']]
        except Exception as e:
            logger.error(f"Error getting regions: {e}")
            return ['us-east-1', 'us-west-2']  # Fallback
    
    def get_account_id(self) -> str:
        """Get current AWS account ID"""
        try:
            sts = self.get_client('sts')
            response = sts.get_caller_identity()
            return response['Account']
        except Exception as e:
            logger.error(f"Error getting account ID: {e}")
            return 'unknown'
    
    def test_credentials(self) -> bool:
        """Test if AWS credentials are valid"""
        try:
            sts = self.get_client('sts')
            sts.get_caller_identity()
            return True
        except Exception as e:
            logger.error(f"AWS credentials test failed: {e}")
            return False

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = 'config/config.yaml'):
        """Save configuration to YAML file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'global': {
                'regions': ['us-east-1'],
                'services': ['ec2', 's3', 'rds', 'ebs'],
                'lookback_days': 14,
                'enable_patterns': True,
                'enable_safety_checks': True,
                'enable_anomaly_detection': True
            },
            'ec2': {
                'cpu_threshold': 10.0,
                'memory_threshold': 20.0,
                'network_threshold': 10.0,
                'uptime_threshold': 5.0
            },
            's3': {
                'ia_threshold_days': 30,
                'glacier_threshold_days': 90,
                'deep_archive_threshold_days': 180,
                'min_savings_threshold': 10.0
            },
            'rds': {
                'cpu_threshold': 20.0,
                'memory_threshold': 30.0,
                'connections_threshold': 10.0,
                'min_savings_threshold': 10.0
            },
            'ebs': {
                'utilization_threshold': 20.0,
                'iops_threshold': 100.0,
                'unattached_age_days': 7,
                'snapshot_age_days': 30
            },
            'ri': {
                'lookback_days': 90,
                'min_utilization_threshold': 75.0,
                'min_savings_threshold': 100.0,
                'confidence_threshold': 0.7
            },
            'safety': {
                'strict_mode': True,
                'require_backups': True,
                'business_hours_only': False,
                'max_cost_impact': 10000.0,
                'protected_environments': ['prod', 'production']
            },
            'patterns': {
                'analysis_period_days': 90,
                'min_data_points': 100,
                'seasonality_threshold': 0.3,
                'predictability_threshold': 0.7
            },
            'anomalies': {
                'lookback_days': 90,
                'sensitivity': 'medium',
                'min_cost_threshold': 10.0
            }
        }

class CostCalculator:
    """Utilities for cost calculations"""
    
    # Simplified pricing data (in production, use AWS Pricing API)
    EC2_PRICING = {
        'us-east-1': {
            't3.nano': 0.0052, 't3.micro': 0.0104, 't3.small': 0.0208,
            't3.medium': 0.0416, 't3.large': 0.0832, 't3.xlarge': 0.1664,
            'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
            'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
            'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504
        }
    }
    
    S3_PRICING = {
        'standard': 0.023,
        'standard_ia': 0.0125,
        'glacier': 0.004,
        'deep_archive': 0.00099,
        'intelligent_tiering': 0.0125
    }
    
    EBS_PRICING = {
        'gp2': 0.10, 'gp3': 0.08, 'io1': 0.125, 'io2': 0.125,
        'st1': 0.045, 'sc1': 0.025, 'standard': 0.05
    }
    
    @classmethod
    def calculate_ec2_monthly_cost(cls, instance_type: str, region: str = 'us-east-1') -> float:
        """Calculate monthly cost for EC2 instance"""
        hourly_rate = cls.EC2_PRICING.get(region, cls.EC2_PRICING['us-east-1']).get(instance_type, 0.10)
        return hourly_rate * 24 * 30
    
    @classmethod
    def calculate_s3_monthly_cost(cls, size_gb: float, storage_class: str = 'standard') -> float:
        """Calculate monthly cost for S3 storage"""
        price_per_gb = cls.S3_PRICING.get(storage_class.lower(), 0.023)
        return size_gb * price_per_gb
    
    @classmethod
    def calculate_ebs_monthly_cost(cls, size_gb: int, volume_type: str = 'gp2', iops: int = 0) -> float:
        """Calculate monthly cost for EBS volume"""
        storage_cost = size_gb * cls.EBS_PRICING.get(volume_type, 0.10)
        
        # Add IOPS cost for provisioned volumes
        iops_cost = 0
        if volume_type in ['io1', 'io2'] and iops > 0:
            iops_cost = iops * 0.065  # $0.065 per IOPS per month
        elif volume_type == 'gp3' and iops > 3000:
            extra_iops = iops - 3000
            iops_cost = extra_iops * 0.005
        
        return storage_cost + iops_cost
    
    @classmethod
    def calculate_savings_percentage(cls, current_cost: float, new_cost: float) -> float:
        """Calculate savings percentage"""
        if current_cost <= 0:
            return 0.0
        return ((current_cost - new_cost) / current_cost) * 100

class TagUtils:
    """Utilities for working with AWS resource tags"""
    
    @staticmethod
    def extract_environment(tags: Dict[str, str]) -> str:
        """Extract environment from tags"""
        env_tags = ['Environment', 'environment', 'Env', 'env', 'Stage', 'stage']
        
        for tag_key in env_tags:
            if tag_key in tags:
                return tags[tag_key].lower()
        
        # Check in tag values
        for tag_value in tags.values():
            value_lower = tag_value.lower()
            if any(env in value_lower for env in ['prod', 'dev', 'test', 'staging']):
                return value_lower
        
        return 'unknown'
    
    @staticmethod
    def extract_application(tags: Dict[str, str]) -> str:
        """Extract application name from tags"""
        app_tags = ['Application', 'application', 'App', 'app', 'Service', 'service', 'Name', 'name']
        
        for tag_key in app_tags:
            if tag_key in tags:
                return tags[tag_key]
        
        return 'unknown'
    
    @staticmethod
    def extract_owner(tags: Dict[str, str]) -> str:
        """Extract owner from tags"""
        owner_tags = ['Owner', 'owner', 'Team', 'team', 'Contact', 'contact']
        
        for tag_key in owner_tags:
            if tag_key in tags:
                return tags[tag_key]
        
        return 'unknown'
    
    @staticmethod
    def is_critical_resource(tags: Dict[str, str]) -> bool:
        """Check if resource is tagged as critical"""
        critical_patterns = [
            r'critical', r'production', r'prod', r'important',
            r'do.not.delete', r'permanent', r'backup', r'master', r'primary'
        ]
        
        for tag_key, tag_value in tags.items():
            tag_text = f"{tag_key}:{tag_value}".lower()
            if any(re.search(pattern, tag_text, re.IGNORECASE) for pattern in critical_patterns):
                return True
        
        return False
    
    @staticmethod
    def should_optimize(tags: Dict[str, str]) -> bool:
        """Check if resource should be optimized based on tags"""
        # Check for explicit do not optimize tag
        do_not_optimize = tags.get('DoNotOptimize', '').lower()
        if do_not_optimize in ['true', 'yes', '1']:
            return False
        
        # Check for optimization enabled tag
        optimize_enabled = tags.get('OptimizationEnabled', '').lower()
        if optimize_enabled in ['true', 'yes', '1']:
            return True
        
        # Default: allow optimization unless explicitly forbidden
        return True

class ReportGenerator:
    """Utilities for generating reports"""
    
    @staticmethod
    def create_excel_report(data: List[Dict], output_file: str, sheet_name: str = 'Report'):
        """Create Excel report from data"""
        try:
            df = pd.DataFrame(data)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            logger.info(f"Excel report created: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating Excel report: {e}")
    
    @staticmethod
    def create_csv_report(data: List[Dict], output_file: str):
        """Create CSV report from data"""
        try:
            if not data:
                logger.warning("No data to export to CSV")
                return
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"CSV report created: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating CSV report: {e}")
    
    @staticmethod
    def create_json_report(data: Any, output_file: str):
        """Create JSON report from data"""
        try:
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)
            
            logger.info(f"JSON report created: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating JSON report: {e}")

class DateTimeUtils:
    """Date and time utilities"""
    
    @staticmethod
    def get_date_range(days: int) -> Tuple[datetime, datetime]:
        """Get date range for the last N days"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        return start_time, end_time
    
    @staticmethod
    def is_business_hours(dt: datetime = None) -> bool:
        """Check if datetime is within business hours (9 AM - 5 PM UTC, weekdays)"""
        if dt is None:
            dt = datetime.utcnow()
        
        # Check if weekday (0 = Monday, 6 = Sunday)
        if dt.weekday() >= 5:  # Weekend
            return False
        
        # Check if business hours (9 AM - 5 PM UTC)
        if 9 <= dt.hour < 17:
            return True
        
        return False
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"

class ValidationUtils:
    """Validation utilities"""
    
    @staticmethod
    def validate_aws_region(region: str) -> bool:
        """Validate AWS region format"""
        # AWS region pattern: us-east-1, eu-west-1, etc.
        pattern = r'^[a-z]{2}-[a-z]+-\d+$'
        return bool(re.match(pattern, region))
    
    @staticmethod
    def validate_instance_id(instance_id: str) -> bool:
        """Validate EC2 instance ID format"""
        # EC2 instance ID pattern: i-1234567890abcdef0
        pattern = r'^i-[0-9a-f]{8,17}$'
        return bool(re.match(pattern, instance_id))
    
    @staticmethod
    def validate_volume_id(volume_id: str) -> bool:
        """Validate EBS volume ID format"""
        # EBS volume ID pattern: vol-1234567890abcdef0
        pattern = r'^vol-[0-9a-f]{8,17}$'
        return bool(re.match(pattern, volume_id))
    
    @staticmethod
    def validate_bucket_name(bucket_name: str) -> bool:
        """Validate S3 bucket name format"""
        # S3 bucket naming rules (simplified)
        if len(bucket_name) < 3 or len(bucket_name) > 63:
            return False
        
        # Must start and end with lowercase letter or number
        if not (bucket_name[0].isalnum() and bucket_name[-1].isalnum()):
            return False
        
        # No uppercase letters, underscores, or adjacent periods
        if any(c.isupper() or c == '_' for c in bucket_name):
            return False
        
        if '..' in bucket_name:
            return False
        
        return True

class CacheUtils:
    """Simple caching utilities"""
    
    def __init__(self, cache_dir: str = 'cache', ttl_seconds: int = 3600):
        """Initialize cache with directory and TTL"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cache_key = self._get_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if not cache_file.exists():
                return None
            
            # Check if cache is expired
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > self.ttl_seconds:
                cache_file.unlink()  # Remove expired cache
                return None
            
            with open(cache_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        try:
            cache_key = self._get_cache_key(key)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(value, f, default=str)
                
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")
    
    def clear(self):
        """Clear all cache files"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """Initialize progress tracker"""
        self.total_items = total_items
        self.current_item = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current_item += increment
        self._display_progress()
    
    def _display_progress(self):
        """Display progress bar"""
        if self.total_items == 0:
            return
        
        percentage = (self.current_item / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        # Estimate remaining time
        if self.current_item > 0:
            time_per_item = elapsed_time / self.current_item
            remaining_items = self.total_items - self.current_item
            eta = remaining_items * time_per_item
        else:
            eta = 0
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * self.current_item // self.total_items)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format output
        print(f'\r{self.description}: |{bar}| {percentage:.1f}% ({self.current_item}/{self.total_items}) ETA: {eta:.0f}s', end='')
        
        if self.current_item >= self.total_items:
            print(f'\n{self.description} completed in {elapsed_time:.1f} seconds')

# Global utility instances
aws_helper = AWSHelper()
config_manager = ConfigManager()
cost_calculator = CostCalculator()
tag_utils = TagUtils()
report_generator = ReportGenerator()
datetime_utils = DateTimeUtils()
validation_utils = ValidationUtils()

# Convenience functions
def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return config_manager.get_default_config()

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from file"""
    return config_manager.load_config(config_path)

def save_config(config: Dict[str, Any], config_path: str = 'config/config.yaml'):
    """Save configuration to file"""
    return config_manager.save_config(config, config_path)

def test_aws_connection(profile: str = None) -> bool:
    """Test AWS connection"""
    helper = AWSHelper(profile)
    return helper.test_credentials()

def calculate_ec2_cost(instance_type: str, region: str = 'us-east-1') -> float:
    """Calculate EC2 monthly cost"""
    return cost_calculator.calculate_ec2_monthly_cost(instance_type, region)

def is_business_hours() -> bool:
    """Check if current time is business hours"""
    return datetime_utils.is_business_hours()

def validate_resource_id(resource_id: str, resource_type: str) -> bool:
    """Validate AWS resource ID format"""
    if resource_type == 'ec2':
        return validation_utils.validate_instance_id(resource_id)
    elif resource_type == 'ebs':
        return validation_utils.validate_volume_id(resource_id)
    elif resource_type == 's3':
        return validation_utils.validate_bucket_name(resource_id)
    else:
        return True  # Unknown type, assume valid
