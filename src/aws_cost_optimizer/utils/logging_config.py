"""
Logging configuration for AWS Cost Optimizer
"""
import logging
import logging.config
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import colorlog

class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)

def setup_logging(log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 log_format: str = 'console',
                 enable_color: bool = True) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Format type ('console', 'json', 'detailed')
        enable_color: Enable colored output for console
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {},
        'handlers': {},
        'loggers': {
            'aws_cost_optimizer': {
                'level': log_level,
                'handlers': [],
                'propagate': False
            },
            'boto3': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            },
            'botocore': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            },
            'urllib3': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': []
        }
    }
    
    # Configure formatters
    if log_format == 'json':
        config['formatters']['json'] = {
            '()': StructuredFormatter
        }
        formatter_name = 'json'
    elif log_format == 'detailed':
        config['formatters']['detailed'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        formatter_name = 'detailed'
    else:  # console
        if enable_color and sys.stderr.isatty():
            config['formatters']['console'] = {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(levelname)-8s%(reset)s %(message)s',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'
                }
            }
        else:
            config['formatters']['console'] = {
                'format': '%(levelname)-8s %(message)s'
            }
        formatter_name = 'console'
    
    # Configure console handler
    config['handlers']['console'] = {
        'class': 'logging.StreamHandler',
        'level': log_level,
        'formatter': formatter_name,
        'stream': 'ext://sys.stderr'
    }
    
    # Add console handler to all loggers
    for logger_name in config['loggers']:
        config['loggers'][logger_name]['handlers'].append('console')
    config['root']['handlers'].append('console')
    
    # Configure file handler if specified
    if log_file:
        config['formatters']['file'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'file' if log_format != 'json' else 'json',
            'filename': log_file,
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        
        # Add file handler to all loggers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper

def log_resource_action(action: str, 
                       resource_type: str,
                       resource_id: str,
                       details: Dict[str, Any] = None):
    """Log a resource action for audit trail"""
    logger = logging.getLogger('aws_cost_optimizer.audit')
    
    log_entry = {
        'action': action,
        'resource_type': resource_type,
        'resource_id': resource_id,
        'timestamp': datetime.utcnow().isoformat(),
        'details': details or {}
    }
    
    logger.info(f"Resource action: {action} on {resource_type} {resource_id}", 
                extra=log_entry)

class CloudWatchLogHandler(logging.Handler):
    """Custom handler to send logs to CloudWatch Logs"""
    
    def __init__(self, log_group: str, log_stream: str, session=None):
        super().__init__()
        
        if not session:
            import boto3
            session = boto3.Session()
        
        self.logs_client = session.client('logs')
        self.log_group = log_group
        self.log_stream = log_stream
        self.sequence_token = None
        
        # Create log group and stream if they don't exist
        self._ensure_log_group_exists()
        self._ensure_log_stream_exists()
    
    def _ensure_log_group_exists(self):
        """Create log group if it doesn't exist"""
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group)
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass
    
    def _ensure_log_stream_exists(self):
        """Create log stream if it doesn't exist"""
        try:
            self.logs_client.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream
            )
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            # Get the sequence token
            response = self.logs_client.describe_log_streams(
                logGroupName=self.log_group,
                logStreamNamePrefix=self.log_stream
            )
            if response['logStreams']:
                self.sequence_token = response['logStreams'][0].get('uploadSequenceToken')
    
    def emit(self, record):
        """Send log record to CloudWatch"""
        try:
            log_entry = {
                'timestamp': int(record.created * 1000),
                'message': self.format(record)
            }
            
            params = {
                'logGroupName': self.log_group,
                'logStreamName': self.log_stream,
                'logEvents': [log_entry]
            }
            
            if self.sequence_token:
                params['sequenceToken'] = self.sequence_token
            
            response = self.logs_client.put_log_events(**params)
            self.sequence_token = response.get('nextSequenceToken')
            
        except Exception as e:
            # Fallback to stderr
            sys.stderr.write(f"Failed to send log to CloudWatch: {e}\n")

def enable_cloudwatch_logging(log_group: str, log_stream: str = None):
    """Enable CloudWatch Logs integration"""
    if not log_stream:
        log_stream = f"cost-optimizer-{datetime.utcnow().strftime('%Y-%m-%d')}"
    
    handler = CloudWatchLogHandler(log_group, log_stream)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Add to root logger
    logging.getLogger().addHandler(handler)
    
    # Add to specific logger
    logging.getLogger('aws_cost_optimizer').addHandler(handler)

# Convenience function for quick setup
def setup_default_logging():
    """Setup default logging configuration"""
    setup_logging(
        log_level='INFO',
        log_file='logs/aws-cost-optimizer.log',
        log_format='console',
        enable_color=True
    )