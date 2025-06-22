"""Discovery module for AWS resources across multiple accounts"""

from .s3_discovery import S3Discovery

__all__ = ['S3Discovery']