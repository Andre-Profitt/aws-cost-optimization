"""
RDS Optimizer module for identifying and eliminating database waste
Targets: Duplicate databases, oversized instances, idle databases
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class RDSOptimizationRecommendation:
    """Represents an RDS optimization recommendation"""
    db_identifier: str
    current_class: str
    recommended_class: Optional[str]
    action: str
    estimated_monthly_savings: float
    confidence: float
    reason: str
    risk_level: str  # low, medium, high
    implementation_steps: List[str]
    rollback_plan: str

@dataclass
class DatabaseFingerprint:
    """Fingerprint for identifying duplicate databases"""
    db_identifier: str
    engine: str
    engine_version: str
    allocated_storage: int
    schema_hash: str
    table_count: int
    approximate_row_count: int
    connection_pattern_hash: str
    tags: Dict[str, str]

class RDSOptimizer:
    """Comprehensive RDS optimization engine"""
    
    # RDS instance pricing (simplified - should load from pricing API)
    INSTANCE_PRICING = {
        'db.t3.micro': 14.40,
        'db.t3.small': 28.80,
        'db.t3.medium': 57.60,
        'db.t3.large': 115.20,
        'db.m5.large': 158.40,
        'db.m5.xlarge': 316.80,
        'db.m5.2xlarge': 633.60,
        'db.r5.large': 201.60,
        'db.r5.xlarge': 403.20,
    }
    
    def __init__(self, 
                 connection_threshold: int = 7,
                 cpu_threshold: float = 25.0,
                 observation_days: int = 60,
                 session: Optional[boto3.Session] = None):
        """
        Initialize RDS Optimizer
        
        Args:
            connection_threshold: Max connections to consider database idle
            cpu_threshold: CPU % threshold for rightsizing
            observation_days: Days of metrics to analyze
            session: Boto3 session (optional)
        """
        self.connection_threshold = connection_threshold
        self.cpu_threshold = cpu_threshold
        self.observation_days = observation_days
        self.session = session or boto3.Session()
        self.rds = self.session.client('rds')
        self.cloudwatch = self.session.client('cloudwatch')
        self.ce = self.session.client('ce')  # Cost Explorer
        
    def analyze_all_databases(self, region_name: str = None) -> List[RDSOptimizationRecommendation]:
        """
        Analyze all RDS instances in the account/region
        
        Args:
            region_name: AWS region (uses session default if None)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Get all RDS instances
        databases = self._get_all_databases(region_name)
        logger.info(f"Found {len(databases)} RDS instances to analyze")
        
        # Analyze each database
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_db = {
                executor.submit(self._analyze_single_database, db): db 
                for db in databases
            }
            
            for future in as_completed(future_to_db):
                db = future_to_db[future]
                try:
                    db_recommendations = future.result()
                    recommendations.extend(db_recommendations)
                except Exception as e:
                    logger.error(f"Failed to analyze {db['DBInstanceIdentifier']}: {e}")
        
        # Find duplicate databases
        duplicate_recommendations = self._find_duplicate_databases(databases)
        recommendations.extend(duplicate_recommendations)
        
        # Find Aurora migration candidates
        aurora_recommendations = self._identify_aurora_candidates(databases)
        recommendations.extend(aurora_recommendations)
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.estimated_monthly_savings, reverse=True)
        
        return recommendations
    
    def _get_all_databases(self, region_name: str = None) -> List[Dict[str, Any]]:
        """Get all RDS instances in the region"""
        if region_name:
            rds = self.session.client('rds', region_name=region_name)
        else:
            rds = self.rds
            
        databases = []
        paginator = rds.get_paginator('describe_db_instances')
        
        for page in paginator.paginate():
            databases.extend(page['DBInstances'])
            
        return databases
    
    def _analyze_single_database(self, db: Dict[str, Any]) -> List[RDSOptimizationRecommendation]:
        """Analyze a single database for optimization opportunities"""
        recommendations = []
        db_id = db['DBInstanceIdentifier']
        
        # Skip if database is not available
        if db['DBInstanceStatus'] != 'available':
            logger.info(f"Skipping {db_id} - status: {db['DBInstanceStatus']}")
            return recommendations
        
        # Get CloudWatch metrics
        metrics = self._get_database_metrics(db_id)
        
        # Check if database is idle
        idle_rec = self._check_idle_database(db, metrics)
        if idle_rec:
            recommendations.append(idle_rec)
            
        # Check for rightsizing opportunities
        rightsize_rec = self._check_rightsizing(db, metrics)
        if rightsize_rec:
            recommendations.append(rightsize_rec)
            
        # Check for Multi-AZ optimization
        multi_az_rec = self._check_multi_az_necessity(db, metrics)
        if multi_az_rec:
            recommendations.append(multi_az_rec)
            
        # Check backup retention optimization
        backup_rec = self._check_backup_optimization(db)
        if backup_rec:
            recommendations.append(backup_rec)
            
        return recommendations
    
    def _get_database_metrics(self, db_identifier: str) -> Dict[str, Any]:
        """Get CloudWatch metrics for a database"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.observation_days)
        
        metrics = {}
        
        # Get CPU utilization
        cpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_identifier}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average', 'Maximum']
        )
        
        if cpu_response['Datapoints']:
            cpu_data = pd.DataFrame(cpu_response['Datapoints'])
            metrics['cpu_avg'] = cpu_data['Average'].mean()
            metrics['cpu_max'] = cpu_data['Maximum'].max()
            metrics['cpu_p95'] = cpu_data['Average'].quantile(0.95)
        else:
            metrics['cpu_avg'] = 0
            metrics['cpu_max'] = 0
            metrics['cpu_p95'] = 0
        
        # Get database connections
        conn_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='DatabaseConnections',
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_identifier}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )
        
        if conn_response['Datapoints']:
            conn_data = pd.DataFrame(conn_response['Datapoints'])
            metrics['connections_avg'] = conn_data['Average'].mean()
            metrics['connections_max'] = conn_data['Maximum'].max()
        else:
            metrics['connections_avg'] = 0
            metrics['connections_max'] = 0
        
        # Get read/write IOPS
        read_iops_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/RDS',
            MetricName='ReadIOPS',
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_identifier}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Sum']
        )
        
        if read_iops_response['Datapoints']:
            metrics['read_iops_avg'] = pd.DataFrame(read_iops_response['Datapoints'])['Average'].mean()
        else:
            metrics['read_iops_avg'] = 0
            
        return metrics
    
    def _check_idle_database(self, db: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[RDSOptimizationRecommendation]:
        """Check if database is idle and can be stopped or deleted"""
        db_id = db['DBInstanceIdentifier']
        
        # Check connection patterns
        if metrics['connections_avg'] < 1 and metrics['connections_max'] < self.connection_threshold:
            # Check if it's a non-production database
            tags = {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
            env = tags.get('Environment', '').lower()
            
            if env in ['dev', 'test', 'staging', 'development']:
                current_cost = self._calculate_monthly_cost(db)
                
                return RDSOptimizationRecommendation(
                    db_identifier=db_id,
                    current_class=db['DBInstanceClass'],
                    recommended_class=None,
                    action='stop' if 'aurora' not in db['Engine'] else 'delete',
                    estimated_monthly_savings=current_cost * 0.9,  # Save 90% by stopping
                    confidence=0.95,
                    reason=f"Database has avg {metrics['connections_avg']:.1f} connections, max {metrics['connections_max']:.0f}",
                    risk_level='low' if env != 'staging' else 'medium',
                    implementation_steps=[
                        f"1. Create final snapshot: aws rds create-db-snapshot --db-instance-identifier {db_id}",
                        f"2. Stop database: aws rds stop-db-instance --db-instance-identifier {db_id}",
                        "3. Set up CloudWatch alarm to monitor for restart needs",
                        "4. Document in team wiki"
                    ],
                    rollback_plan=f"aws rds start-db-instance --db-instance-identifier {db_id}"
                )
        
        return None
    
    def _check_rightsizing(self, db: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[RDSOptimizationRecommendation]:
        """Check if database can be rightsized to a smaller instance"""
        current_class = db['DBInstanceClass']
        
        # Only rightsize if CPU is consistently low
        if metrics['cpu_avg'] < self.cpu_threshold and metrics['cpu_p95'] < self.cpu_threshold * 1.5:
            # Determine recommended instance class
            recommended_class = self._get_recommended_instance_class(
                current_class, 
                metrics['cpu_avg'],
                db['AllocatedStorage']
            )
            
            if recommended_class and recommended_class != current_class:
                current_cost = self._calculate_monthly_cost(db)
                new_cost = self._calculate_monthly_cost(db, override_class=recommended_class)
                savings = current_cost - new_cost
                
                if savings > 50:  # Only recommend if saving > $50/month
                    return RDSOptimizationRecommendation(
                        db_identifier=db['DBInstanceIdentifier'],
                        current_class=current_class,
                        recommended_class=recommended_class,
                        action='rightsize',
                        estimated_monthly_savings=savings,
                        confidence=0.8,
                        reason=f"CPU averages {metrics['cpu_avg']:.1f}%, P95 {metrics['cpu_p95']:.1f}%",
                        risk_level='medium',
                        implementation_steps=[
                            f"1. Create snapshot: aws rds create-db-snapshot --db-instance-identifier {db['DBInstanceIdentifier']}",
                            f"2. Modify instance: aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --db-instance-class {recommended_class} --apply-immediately",
                            "3. Monitor performance for 24 hours",
                            "4. Be ready to scale up if needed"
                        ],
                        rollback_plan=f"aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --db-instance-class {current_class} --apply-immediately"
                    )
        
        return None
    
    def _check_multi_az_necessity(self, db: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[RDSOptimizationRecommendation]:
        """Check if Multi-AZ is necessary for non-critical databases"""
        if not db.get('MultiAZ', False):
            return None
            
        tags = {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
        env = tags.get('Environment', '').lower()
        
        # Only recommend disabling Multi-AZ for non-production
        if env in ['dev', 'test', 'development']:
            current_cost = self._calculate_monthly_cost(db)
            savings = current_cost * 0.5  # Multi-AZ doubles the cost
            
            return RDSOptimizationRecommendation(
                db_identifier=db['DBInstanceIdentifier'],
                current_class=db['DBInstanceClass'],
                recommended_class=db['DBInstanceClass'],
                action='disable_multi_az',
                estimated_monthly_savings=savings,
                confidence=0.9,
                reason=f"Non-production database ({env}) doesn't require Multi-AZ",
                risk_level='low',
                implementation_steps=[
                    f"1. Create snapshot: aws rds create-db-snapshot --db-instance-identifier {db['DBInstanceIdentifier']}",
                    f"2. Disable Multi-AZ: aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --no-multi-az --apply-immediately",
                    "3. Verify single-AZ operation"
                ],
                rollback_plan=f"aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --multi-az --apply-immediately"
            )
            
        return None
    
    def _check_backup_optimization(self, db: Dict[str, Any]) -> Optional[RDSOptimizationRecommendation]:
        """Check if backup retention can be optimized"""
        retention_period = db.get('BackupRetentionPeriod', 0)
        
        if retention_period > 7:
            tags = {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
            env = tags.get('Environment', '').lower()
            
            if env in ['dev', 'test']:
                # Dev/test doesn't need long retention
                recommended_retention = 1
            elif env == 'staging':
                recommended_retention = 3
            else:
                return None  # Don't change production
                
            if recommended_retention < retention_period:
                # Rough estimate: each backup day costs ~2% of instance cost
                current_cost = self._calculate_monthly_cost(db)
                savings = current_cost * 0.02 * (retention_period - recommended_retention)
                
                return RDSOptimizationRecommendation(
                    db_identifier=db['DBInstanceIdentifier'],
                    current_class=db['DBInstanceClass'],
                    recommended_class=db['DBInstanceClass'],
                    action='reduce_backup_retention',
                    estimated_monthly_savings=savings,
                    confidence=0.95,
                    reason=f"{env} database has {retention_period} day retention",
                    risk_level='low',
                    implementation_steps=[
                        f"1. Modify retention: aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --backup-retention-period {recommended_retention} --apply-immediately"
                    ],
                    rollback_plan=f"aws rds modify-db-instance --db-instance-identifier {db['DBInstanceIdentifier']} --backup-retention-period {retention_period}"
                )
                
        return None
    
    def _find_duplicate_databases(self, databases: List[Dict[str, Any]]) -> List[RDSOptimizationRecommendation]:
        """Find potential duplicate databases based on fingerprinting"""
        recommendations = []
        fingerprints = []
        
        # Generate fingerprints for each database
        for db in databases:
            if db['DBInstanceStatus'] == 'available':
                fingerprint = self._generate_database_fingerprint(db)
                if fingerprint:
                    fingerprints.append(fingerprint)
        
        # Group by similar characteristics
        grouped = self._group_similar_databases(fingerprints)
        
        # Analyze each group for duplicates
        for group in grouped:
            if len(group) > 1:
                # Sort by environment (prod > staging > test > dev)
                env_priority = {'production': 4, 'prod': 4, 'staging': 3, 'test': 2, 'dev': 1, 'development': 1}
                group.sort(key=lambda fp: env_priority.get(fp.tags.get('Environment', '').lower(), 0), reverse=True)
                
                # Keep the highest priority, recommend removing others
                primary = group[0]
                for duplicate in group[1:]:
                    db = next(d for d in databases if d['DBInstanceIdentifier'] == duplicate.db_identifier)
                    cost = self._calculate_monthly_cost(db)
                    
                    recommendations.append(RDSOptimizationRecommendation(
                        db_identifier=duplicate.db_identifier,
                        current_class=db['DBInstanceClass'],
                        recommended_class=None,
                        action='delete_duplicate',
                        estimated_monthly_savings=cost,
                        confidence=0.7,
                        reason=f"Appears to be duplicate of {primary.db_identifier}",
                        risk_level='high',
                        implementation_steps=[
                            f"1. Verify not in use: Check application configs and connection logs",
                            f"2. Create final snapshot: aws rds create-db-snapshot --db-instance-identifier {duplicate.db_identifier}",
                            f"3. Stop for 7 days: aws rds stop-db-instance --db-instance-identifier {duplicate.db_identifier}",
                            f"4. If no issues, delete: aws rds delete-db-instance --db-instance-identifier {duplicate.db_identifier} --skip-final-snapshot"
                        ],
                        rollback_plan=f"Restore from snapshot created in step 2"
                    ))
                    
        return recommendations
    
    def _generate_database_fingerprint(self, db: Dict[str, Any]) -> Optional[DatabaseFingerprint]:
        """Generate a fingerprint for database comparison"""
        try:
            # Get basic properties
            tags = {tag['Key']: tag['Value'] for tag in db.get('TagList', [])}
            
            # Create a simple schema hash based on available info
            schema_components = [
                db['Engine'],
                db['EngineVersion'],
                str(db['AllocatedStorage']),
                db.get('DBName', 'default')
            ]
            schema_hash = hashlib.md5('|'.join(schema_components).encode()).hexdigest()
            
            return DatabaseFingerprint(
                db_identifier=db['DBInstanceIdentifier'],
                engine=db['Engine'],
                engine_version=db['EngineVersion'],
                allocated_storage=db['AllocatedStorage'],
                schema_hash=schema_hash,
                table_count=0,  # Would need to connect to get actual count
                approximate_row_count=0,  # Would need to connect
                connection_pattern_hash="",  # Would analyze CloudWatch patterns
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to fingerprint {db['DBInstanceIdentifier']}: {e}")
            return None
    
    def _group_similar_databases(self, fingerprints: List[DatabaseFingerprint]) -> List[List[DatabaseFingerprint]]:
        """Group databases that appear to be similar/duplicates"""
        groups = []
        used = set()
        
        for i, fp1 in enumerate(fingerprints):
            if fp1.db_identifier in used:
                continue
                
            group = [fp1]
            used.add(fp1.db_identifier)
            
            for j, fp2 in enumerate(fingerprints[i+1:], i+1):
                if fp2.db_identifier in used:
                    continue
                    
                # Check similarity
                if (fp1.engine == fp2.engine and 
                    fp1.allocated_storage == fp2.allocated_storage and
                    self._are_versions_compatible(fp1.engine_version, fp2.engine_version)):
                    
                    # Check if names suggest relationship (e.g., myapp-prod vs myapp-dev)
                    if self._are_names_related(fp1.db_identifier, fp2.db_identifier):
                        group.append(fp2)
                        used.add(fp2.db_identifier)
                        
            if len(group) > 1:
                groups.append(group)
                
        return groups
    
    def _are_versions_compatible(self, v1: str, v2: str) -> bool:
        """Check if two database versions are compatible (same major version)"""
        try:
            major1 = v1.split('.')[0]
            major2 = v2.split('.')[0]
            return major1 == major2
        except:
            return False
    
    def _are_names_related(self, name1: str, name2: str) -> bool:
        """Check if database names suggest they're related"""
        # Remove common environment suffixes
        environments = ['-prod', '-production', '-staging', '-stage', '-test', '-dev', '-development', '-qa']
        
        base1 = name1.lower()
        base2 = name2.lower()
        
        for env in environments:
            base1 = base1.replace(env, '')
            base2 = base2.replace(env, '')
            
        # Check if bases are similar
        return base1 == base2 or base1.startswith(base2) or base2.startswith(base1)
    
    def _identify_aurora_candidates(self, databases: List[Dict[str, Any]]) -> List[RDSOptimizationRecommendation]:
        """Identify databases that could benefit from Aurora Serverless v2"""
        recommendations = []
        
        for db in databases:
            # Only consider MySQL and PostgreSQL
            if db['Engine'] not in ['mysql', 'postgres']:
                continue
                
            # Skip if already Aurora
            if 'aurora' in db['Engine']:
                continue
                
            # Get metrics
            metrics = self._get_database_metrics(db['DBInstanceIdentifier'])
            
            # Good candidate if variable load
            if metrics['connections_max'] > 5 * metrics['connections_avg']:
                current_cost = self._calculate_monthly_cost(db)
                # Aurora Serverless v2 can save 30-50% for variable workloads
                estimated_savings = current_cost * 0.3
                
                recommendations.append(RDSOptimizationRecommendation(
                    db_identifier=db['DBInstanceIdentifier'],
                    current_class=db['DBInstanceClass'],
                    recommended_class='Aurora Serverless v2',
                    action='migrate_to_aurora_serverless',
                    estimated_monthly_savings=estimated_savings,
                    confidence=0.6,
                    reason=f"Variable load: avg {metrics['connections_avg']:.1f} connections, max {metrics['connections_max']:.0f}",
                    risk_level='high',
                    implementation_steps=[
                        "1. Create Aurora Serverless v2 cluster",
                        "2. Set up DMS replication from source database",
                        "3. Test application with Aurora endpoint",
                        "4. Cutover during maintenance window",
                        "5. Monitor performance and costs"
                    ],
                    rollback_plan="Switch application back to original database, stop DMS replication"
                ))
                
        return recommendations
    
    def _get_recommended_instance_class(self, current_class: str, avg_cpu: float, storage_gb: int) -> Optional[str]:
        """Determine recommended instance class based on usage"""
        # Parse current class
        parts = current_class.split('.')
        if len(parts) != 3:
            return None
            
        family = parts[1]  # e.g., 'm5', 'r5', 't3'
        size = parts[2]    # e.g., 'large', 'xlarge'
        
        # Define size hierarchy
        sizes = ['micro', 'small', 'medium', 'large', 'xlarge', '2xlarge', '4xlarge', '8xlarge']
        
        current_size_idx = sizes.index(size) if size in sizes else -1
        if current_size_idx == -1:
            return None
            
        # Recommend downsizing if CPU is very low
        if avg_cpu < 10 and current_size_idx > 0:
            # Downsize by 1-2 levels
            new_size_idx = max(0, current_size_idx - 2)
        elif avg_cpu < 20 and current_size_idx > 0:
            # Downsize by 1 level
            new_size_idx = current_size_idx - 1
        else:
            return None  # No downsizing needed
            
        # Construct new class
        new_class = f"db.{family}.{sizes[new_size_idx]}"
        
        # Verify it exists in our pricing
        if new_class in self.INSTANCE_PRICING:
            return new_class
            
        # Try t3 family for cost optimization
        t3_class = f"db.t3.{sizes[new_size_idx]}"
        if t3_class in self.INSTANCE_PRICING:
            return t3_class
            
        return None
    
    def _calculate_monthly_cost(self, db: Dict[str, Any], override_class: str = None) -> float:
        """Calculate monthly cost for a database instance"""
        instance_class = override_class or db['DBInstanceClass']
        
        # Base instance cost
        base_cost = self.INSTANCE_PRICING.get(instance_class, 200)  # Default $200
        
        # Multi-AZ doubles the cost
        if db.get('MultiAZ', False) and not override_class:
            base_cost *= 2
            
        # Add storage cost (~$0.10 per GB)
        storage_cost = db['AllocatedStorage'] * 0.10
        
        # Add backup storage cost
        backup_cost = db['AllocatedStorage'] * 0.05 * (db.get('BackupRetentionPeriod', 7) / 30)
        
        return base_cost + storage_cost + backup_cost
    
    def generate_optimization_report(self, recommendations: List[RDSOptimizationRecommendation]) -> pd.DataFrame:
        """Generate a detailed optimization report"""
        data = []
        
        for rec in recommendations:
            data.append({
                'Database': rec.db_identifier,
                'Current Class': rec.current_class,
                'Recommended Class': rec.recommended_class or 'N/A',
                'Action': rec.action,
                'Monthly Savings': f"${rec.estimated_monthly_savings:,.2f}",
                'Annual Savings': f"${rec.estimated_monthly_savings * 12:,.2f}",
                'Confidence': f"{rec.confidence:.0%}",
                'Risk Level': rec.risk_level,
                'Reason': rec.reason
            })
            
        df = pd.DataFrame(data)
        
        # Add summary
        total_monthly = sum(rec.estimated_monthly_savings for rec in recommendations)
        total_annual = total_monthly * 12
        
        print(f"\nRDS Optimization Summary:")
        print(f"Total Recommendations: {len(recommendations)}")
        print(f"Total Monthly Savings: ${total_monthly:,.2f}")
        print(f"Total Annual Savings: ${total_annual:,.2f}")
        print(f"\nTop 5 Opportunities:")
        print(df.head())
        
        return df
    
    def export_recommendations(self, 
                             recommendations: List[RDSOptimizationRecommendation],
                             output_file: str = 'rds_optimization_report.xlsx'):
        """Export recommendations to Excel with multiple sheets"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self.generate_optimization_report(recommendations)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed recommendations
            detailed_data = []
            for rec in recommendations:
                detailed_data.append({
                    'Database': rec.db_identifier,
                    'Action': rec.action,
                    'Monthly Savings': rec.estimated_monthly_savings,
                    'Implementation Steps': '\n'.join(rec.implementation_steps),
                    'Rollback Plan': rec.rollback_plan
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Implementation Details', index=False)
            
            # Risk matrix
            risk_data = []
            for risk_level in ['low', 'medium', 'high']:
                risk_recs = [r for r in recommendations if r.risk_level == risk_level]
                risk_data.append({
                    'Risk Level': risk_level.upper(),
                    'Count': len(risk_recs),
                    'Total Monthly Savings': sum(r.estimated_monthly_savings for r in risk_recs)
                })
            
            risk_df = pd.DataFrame(risk_data)
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
        logger.info(f"Exported RDS optimization report to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = RDSOptimizer(
        connection_threshold=7,
        cpu_threshold=25.0,
        observation_days=60
    )
    
    # Analyze all databases
    recommendations = optimizer.analyze_all_databases()
    
    # Generate and export report
    optimizer.export_recommendations(recommendations, 'rds_optimization_report.xlsx')