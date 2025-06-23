"""
Savings Tracker Module

Tracks implemented optimizations, compares projected vs actual savings,
and provides detailed reporting on cost optimization effectiveness.
"""

import json
import boto3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimizations"""
    RIGHTSIZING = "rightsizing"
    SCHEDULING = "scheduling"
    TERMINATION = "termination"
    RESERVED_INSTANCE = "reserved_instance"
    SAVINGS_PLAN = "savings_plan"
    STORAGE_OPTIMIZATION = "storage_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    LICENSE_OPTIMIZATION = "license_optimization"
    TAGGING = "tagging"
    OTHER = "other"


class SavingsStatus(Enum):
    """Status of savings realization"""
    PROJECTED = "projected"
    IN_PROGRESS = "in_progress"
    REALIZED = "realized"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class OptimizationRecord:
    """Record of an optimization action"""
    optimization_id: str
    resource_id: str
    resource_type: str
    optimization_type: OptimizationType
    description: str
    implemented_date: datetime
    projected_monthly_savings: float
    actual_monthly_savings: Optional[float] = None
    status: SavingsStatus = SavingsStatus.PROJECTED
    confidence_score: float = 0.8
    rollback_available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class SavingsSummary:
    """Summary of savings for a period"""
    period_start: datetime
    period_end: datetime
    total_projected_savings: float
    total_realized_savings: float
    realization_rate: float  # actual/projected
    by_type: Dict[OptimizationType, float]
    by_service: Dict[str, float]
    top_optimizations: List[OptimizationRecord]
    failed_optimizations: List[OptimizationRecord]


class SavingsTracker:
    """Track and analyze cost optimization savings"""
    
    def __init__(self,
                 dynamodb_table: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 ce_client=None,
                 dynamodb_client=None,
                 s3_client=None):
        """
        Initialize savings tracker
        
        Args:
            dynamodb_table: DynamoDB table for storing optimization records
            s3_bucket: S3 bucket for detailed reports
            ce_client: Boto3 Cost Explorer client
            dynamodb_client: Boto3 DynamoDB client
            s3_client: Boto3 S3 client
        """
        self.dynamodb_table = dynamodb_table
        self.s3_bucket = s3_bucket
        
        # AWS clients
        self.ce_client = ce_client or boto3.client('ce')
        self.dynamodb = dynamodb_client or boto3.client('dynamodb')
        self.s3_client = s3_client or boto3.client('s3')
        
        # Local cache
        self.optimization_cache: Dict[str, OptimizationRecord] = {}
        
    def record_optimization(self, optimization: OptimizationRecord) -> str:
        """
        Record a new optimization
        
        Args:
            optimization: Optimization record to store
            
        Returns:
            Optimization ID
        """
        logger.info(f"Recording optimization: {optimization.optimization_id}")
        
        # Store in cache
        self.optimization_cache[optimization.optimization_id] = optimization
        
        # Store in DynamoDB if configured
        if self.dynamodb_table:
            self._store_in_dynamodb(optimization)
        
        # Log the optimization
        logger.info(f"Recorded {optimization.optimization_type.value} optimization for {optimization.resource_id} "
                   f"with projected savings of ${optimization.projected_monthly_savings:.2f}/month")
        
        return optimization.optimization_id
    
    def update_actual_savings(self,
                            optimization_id: str,
                            actual_savings: float,
                            status: Optional[SavingsStatus] = None) -> OptimizationRecord:
        """
        Update actual savings for an optimization
        
        Args:
            optimization_id: ID of the optimization
            actual_savings: Actual monthly savings achieved
            status: Updated status
            
        Returns:
            Updated optimization record
        """
        optimization = self._get_optimization(optimization_id)
        if not optimization:
            raise ValueError(f"Optimization {optimization_id} not found")
        
        optimization.actual_monthly_savings = actual_savings
        
        # Auto-determine status if not provided
        if status:
            optimization.status = status
        else:
            if actual_savings >= optimization.projected_monthly_savings * 0.9:
                optimization.status = SavingsStatus.REALIZED
            elif actual_savings >= optimization.projected_monthly_savings * 0.5:
                optimization.status = SavingsStatus.PARTIAL
            else:
                optimization.status = SavingsStatus.FAILED
        
        # Update in storage
        if self.dynamodb_table:
            self._store_in_dynamodb(optimization)
        
        logger.info(f"Updated optimization {optimization_id}: "
                   f"Actual savings ${actual_savings:.2f}/month ({optimization.status.value})")
        
        return optimization
    
    def calculate_savings_metrics(self,
                                start_date: datetime,
                                end_date: datetime,
                                service_filter: Optional[str] = None) -> SavingsSummary:
        """
        Calculate savings metrics for a period
        
        Args:
            start_date: Start of period
            end_date: End of period
            service_filter: Optional service to filter by
            
        Returns:
            SavingsSummary for the period
        """
        # Get all optimizations in the period
        optimizations = self._get_optimizations_in_period(start_date, end_date)
        
        # Filter by service if specified
        if service_filter:
            optimizations = [
                opt for opt in optimizations
                if opt.metadata.get('service') == service_filter
            ]
        
        # Calculate totals
        total_projected = sum(opt.projected_monthly_savings for opt in optimizations)
        total_realized = sum(
            opt.actual_monthly_savings or 0
            for opt in optimizations
            if opt.status in [SavingsStatus.REALIZED, SavingsStatus.PARTIAL]
        )
        
        # Group by type
        by_type = {}
        for opt_type in OptimizationType:
            type_savings = sum(
                opt.actual_monthly_savings or opt.projected_monthly_savings
                for opt in optimizations
                if opt.optimization_type == opt_type
            )
            if type_savings > 0:
                by_type[opt_type] = type_savings
        
        # Group by service
        by_service = {}
        for opt in optimizations:
            service = opt.metadata.get('service', 'Unknown')
            savings = opt.actual_monthly_savings or opt.projected_monthly_savings
            by_service[service] = by_service.get(service, 0) + savings
        
        # Find top and failed optimizations
        sorted_opts = sorted(
            optimizations,
            key=lambda x: x.actual_monthly_savings or x.projected_monthly_savings,
            reverse=True
        )
        
        top_optimizations = sorted_opts[:10]
        failed_optimizations = [
            opt for opt in optimizations
            if opt.status == SavingsStatus.FAILED
        ]
        
        return SavingsSummary(
            period_start=start_date,
            period_end=end_date,
            total_projected_savings=total_projected,
            total_realized_savings=total_realized,
            realization_rate=total_realized / total_projected if total_projected > 0 else 0,
            by_type=by_type,
            by_service=by_service,
            top_optimizations=top_optimizations,
            failed_optimizations=failed_optimizations
        )
    
    def compare_projected_vs_actual(self,
                                  optimization_ids: Optional[List[str]] = None,
                                  days_back: int = 30) -> Dict[str, Any]:
        """
        Compare projected vs actual savings
        
        Args:
            optimization_ids: Specific optimizations to analyze
            days_back: Number of days to look back
            
        Returns:
            Comparison analysis
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        if optimization_ids:
            optimizations = [
                self._get_optimization(opt_id)
                for opt_id in optimization_ids
                if self._get_optimization(opt_id)
            ]
        else:
            optimizations = self._get_optimizations_in_period(start_date, end_date)
        
        # Only include optimizations that have actual savings data
        optimizations = [
            opt for opt in optimizations
            if opt.actual_monthly_savings is not None
        ]
        
        if not optimizations:
            return {
                'error': 'No optimizations with actual savings data found',
                'period_days': days_back
            }
        
        # Calculate statistics
        projected = [opt.projected_monthly_savings for opt in optimizations]
        actual = [opt.actual_monthly_savings for opt in optimizations]
        
        # Accuracy metrics
        differences = [a - p for a, p in zip(actual, projected)]
        percentage_differences = [
            (a - p) / p * 100 if p > 0 else 0
            for a, p in zip(actual, projected)
        ]
        
        # Group by optimization type
        accuracy_by_type = {}
        for opt_type in OptimizationType:
            type_opts = [
                opt for opt in optimizations
                if opt.optimization_type == opt_type
            ]
            if type_opts:
                type_projected = sum(opt.projected_monthly_savings for opt in type_opts)
                type_actual = sum(opt.actual_monthly_savings for opt in type_opts)
                accuracy_by_type[opt_type.value] = {
                    'count': len(type_opts),
                    'projected': type_projected,
                    'actual': type_actual,
                    'accuracy': type_actual / type_projected if type_projected > 0 else 0
                }
        
        return {
            'period_days': days_back,
            'optimization_count': len(optimizations),
            'total_projected': sum(projected),
            'total_actual': sum(actual),
            'overall_accuracy': sum(actual) / sum(projected) if sum(projected) > 0 else 0,
            'mean_difference': np.mean(differences),
            'median_difference': np.median(differences),
            'std_difference': np.std(differences),
            'mean_percentage_difference': np.mean(percentage_differences),
            'overestimated_count': sum(1 for d in differences if d < 0),
            'underestimated_count': sum(1 for d in differences if d > 0),
            'accurate_count': sum(1 for p in percentage_differences if abs(p) <= 10),
            'accuracy_by_type': accuracy_by_type,
            'worst_predictions': [
                {
                    'optimization_id': opt.optimization_id,
                    'resource_id': opt.resource_id,
                    'type': opt.optimization_type.value,
                    'projected': opt.projected_monthly_savings,
                    'actual': opt.actual_monthly_savings,
                    'difference_pct': (opt.actual_monthly_savings - opt.projected_monthly_savings) / opt.projected_monthly_savings * 100
                }
                for opt in sorted(
                    optimizations,
                    key=lambda x: abs(x.actual_monthly_savings - x.projected_monthly_savings),
                    reverse=True
                )[:5]
            ]
        }
    
    def get_savings_trend(self,
                        period_days: int = 90,
                        granularity: str = 'daily') -> pd.DataFrame:
        """
        Get savings trend over time
        
        Args:
            period_days: Number of days to analyze
            granularity: 'daily', 'weekly', or 'monthly'
            
        Returns:
            DataFrame with savings trend
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get all optimizations
        optimizations = self._get_optimizations_in_period(
            start_date - timedelta(days=365),  # Look back further to catch earlier optimizations
            end_date
        )
        
        # Create daily savings data
        date_range = pd.date_range(start_date, end_date, freq='D')
        daily_data = []
        
        for date in date_range:
            # Calculate cumulative savings up to this date
            active_optimizations = [
                opt for opt in optimizations
                if opt.implemented_date <= date
            ]
            
            projected = sum(opt.projected_monthly_savings for opt in active_optimizations) / 30  # Daily
            realized = sum(
                (opt.actual_monthly_savings or 0) / 30
                for opt in active_optimizations
                if opt.status in [SavingsStatus.REALIZED, SavingsStatus.PARTIAL]
            )
            
            daily_data.append({
                'date': date,
                'projected_daily_savings': projected,
                'realized_daily_savings': realized,
                'active_optimizations': len(active_optimizations)
            })
        
        df = pd.DataFrame(daily_data)
        df.set_index('date', inplace=True)
        
        # Aggregate based on granularity
        if granularity == 'weekly':
            df = df.resample('W').agg({
                'projected_daily_savings': 'sum',
                'realized_daily_savings': 'sum',
                'active_optimizations': 'last'
            })
            df.columns = ['projected_weekly_savings', 'realized_weekly_savings', 'active_optimizations']
        elif granularity == 'monthly':
            df = df.resample('M').agg({
                'projected_daily_savings': 'sum',
                'realized_daily_savings': 'sum',
                'active_optimizations': 'last'
            })
            df.columns = ['projected_monthly_savings', 'realized_monthly_savings', 'active_optimizations']
        
        # Add cumulative columns
        df['cumulative_projected'] = df.iloc[:, 0].cumsum()
        df['cumulative_realized'] = df.iloc[:, 1].cumsum()
        
        return df
    
    def generate_executive_report(self,
                                period_days: int = 30,
                                output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate executive savings report
        
        Args:
            period_days: Period to analyze
            output_file: Optional file to save report
            
        Returns:
            Executive report data
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get summary
        summary = self.calculate_savings_metrics(start_date, end_date)
        
        # Get comparison data
        comparison = self.compare_projected_vs_actual(days_back=period_days)
        
        # Get trend data
        trend_df = self.get_savings_trend(period_days=period_days, granularity='weekly')
        
        # Calculate additional metrics
        optimizations = self._get_optimizations_in_period(start_date, end_date)
        
        # Success rate by type
        success_by_type = {}
        for opt_type in OptimizationType:
            type_opts = [opt for opt in optimizations if opt.optimization_type == opt_type]
            if type_opts:
                successful = sum(
                    1 for opt in type_opts
                    if opt.status in [SavingsStatus.REALIZED, SavingsStatus.PARTIAL]
                )
                success_by_type[opt_type.value] = {
                    'total': len(type_opts),
                    'successful': successful,
                    'success_rate': successful / len(type_opts)
                }
        
        # Build report
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': period_days
            },
            'executive_summary': {
                'total_savings_realized': summary.total_realized_savings,
                'total_savings_projected': summary.total_projected_savings,
                'realization_rate': summary.realization_rate,
                'annualized_savings': summary.total_realized_savings * 12,
                'optimization_count': len(optimizations),
                'success_rate': sum(
                    1 for opt in optimizations
                    if opt.status in [SavingsStatus.REALIZED, SavingsStatus.PARTIAL]
                ) / len(optimizations) if optimizations else 0
            },
            'savings_by_type': {
                k.value: v for k, v in summary.by_type.items()
            },
            'savings_by_service': summary.by_service,
            'top_optimizations': [
                {
                    'resource_id': opt.resource_id,
                    'type': opt.optimization_type.value,
                    'description': opt.description,
                    'monthly_savings': opt.actual_monthly_savings or opt.projected_monthly_savings,
                    'status': opt.status.value
                }
                for opt in summary.top_optimizations[:5]
            ],
            'projection_accuracy': {
                'overall_accuracy': comparison.get('overall_accuracy', 0),
                'accurate_predictions': comparison.get('accurate_count', 0),
                'total_predictions': comparison.get('optimization_count', 0)
            },
            'success_rates_by_type': success_by_type,
            'trend_summary': {
                'weekly_average': trend_df['realized_weekly_savings'].mean() if 'realized_weekly_savings' in trend_df else 0,
                'growth_rate': (
                    (trend_df['cumulative_realized'].iloc[-1] - trend_df['cumulative_realized'].iloc[0]) /
                    trend_df['cumulative_realized'].iloc[0]
                ) if len(trend_df) > 1 and trend_df['cumulative_realized'].iloc[0] > 0 else 0
            },
            'recommendations': self._generate_recommendations(optimizations, summary)
        }
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Executive report saved to {output_file}")
        
        return report
    
    def _get_optimization(self, optimization_id: str) -> Optional[OptimizationRecord]:
        """Get optimization from cache or storage"""
        # Check cache first
        if optimization_id in self.optimization_cache:
            return self.optimization_cache[optimization_id]
        
        # Load from DynamoDB if configured
        if self.dynamodb_table:
            return self._load_from_dynamodb(optimization_id)
        
        return None
    
    def _get_optimizations_in_period(self,
                                   start_date: datetime,
                                   end_date: datetime) -> List[OptimizationRecord]:
        """Get all optimizations in a time period"""
        optimizations = []
        
        # From cache
        for opt in self.optimization_cache.values():
            if start_date <= opt.implemented_date <= end_date:
                optimizations.append(opt)
        
        # From DynamoDB if configured
        if self.dynamodb_table:
            db_opts = self._query_dynamodb_period(start_date, end_date)
            optimizations.extend(db_opts)
        
        # Deduplicate
        seen = set()
        unique_opts = []
        for opt in optimizations:
            if opt.optimization_id not in seen:
                seen.add(opt.optimization_id)
                unique_opts.append(opt)
        
        return unique_opts
    
    def _store_in_dynamodb(self, optimization: OptimizationRecord):
        """Store optimization in DynamoDB"""
        try:
            item = {
                'optimization_id': {'S': optimization.optimization_id},
                'resource_id': {'S': optimization.resource_id},
                'resource_type': {'S': optimization.resource_type},
                'optimization_type': {'S': optimization.optimization_type.value},
                'description': {'S': optimization.description},
                'implemented_date': {'S': optimization.implemented_date.isoformat()},
                'projected_monthly_savings': {'N': str(optimization.projected_monthly_savings)},
                'status': {'S': optimization.status.value},
                'confidence_score': {'N': str(optimization.confidence_score)},
                'rollback_available': {'BOOL': optimization.rollback_available},
                'metadata': {'S': json.dumps(optimization.metadata)}
            }
            
            if optimization.actual_monthly_savings is not None:
                item['actual_monthly_savings'] = {'N': str(optimization.actual_monthly_savings)}
            
            self.dynamodb.put_item(
                TableName=self.dynamodb_table,
                Item=item
            )
            
        except Exception as e:
            logger.error(f"Error storing optimization in DynamoDB: {e}")
    
    def _load_from_dynamodb(self, optimization_id: str) -> Optional[OptimizationRecord]:
        """Load optimization from DynamoDB"""
        try:
            response = self.dynamodb.get_item(
                TableName=self.dynamodb_table,
                Key={'optimization_id': {'S': optimization_id}}
            )
            
            if 'Item' not in response:
                return None
            
            item = response['Item']
            
            optimization = OptimizationRecord(
                optimization_id=item['optimization_id']['S'],
                resource_id=item['resource_id']['S'],
                resource_type=item['resource_type']['S'],
                optimization_type=OptimizationType(item['optimization_type']['S']),
                description=item['description']['S'],
                implemented_date=datetime.fromisoformat(item['implemented_date']['S']),
                projected_monthly_savings=float(item['projected_monthly_savings']['N']),
                status=SavingsStatus(item['status']['S']),
                confidence_score=float(item['confidence_score']['N']),
                rollback_available=item['rollback_available']['BOOL'],
                metadata=json.loads(item['metadata']['S'])
            )
            
            if 'actual_monthly_savings' in item:
                optimization.actual_monthly_savings = float(item['actual_monthly_savings']['N'])
            
            # Cache it
            self.optimization_cache[optimization_id] = optimization
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error loading optimization from DynamoDB: {e}")
            return None
    
    def _query_dynamodb_period(self,
                             start_date: datetime,
                             end_date: datetime) -> List[OptimizationRecord]:
        """Query DynamoDB for optimizations in a period"""
        # Implementation would use GSI on implemented_date
        # For now, return empty list
        return []
    
    def _generate_recommendations(self,
                                optimizations: List[OptimizationRecord],
                                summary: SavingsSummary) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check realization rate
        if summary.realization_rate < 0.7:
            recommendations.append(
                "Projection accuracy is below 70%. Consider reviewing estimation methodology."
            )
        
        # Check for failed optimizations
        if len(summary.failed_optimizations) > len(optimizations) * 0.2:
            recommendations.append(
                f"{len(summary.failed_optimizations)} optimizations failed. "
                "Investigate root causes and improve implementation process."
            )
        
        # Check optimization distribution
        if len(summary.by_type) < 3:
            recommendations.append(
                "Limited optimization diversity. Explore additional optimization types "
                "like Reserved Instances, Savings Plans, or storage optimization."
            )
        
        # Service concentration
        top_service_savings = max(summary.by_service.values()) if summary.by_service else 0
        if top_service_savings > summary.total_realized_savings * 0.5:
            recommendations.append(
                "Over 50% of savings from a single service. "
                "Diversify optimization efforts across services."
            )
        
        # Success by type
        low_success_types = []
        for opt in optimizations:
            if opt.status == SavingsStatus.FAILED:
                low_success_types.append(opt.optimization_type.value)
        
        if low_success_types:
            most_failed = max(set(low_success_types), key=low_success_types.count)
            recommendations.append(
                f"{most_failed} optimizations have high failure rate. "
                "Review and improve this optimization type."
            )
        
        return recommendations
    
    def export_detailed_report(self,
                             start_date: datetime,
                             end_date: datetime,
                             format: str = 'excel') -> str:
        """Export detailed savings report"""
        import pandas as pd
        
        # Get all optimizations
        optimizations = self._get_optimizations_in_period(start_date, end_date)
        
        # Convert to DataFrame
        data = []
        for opt in optimizations:
            data.append({
                'Optimization ID': opt.optimization_id,
                'Resource ID': opt.resource_id,
                'Resource Type': opt.resource_type,
                'Optimization Type': opt.optimization_type.value,
                'Description': opt.description,
                'Implemented Date': opt.implemented_date,
                'Projected Monthly Savings': opt.projected_monthly_savings,
                'Actual Monthly Savings': opt.actual_monthly_savings or 0,
                'Status': opt.status.value,
                'Confidence Score': opt.confidence_score,
                'Realization Rate': (
                    opt.actual_monthly_savings / opt.projected_monthly_savings
                    if opt.actual_monthly_savings and opt.projected_monthly_savings > 0
                    else 0
                ),
                'Service': opt.metadata.get('service', 'Unknown')
            })
        
        df = pd.DataFrame(data)
        
        # Generate filename
        filename = f"savings_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        if format == 'excel':
            output_file = f"{filename}.xlsx"
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='Optimizations', index=False)
                
                # Summary sheet
                summary = self.calculate_savings_metrics(start_date, end_date)
                summary_df = pd.DataFrame([{
                    'Total Projected Savings': summary.total_projected_savings,
                    'Total Realized Savings': summary.total_realized_savings,
                    'Realization Rate': summary.realization_rate,
                    'Number of Optimizations': len(optimizations)
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # By type
                type_df = pd.DataFrame([
                    {'Type': k.value, 'Savings': v}
                    for k, v in summary.by_type.items()
                ])
                type_df.to_excel(writer, sheet_name='By Type', index=False)
                
        elif format == 'csv':
            output_file = f"{filename}.csv"
            df.to_csv(output_file, index=False)
        
        else:
            output_file = f"{filename}.json"
            df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Detailed report exported to {output_file}")
        return output_file