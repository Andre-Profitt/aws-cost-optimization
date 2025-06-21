"""
Reserved Instance Analyzer - Optimizes RI and Savings Plans purchases
Analyzes usage patterns and recommends optimal commitment strategies
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)

@dataclass
class RIRecommendation:
    """Represents a Reserved Instance recommendation"""
    instance_type: str
    platform: str
    region: str
    availability_zone: Optional[str]
    tenancy: str
    offering_class: str  # 'standard' or 'convertible'
    term_length: int  # 1 or 3 years
    payment_option: str  # 'all_upfront', 'partial_upfront', 'no_upfront'
    quantity: int
    monthly_on_demand_cost: float
    monthly_ri_cost: float
    monthly_savings: float
    annual_savings: float
    roi_months: float  # Return on investment period
    utilization_rate: float
    confidence_score: float
    break_even_months: float

@dataclass
class SavingsPlanRecommendation:
    """Represents a Savings Plan recommendation"""
    plan_type: str  # 'Compute', 'EC2Instance', 'SageMaker'
    term_length: int  # 1 or 3 years
    payment_option: str
    hourly_commitment: float
    monthly_on_demand_cost: float
    monthly_savings_plan_cost: float
    monthly_savings: float
    annual_savings: float
    coverage_percentage: float
    utilization_forecast: float
    affected_services: List[str]
    confidence_score: float

@dataclass
class UsagePattern:
    """Represents resource usage patterns"""
    resource_type: str
    instance_type: str
    region: str
    daily_hours: List[float]  # 24 hours of usage
    weekly_pattern: List[float]  # 7 days
    monthly_pattern: List[float]  # 30 days
    minimum_instances: int
    maximum_instances: int
    average_instances: float
    stability_score: float  # 0-1, how stable the usage is

class ReservedInstanceAnalyzer:
    """Analyzes usage patterns and recommends optimal RI/SP purchases"""
    
    def __init__(self, 
                 lookback_days: int = 90,
                 forecast_days: int = 365,
                 minimum_savings_threshold: float = 100,  # Min monthly savings
                 session: Optional[boto3.Session] = None):
        """
        Initialize RI Analyzer
        
        Args:
            lookback_days: Days of historical data to analyze
            forecast_days: Days to forecast for ROI calculation
            minimum_savings_threshold: Minimum monthly savings to recommend
            session: Boto3 session
        """
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.minimum_savings_threshold = minimum_savings_threshold
        self.session = session or boto3.Session()
        self.ce = self.session.client('ce')
        self.ec2 = self.session.client('ec2')
        self.pricing = self.session.client('pricing', region_name='us-east-1')
        
    def analyze_all_opportunities(self) -> Dict[str, Any]:
        """
        Analyze all RI and Savings Plan opportunities
        
        Returns:
            Dictionary containing recommendations and analysis
        """
        logger.info("Starting Reserved Instance and Savings Plan analysis...")
        
        results = {
            'ri_recommendations': [],
            'sp_recommendations': [],
            'usage_patterns': [],
            'total_monthly_savings': 0,
            'total_annual_savings': 0,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        # Get current RI and SP utilization
        current_utilization = self._get_current_utilization()
        results['current_utilization'] = current_utilization
        
        # Analyze usage patterns
        usage_patterns = self._analyze_usage_patterns()
        results['usage_patterns'] = usage_patterns
        
        # Generate RI recommendations
        ri_recommendations = self._generate_ri_recommendations(usage_patterns)
        results['ri_recommendations'] = ri_recommendations
        
        # Generate Savings Plan recommendations
        sp_recommendations = self._generate_sp_recommendations(usage_patterns)
        results['sp_recommendations'] = sp_recommendations
        
        # Calculate total savings
        total_monthly_ri = sum(r.monthly_savings for r in ri_recommendations)
        total_monthly_sp = sum(r.monthly_savings for r in sp_recommendations)
        
        results['total_monthly_savings'] = total_monthly_ri + total_monthly_sp
        results['total_annual_savings'] = results['total_monthly_savings'] * 12
        
        # Generate purchase strategy
        results['purchase_strategy'] = self._generate_purchase_strategy(
            ri_recommendations, sp_recommendations
        )
        
        return results
    
    def _get_current_utilization(self) -> Dict[str, Any]:
        """Get current RI and SP utilization"""
        utilization = {
            'reserved_instances': {},
            'savings_plans': {},
            'coverage': {}
        }
        
        try:
            # Get RI utilization
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=7)
            
            ri_utilization = self.ce.get_reservation_utilization(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'INSTANCE_TYPE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ]
            )
            
            utilization['reserved_instances'] = {
                'total_utilization': float(ri_utilization.get('Total', {}).get('UtilizationPercentage', '0')),
                'total_hours': float(ri_utilization.get('Total', {}).get('TotalReservedInstanceHours', '0')),
                'used_hours': float(ri_utilization.get('Total', {}).get('TotalUsedInstanceHours', '0')),
                'by_instance_type': {}
            }
            
            # Parse by instance type
            for group in ri_utilization.get('UtilizationsByTime', []):
                for detail in group.get('Groups', []):
                    instance_type = detail['Keys'][0]
                    region = detail['Keys'][1]
                    key = f"{instance_type}_{region}"
                    
                    utilization['reserved_instances']['by_instance_type'][key] = {
                        'utilization': float(detail['Utilization']['UtilizationPercentage']),
                        'hours': float(detail['Utilization']['TotalReservedInstanceHours']),
                        'used': float(detail['Utilization']['TotalUsedInstanceHours'])
                    }
            
            # Get Savings Plans utilization
            sp_utilization = self.ce.get_savings_plans_utilization(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                }
            )
            
            utilization['savings_plans'] = {
                'total_utilization': float(sp_utilization.get('Total', {}).get('UtilizationPercentage', '0')),
                'total_commitment': float(sp_utilization.get('Total', {}).get('TotalCommitment', '0')),
                'used_commitment': float(sp_utilization.get('Total', {}).get('UsedCommitment', '0'))
            }
            
            # Get coverage metrics
            coverage = self.ce.get_savings_plans_coverage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            utilization['coverage'] = {
                'average_coverage': float(coverage.get('SavingsPlansCoverages', [{}])[0].get('Coverage', {}).get('CoveragePercentage', '0')),
                'by_service': {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get current utilization: {e}")
            
        return utilization
    
    def _analyze_usage_patterns(self) -> List[UsagePattern]:
        """Analyze EC2 usage patterns"""
        patterns = []
        
        try:
            # Get instance usage data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Query Cost Explorer for detailed usage
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='HOURLY',
                Metrics=['UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'INSTANCE_TYPE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'},
                    {'Type': 'DIMENSION', 'Key': 'PLATFORM'}
                ],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                }
            )
            
            # Process usage data by instance type
            usage_by_type = {}
            
            for result in response.get('ResultsByTime', []):
                timestamp = pd.to_datetime(result['TimePeriod']['Start'])
                
                for group in result.get('Groups', []):
                    if len(group['Keys']) >= 3:
                        instance_type = group['Keys'][0]
                        region = group['Keys'][1]
                        platform = group['Keys'][2]
                        usage = float(group['Metrics']['UsageQuantity']['Amount'])
                        
                        key = f"{instance_type}|{region}|{platform}"
                        
                        if key not in usage_by_type:
                            usage_by_type[key] = {
                                'hourly_usage': [],
                                'timestamps': []
                            }
                        
                        usage_by_type[key]['hourly_usage'].append(usage)
                        usage_by_type[key]['timestamps'].append(timestamp)
            
            # Analyze patterns for each instance type
            for key, data in usage_by_type.items():
                instance_type, region, platform = key.split('|')
                
                if len(data['hourly_usage']) < 168:  # Need at least a week of data
                    continue
                
                # Convert to numpy array for analysis
                usage_array = np.array(data['hourly_usage'])
                
                # Skip if usage is too low
                if usage_array.mean() < 0.1:
                    continue
                
                # Calculate patterns
                pattern = self._calculate_usage_pattern(
                    usage_array, data['timestamps'], 
                    instance_type, region, platform
                )
                
                if pattern.stability_score > 0.3:  # Only stable patterns
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Failed to analyze usage patterns: {e}")
            
        return patterns
    
    def _calculate_usage_pattern(self, 
                                usage_array: np.ndarray,
                                timestamps: List[datetime],
                                instance_type: str,
                                region: str,
                                platform: str) -> UsagePattern:
        """Calculate detailed usage pattern from hourly data"""
        # Create DataFrame for easier analysis
        df = pd.DataFrame({
            'usage': usage_array,
            'timestamp': timestamps
        })
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Calculate daily patterns (average usage by hour)
        daily_pattern = df.groupby('hour')['usage'].mean().values
        
        # Calculate weekly patterns (average usage by day of week)
        weekly_pattern = df.groupby('day_of_week')['usage'].mean().values
        
        # Calculate monthly patterns (average usage by day of month)
        monthly_pattern = df.groupby('day_of_month')['usage'].mean().values
        
        # Calculate statistics
        minimum = usage_array.min()
        maximum = usage_array.max()
        average = usage_array.mean()
        std_dev = usage_array.std()
        
        # Calculate stability score (lower coefficient of variation = more stable)
        stability_score = 1 - (std_dev / (average + 0.001))
        stability_score = max(0, min(1, stability_score))
        
        return UsagePattern(
            resource_type='EC2',
            instance_type=instance_type,
            region=region,
            daily_hours=daily_pattern.tolist(),
            weekly_pattern=weekly_pattern.tolist(),
            monthly_pattern=monthly_pattern.tolist(),
            minimum_instances=int(minimum),
            maximum_instances=int(np.ceil(maximum)),
            average_instances=average,
            stability_score=stability_score
        )
    
    def _generate_ri_recommendations(self, 
                                   usage_patterns: List[UsagePattern]) -> List[RIRecommendation]:
        """Generate Reserved Instance recommendations"""
        recommendations = []
        
        # Group patterns by instance type and region
        grouped_patterns = {}
        for pattern in usage_patterns:
            key = f"{pattern.instance_type}|{pattern.region}"
            if key not in grouped_patterns:
                grouped_patterns[key] = []
            grouped_patterns[key].append(pattern)
        
        # Generate recommendations for each group
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for key, patterns in grouped_patterns.items():
                instance_type, region = key.split('|')
                
                # Use the most stable pattern
                pattern = max(patterns, key=lambda p: p.stability_score)
                
                # Only recommend RI for stable, consistent usage
                if pattern.stability_score > 0.6 and pattern.minimum_instances > 0:
                    futures.append(
                        executor.submit(
                            self._calculate_ri_recommendation,
                            pattern, instance_type, region
                        )
                    )
            
            for future in as_completed(futures):
                try:
                    rec = future.result()
                    if rec and rec.monthly_savings >= self.minimum_savings_threshold:
                        recommendations.append(rec)
                except Exception as e:
                    logger.error(f"Failed to calculate RI recommendation: {e}")
        
        # Sort by ROI
        recommendations.sort(key=lambda x: x.roi_months)
        
        return recommendations
    
    def _calculate_ri_recommendation(self,
                                   pattern: UsagePattern,
                                   instance_type: str,
                                   region: str) -> Optional[RIRecommendation]:
        """Calculate specific RI recommendation"""
        try:
            # Get pricing information
            on_demand_price = self._get_on_demand_price(instance_type, region)
            
            # Analyze different RI options
            best_recommendation = None
            best_savings = 0
            
            for term_length in [1, 3]:  # 1 year and 3 year terms
                for payment_option in ['all_upfront', 'partial_upfront', 'no_upfront']:
                    for offering_class in ['standard', 'convertible']:
                        
                        # Get RI price
                        ri_price = self._get_ri_price(
                            instance_type, region, term_length, 
                            payment_option, offering_class
                        )
                        
                        if not ri_price:
                            continue
                        
                        # Calculate savings based on minimum consistent usage
                        quantity = pattern.minimum_instances
                        
                        # Monthly costs
                        monthly_on_demand = on_demand_price * 730 * quantity
                        monthly_ri = ri_price['effective_hourly'] * 730 * quantity
                        monthly_savings = monthly_on_demand - monthly_ri
                        
                        # Calculate ROI
                        if ri_price['upfront'] > 0:
                            roi_months = ri_price['upfront'] / monthly_savings if monthly_savings > 0 else float('inf')
                        else:
                            roi_months = 0
                        
                        # Calculate break-even
                        break_even_months = roi_months
                        
                        # Utilization rate (how much of the RI will be used)
                        utilization_rate = min(1.0, pattern.average_instances / quantity)
                        
                        if monthly_savings > best_savings:
                            best_savings = monthly_savings
                            best_recommendation = RIRecommendation(
                                instance_type=instance_type,
                                platform='Linux/UNIX',  # Simplified
                                region=region,
                                availability_zone=None,  # Regional RI
                                tenancy='default',
                                offering_class=offering_class,
                                term_length=term_length,
                                payment_option=payment_option,
                                quantity=quantity,
                                monthly_on_demand_cost=monthly_on_demand,
                                monthly_ri_cost=monthly_ri,
                                monthly_savings=monthly_savings,
                                annual_savings=monthly_savings * 12,
                                roi_months=roi_months,
                                utilization_rate=utilization_rate,
                                confidence_score=pattern.stability_score,
                                break_even_months=break_even_months
                            )
            
            return best_recommendation
            
        except Exception as e:
            logger.error(f"Error calculating RI recommendation: {e}")
            return None
    
    def _get_on_demand_price(self, instance_type: str, region: str) -> float:
        """Get on-demand price for instance type"""
        try:
            # Use pricing API
            response = self.pricing.get_products(
                ServiceCode='AmazonEC2',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                on_demand = price_data.get('terms', {}).get('OnDemand', {})
                
                for term in on_demand.values():
                    for price_dimension in term.get('priceDimensions', {}).values():
                        return float(price_dimension['pricePerUnit']['USD'])
            
            # Fallback to estimate
            return self._estimate_on_demand_price(instance_type)
            
        except Exception as e:
            logger.error(f"Failed to get on-demand price: {e}")
            return self._estimate_on_demand_price(instance_type)
    
    def _get_ri_price(self,
                     instance_type: str,
                     region: str,
                     term_length: int,
                     payment_option: str,
                     offering_class: str) -> Optional[Dict[str, float]]:
        """Get RI pricing information"""
        try:
            # This would query the actual RI pricing API
            # Simplified estimation for now
            on_demand = self._get_on_demand_price(instance_type, region)
            
            # Discount factors (approximate)
            discounts = {
                1: {'standard': 0.28, 'convertible': 0.20},
                3: {'standard': 0.48, 'convertible': 0.36}
            }
            
            discount = discounts[term_length][offering_class]
            
            # Payment option factors
            payment_factors = {
                'all_upfront': {'upfront': 1.0, 'hourly': 0},
                'partial_upfront': {'upfront': 0.5, 'hourly': 0.5},
                'no_upfront': {'upfront': 0, 'hourly': 1.0}
            }
            
            effective_hourly = on_demand * (1 - discount)
            hours_in_term = term_length * 365 * 24
            total_cost = effective_hourly * hours_in_term
            
            factor = payment_factors[payment_option]
            upfront_cost = total_cost * factor['upfront']
            hourly_cost = (total_cost * factor['hourly']) / hours_in_term
            
            return {
                'upfront': upfront_cost,
                'hourly': hourly_cost,
                'effective_hourly': effective_hourly
            }
            
        except Exception as e:
            logger.error(f"Failed to get RI price: {e}")
            return None
    
    def _generate_sp_recommendations(self,
                                   usage_patterns: List[UsagePattern]) -> List[SavingsPlanRecommendation]:
        """Generate Savings Plan recommendations"""
        recommendations = []
        
        try:
            # Calculate total compute spend
            total_hourly_spend = 0
            services_covered = set()
            
            for pattern in usage_patterns:
                on_demand_price = self._get_on_demand_price(
                    pattern.instance_type, pattern.region
                )
                total_hourly_spend += on_demand_price * pattern.average_instances
                services_covered.add('EC2')
            
            # Only recommend if significant spend
            if total_hourly_spend * 730 < self.minimum_savings_threshold:
                return recommendations
            
            # Analyze different commitment levels
            commitment_levels = [0.5, 0.7, 0.9]  # 50%, 70%, 90% of average usage
            
            for commitment_pct in commitment_levels:
                hourly_commitment = total_hourly_spend * commitment_pct
                
                for term_length in [1, 3]:
                    for payment_option in ['all_upfront', 'partial_upfront', 'no_upfront']:
                        
                        # Calculate savings (approximate 20-30% discount)
                        discount = 0.20 if term_length == 1 else 0.30
                        
                        monthly_on_demand = total_hourly_spend * 730
                        monthly_sp_cost = hourly_commitment * 730 * (1 - discount)
                        monthly_savings = (hourly_commitment * 730) - monthly_sp_cost
                        
                        # Coverage percentage
                        coverage_pct = commitment_pct * 100
                        
                        # Utilization forecast (how much of SP will be used)
                        avg_stability = np.mean([p.stability_score for p in usage_patterns])
                        utilization_forecast = min(0.95, avg_stability * 1.1)
                        
                        recommendation = SavingsPlanRecommendation(
                            plan_type='Compute',
                            term_length=term_length,
                            payment_option=payment_option,
                            hourly_commitment=hourly_commitment,
                            monthly_on_demand_cost=monthly_on_demand,
                            monthly_savings_plan_cost=monthly_sp_cost,
                            monthly_savings=monthly_savings,
                            annual_savings=monthly_savings * 12,
                            coverage_percentage=coverage_pct,
                            utilization_forecast=utilization_forecast,
                            affected_services=list(services_covered),
                            confidence_score=avg_stability
                        )
                        
                        if monthly_savings >= self.minimum_savings_threshold:
                            recommendations.append(recommendation)
            
            # Sort by savings
            recommendations.sort(key=lambda x: x.monthly_savings, reverse=True)
            
            # Return top 3 options
            return recommendations[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate SP recommendations: {e}")
            return recommendations
    
    def _generate_purchase_strategy(self,
                                  ri_recommendations: List[RIRecommendation],
                                  sp_recommendations: List[SavingsPlanRecommendation]) -> Dict[str, Any]:
        """Generate optimal purchase strategy combining RIs and SPs"""
        strategy = {
            'approach': '',
            'total_upfront_investment': 0,
            'monthly_savings': 0,
            'annual_savings': 0,
            'roi_months': 0,
            'recommendations': []
        }
        
        # Analyze options
        total_ri_savings = sum(r.monthly_savings for r in ri_recommendations)
        total_sp_savings = sum(r.monthly_savings for r in sp_recommendations) if sp_recommendations else 0
        
        # Determine strategy
        if total_ri_savings > total_sp_savings * 1.5:
            # RI-focused strategy
            strategy['approach'] = 'RI-focused'
            strategy['recommendations'] = [
                {
                    'type': 'reserved_instance',
                    'action': f'Purchase {len(ri_recommendations)} Reserved Instances',
                    'details': ri_recommendations[:5]  # Top 5
                }
            ]
            strategy['monthly_savings'] = total_ri_savings
            
        elif total_sp_savings > total_ri_savings * 1.5:
            # SP-focused strategy
            strategy['approach'] = 'SavingsPlan-focused'
            strategy['recommendations'] = [
                {
                    'type': 'savings_plan',
                    'action': 'Purchase Compute Savings Plan',
                    'details': sp_recommendations[0] if sp_recommendations else None
                }
            ]
            strategy['monthly_savings'] = total_sp_savings
            
        else:
            # Hybrid strategy
            strategy['approach'] = 'Hybrid'
            
            # Use RIs for most stable workloads
            stable_ris = [r for r in ri_recommendations if r.confidence_score > 0.8][:3]
            
            # Use SP for remaining coverage
            if sp_recommendations:
                strategy['recommendations'] = [
                    {
                        'type': 'reserved_instance',
                        'action': f'Purchase {len(stable_ris)} RIs for stable workloads',
                        'details': stable_ris
                    },
                    {
                        'type': 'savings_plan',
                        'action': 'Purchase Savings Plan for flexible coverage',
                        'details': sp_recommendations[0]
                    }
                ]
            
            strategy['monthly_savings'] = sum(r.monthly_savings for r in stable_ris)
            if sp_recommendations:
                strategy['monthly_savings'] += sp_recommendations[0].monthly_savings * 0.5
        
        strategy['annual_savings'] = strategy['monthly_savings'] * 12
        
        return strategy
    
    def _region_to_location(self, region: str) -> str:
        """Convert region code to location name for pricing API"""
        region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'EU (Ireland)',
            'eu-central-1': 'EU (Frankfurt)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)'
        }
        return region_map.get(region, region)
    
    def _estimate_on_demand_price(self, instance_type: str) -> float:
        """Estimate on-demand price based on instance type"""
        # Simplified pricing estimates
        base_prices = {
            't3.nano': 0.0052,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'r5.large': 0.126,
            'r5.xlarge': 0.252
        }
        
        return base_prices.get(instance_type, 0.1)  # Default to $0.1/hour
    
    def export_analysis(self,
                       results: Dict[str, Any],
                       output_file: str = 'ri_analysis_report.xlsx'):
        """Export RI/SP analysis to Excel"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Monthly Savings',
                    'Total Annual Savings',
                    'Current RI Utilization',
                    'Current SP Utilization',
                    'Number of RI Recommendations',
                    'Number of SP Recommendations',
                    'Recommended Strategy'
                ],
                'Value': [
                    f"${results['total_monthly_savings']:,.2f}",
                    f"${results['total_annual_savings']:,.2f}",
                    f"{results['current_utilization']['reserved_instances']['total_utilization']:.1f}%",
                    f"{results['current_utilization']['savings_plans']['total_utilization']:.1f}%",
                    len(results['ri_recommendations']),
                    len(results['sp_recommendations']),
                    results['purchase_strategy']['approach']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # RI Recommendations
            if results['ri_recommendations']:
                ri_data = []
                for rec in results['ri_recommendations']:
                    ri_data.append({
                        'Instance Type': rec.instance_type,
                        'Region': rec.region,
                        'Quantity': rec.quantity,
                        'Term': f"{rec.term_length} Year",
                        'Payment': rec.payment_option.replace('_', ' ').title(),
                        'Class': rec.offering_class.title(),
                        'Monthly Savings': f"${rec.monthly_savings:.2f}",
                        'Annual Savings': f"${rec.annual_savings:.2f}",
                        'ROI (months)': f"{rec.roi_months:.1f}",
                        'Confidence': f"{rec.confidence_score:.0%}"
                    })
                pd.DataFrame(ri_data).to_excel(writer, sheet_name='RI Recommendations', index=False)
            
            # SP Recommendations
            if results['sp_recommendations']:
                sp_data = []
                for rec in results['sp_recommendations']:
                    sp_data.append({
                        'Plan Type': rec.plan_type,
                        'Term': f"{rec.term_length} Year",
                        'Payment': rec.payment_option.replace('_', ' ').title(),
                        'Hourly Commitment': f"${rec.hourly_commitment:.2f}",
                        'Monthly Savings': f"${rec.monthly_savings:.2f}",
                        'Annual Savings': f"${rec.annual_savings:.2f}",
                        'Coverage': f"{rec.coverage_percentage:.0f}%",
                        'Services': ', '.join(rec.affected_services)
                    })
                pd.DataFrame(sp_data).to_excel(writer, sheet_name='SP Recommendations', index=False)
        
        logger.info(f"Exported RI/SP analysis to {output_file}")