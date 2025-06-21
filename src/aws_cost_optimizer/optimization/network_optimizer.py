"""
Network Optimizer module for reducing AWS networking costs
Targets: NAT Gateways, Data Transfer, VPC Endpoints, Elastic IPs
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import json

logger = logging.getLogger(__name__)

@dataclass
class NetworkOptimizationRecommendation:
    """Represents a network optimization recommendation"""
    resource_id: str
    resource_type: str  # 'nat_gateway', 'vpc_endpoint', 'elastic_ip', 'data_transfer'
    current_cost: float
    recommended_action: str
    estimated_monthly_savings: float
    confidence: float
    reason: str
    risk_level: str
    implementation_steps: List[str]
    affected_resources: List[str]
    region: str

@dataclass
class DataTransferPattern:
    """Represents data transfer patterns between resources"""
    source_type: str  # 'ec2', 's3', 'rds', etc.
    destination_type: str
    source_region: str
    destination_region: str
    transfer_type: str  # 'inter-az', 'inter-region', 'internet'
    monthly_gb: float
    monthly_cost: float
    optimization_potential: str

class NetworkOptimizer:
    """Comprehensive network cost optimization engine"""
    
    # Network pricing (simplified - should load from pricing API)
    NETWORK_PRICING = {
        'nat_gateway_hourly': 0.045,  # per hour
        'nat_gateway_data_gb': 0.045,  # per GB
        'elastic_ip_unused': 0.005,    # per hour when not attached
        'data_transfer_inter_az': 0.01,  # per GB
        'data_transfer_inter_region': 0.02,  # per GB  
        'data_transfer_internet': 0.09,  # per GB (first 10TB)
        'vpc_endpoint_hourly': 0.01,     # per hour
        'vpc_endpoint_data_gb': 0.01     # per GB
    }
    
    def __init__(self, session: Optional[boto3.Session] = None):
        """
        Initialize Network Optimizer
        
        Args:
            session: Boto3 session (optional)
        """
        self.session = session or boto3.Session()
        self.ec2 = self.session.client('ec2')
        self.cloudwatch = self.session.client('cloudwatch')
        self.ce = self.session.client('ce')  # Cost Explorer
        self.vpc_endpoints = self.session.client('ec2')
        
    def analyze_all_network_costs(self, regions: List[str] = None) -> List[NetworkOptimizationRecommendation]:
        """
        Analyze all network-related costs across regions
        
        Args:
            regions: List of regions to analyze (uses all if None)
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not regions:
            regions = self._get_all_regions()
            
        logger.info(f"Analyzing network costs across {len(regions)} regions")
        
        # Analyze each component in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for region in regions:
                # NAT Gateway analysis
                futures.append(
                    executor.submit(self._analyze_nat_gateways, region)
                )
                # Elastic IP analysis
                futures.append(
                    executor.submit(self._analyze_elastic_ips, region)
                )
                # VPC Endpoint opportunities
                futures.append(
                    executor.submit(self._analyze_vpc_endpoint_opportunities, region)
                )
                # Data transfer patterns
                futures.append(
                    executor.submit(self._analyze_data_transfer_patterns, region)
                )
            
            for future in as_completed(futures):
                try:
                    region_recommendations = future.result()
                    recommendations.extend(region_recommendations)
                except Exception as e:
                    logger.error(f"Failed to analyze region: {e}")
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x.estimated_monthly_savings, reverse=True)
        
        return recommendations
    
    def _get_all_regions(self) -> List[str]:
        """Get all enabled regions"""
        try:
            response = self.ec2.describe_regions()
            return [r['RegionName'] for r in response['Regions']]
        except Exception as e:
            logger.error(f"Failed to get regions: {e}")
            return ['us-east-1', 'us-west-2', 'eu-west-1']  # Fallback
    
    def _analyze_nat_gateways(self, region: str) -> List[NetworkOptimizationRecommendation]:
        """Analyze NAT Gateway usage and costs"""
        recommendations = []
        
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            # Get all NAT Gateways
            response = ec2_regional.describe_nat_gateways(
                Filter=[{'Name': 'state', 'Values': ['available']}]
            )
            
            nat_gateways = response.get('NatGateways', [])
            logger.info(f"Found {len(nat_gateways)} NAT Gateways in {region}")
            
            for nat_gw in nat_gateways:
                nat_id = nat_gw['NatGatewayId']
                
                # Get metrics
                metrics = self._get_nat_gateway_metrics(nat_id, region)
                
                # Check for optimization opportunities
                recommendations.extend(self._check_nat_gateway_optimizations(
                    nat_gw, metrics, region
                ))
                
        except Exception as e:
            logger.error(f"Failed to analyze NAT Gateways in {region}: {e}")
            
        return recommendations
    
    def _get_nat_gateway_metrics(self, nat_id: str, region: str) -> Dict[str, Any]:
        """Get CloudWatch metrics for NAT Gateway"""
        metrics = {}
        
        try:
            cw_regional = self.session.client('cloudwatch', region_name=region)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)
            
            # Get bytes out (data processed)
            bytes_out = cw_regional.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='BytesOutToDestination',
                Dimensions=[{'Name': 'NatGatewayId', 'Value': nat_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # Daily
                Statistics=['Sum']
            )
            
            if bytes_out['Datapoints']:
                total_bytes = sum(dp['Sum'] for dp in bytes_out['Datapoints'])
                metrics['monthly_gb'] = total_bytes / 1e9
                metrics['daily_avg_gb'] = metrics['monthly_gb'] / 30
            else:
                metrics['monthly_gb'] = 0
                metrics['daily_avg_gb'] = 0
            
            # Get active connection count
            active_connections = cw_regional.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='ActiveConnectionCount',
                Dimensions=[{'Name': 'NatGatewayId', 'Value': nat_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # Hourly
                Statistics=['Average', 'Maximum']
            )
            
            if active_connections['Datapoints']:
                metrics['avg_connections'] = np.mean([dp['Average'] for dp in active_connections['Datapoints']])
                metrics['max_connections'] = max(dp['Maximum'] for dp in active_connections['Datapoints'])
            else:
                metrics['avg_connections'] = 0
                metrics['max_connections'] = 0
                
        except Exception as e:
            logger.error(f"Failed to get NAT Gateway metrics: {e}")
            metrics = {'monthly_gb': 0, 'daily_avg_gb': 0, 'avg_connections': 0, 'max_connections': 0}
            
        return metrics
    
    def _check_nat_gateway_optimizations(self, nat_gw: Dict[str, Any], 
                                       metrics: Dict[str, Any], 
                                       region: str) -> List[NetworkOptimizationRecommendation]:
        """Check for NAT Gateway optimization opportunities"""
        recommendations = []
        nat_id = nat_gw['NatGatewayId']
        
        # Calculate current cost
        hourly_cost = self.NETWORK_PRICING['nat_gateway_hourly']
        data_cost = metrics['monthly_gb'] * self.NETWORK_PRICING['nat_gateway_data_gb']
        monthly_cost = (hourly_cost * 24 * 30) + data_cost
        
        # 1. Check for unused NAT Gateways
        if metrics['monthly_gb'] < 10 and metrics['avg_connections'] < 10:
            recommendations.append(NetworkOptimizationRecommendation(
                resource_id=nat_id,
                resource_type='nat_gateway',
                current_cost=monthly_cost,
                recommended_action='remove_unused_nat_gateway',
                estimated_monthly_savings=monthly_cost,
                confidence=0.9,
                reason=f"NAT Gateway has minimal usage: {metrics['monthly_gb']:.1f}GB/month, {metrics['avg_connections']:.0f} avg connections",
                risk_level='medium',
                implementation_steps=[
                    f"1. Identify resources using NAT Gateway in subnet {nat_gw['SubnetId']}",
                    "2. Evaluate if resources need internet access",
                    "3. Consider alternatives: VPC endpoints, NAT instance, or direct internet gateway",
                    f"4. Update route tables to remove NAT Gateway",
                    f"5. Delete NAT Gateway: aws ec2 delete-nat-gateway --nat-gateway-id {nat_id} --region {region}"
                ],
                affected_resources=[nat_gw['SubnetId']],
                region=region
            ))
        
        # 2. Check for multiple NAT Gateways in same AZ (consolidation opportunity)
        elif self._check_nat_consolidation_opportunity(nat_gw, region):
            savings = monthly_cost * 0.5  # Assume 50% savings from consolidation
            recommendations.append(NetworkOptimizationRecommendation(
                resource_id=nat_id,
                resource_type='nat_gateway',
                current_cost=monthly_cost,
                recommended_action='consolidate_nat_gateways',
                estimated_monthly_savings=savings,
                confidence=0.7,
                reason="Multiple NAT Gateways in same VPC could be consolidated",
                risk_level='medium',
                implementation_steps=[
                    "1. Review high availability requirements",
                    "2. Identify NAT Gateways that can be consolidated",
                    "3. Update route tables to use single NAT Gateway per AZ",
                    "4. Monitor for performance impact",
                    "5. Delete redundant NAT Gateways"
                ],
                affected_resources=[nat_gw['VpcId']],
                region=region
            ))
        
        # 3. NAT Instance recommendation for low-traffic gateways
        if metrics['monthly_gb'] < 100 and metrics['monthly_gb'] > 10:
            # NAT instance could be cheaper for low traffic
            nat_instance_cost = 0.0104 * 24 * 30  # t3.micro cost
            savings = monthly_cost - nat_instance_cost
            
            if savings > 20:
                recommendations.append(NetworkOptimizationRecommendation(
                    resource_id=nat_id,
                    resource_type='nat_gateway',
                    current_cost=monthly_cost,
                    recommended_action='replace_with_nat_instance',
                    estimated_monthly_savings=savings,
                    confidence=0.6,
                    reason=f"Low traffic ({metrics['monthly_gb']:.0f}GB/month) suitable for NAT instance",
                    risk_level='high',
                    implementation_steps=[
                        "1. Launch t3.micro instance with NAT AMI",
                        "2. Configure source/destination check",
                        "3. Update route tables",
                        "4. Test thoroughly",
                        "5. Delete NAT Gateway after validation"
                    ],
                    affected_resources=[nat_gw['SubnetId']],
                    region=region
                ))
        
        return recommendations
    
    def _check_nat_consolidation_opportunity(self, nat_gw: Dict[str, Any], region: str) -> bool:
        """Check if NAT Gateways can be consolidated"""
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            vpc_id = nat_gw['VpcId']
            
            # Get all NAT Gateways in the same VPC
            response = ec2_regional.describe_nat_gateways(
                Filters=[
                    {'Name': 'vpc-id', 'Values': [vpc_id]},
                    {'Name': 'state', 'Values': ['available']}
                ]
            )
            
            # If more than 1 NAT Gateway per AZ, consolidation possible
            nat_gateways = response.get('NatGateways', [])
            az_count = {}
            
            for ng in nat_gateways:
                # Get subnet details to find AZ
                subnet_response = ec2_regional.describe_subnets(
                    SubnetIds=[ng['SubnetId']]
                )
                if subnet_response['Subnets']:
                    az = subnet_response['Subnets'][0]['AvailabilityZone']
                    az_count[az] = az_count.get(az, 0) + 1
            
            # Check if any AZ has multiple NAT Gateways
            return any(count > 1 for count in az_count.values())
            
        except Exception as e:
            logger.error(f"Failed to check NAT consolidation: {e}")
            return False
    
    def _analyze_elastic_ips(self, region: str) -> List[NetworkOptimizationRecommendation]:
        """Analyze Elastic IP usage and costs"""
        recommendations = []
        
        try:
            ec2_regional = self.session.client('ec2', region_name=region)
            
            # Get all Elastic IPs
            response = ec2_regional.describe_addresses()
            addresses = response.get('Addresses', [])
            
            for addr in addresses:
                if not addr.get('InstanceId') and not addr.get('NetworkInterfaceId'):
                    # Unattached Elastic IP
                    allocation_id = addr.get('AllocationId', addr.get('PublicIp'))
                    hourly_cost = self.NETWORK_PRICING['elastic_ip_unused']
                    monthly_cost = hourly_cost * 24 * 30
                    
                    # Get age of allocation
                    tags = addr.get('Tags', [])
                    tag_dict = {tag['Key']: tag['Value'] for tag in tags}
                    
                    recommendations.append(NetworkOptimizationRecommendation(
                        resource_id=allocation_id,
                        resource_type='elastic_ip',
                        current_cost=monthly_cost,
                        recommended_action='release_unused_elastic_ip',
                        estimated_monthly_savings=monthly_cost,
                        confidence=0.95,
                        reason=f"Elastic IP {addr['PublicIp']} is not attached to any resource",
                        risk_level='low',
                        implementation_steps=[
                            f"1. Verify IP is not needed: {addr['PublicIp']}",
                            "2. Check DNS records pointing to this IP",
                            "3. Document IP for future reference if needed",
                            f"4. Release: aws ec2 release-address --allocation-id {allocation_id} --region {region}"
                        ],
                        affected_resources=[addr['PublicIp']],
                        region=region
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to analyze Elastic IPs in {region}: {e}")
            
        return recommendations
    
    def _analyze_vpc_endpoint_opportunities(self, region: str) -> List[NetworkOptimizationRecommendation]:
        """Identify opportunities to use VPC endpoints instead of NAT Gateways"""
        recommendations = []
        
        try:
            # Get data transfer costs from Cost Explorer
            end_date = datetime.utcnow().strftime('%Y-%m-%d')
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            # Query for S3 and DynamoDB data transfer through NAT
            ce_response = self.ce.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
                ],
                Filter={
                    'And': [
                        {'Dimensions': {'Key': 'REGION', 'Values': [region]}},
                        {'Or': [
                            {'Dimensions': {'Key': 'USAGE_TYPE', 'Values': ['DataTransfer-Out-Bytes']}},
                            {'Dimensions': {'Key': 'USAGE_TYPE', 'Values': ['NatGateway-Bytes']}}
                        ]}
                    ]
                }
            )
            
            # Analyze if VPC endpoints would be cheaper
            monthly_transfer_cost = 0
            services_used = set()
            
            for result in ce_response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    usage_type = group['Keys'][1]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    
                    if 'S3' in service or 'DynamoDB' in service:
                        services_used.add(service)
                        monthly_transfer_cost += cost
            
            # If significant S3/DynamoDB traffic through NAT
            if monthly_transfer_cost > 50:
                vpc_endpoint_cost = self.NETWORK_PRICING['vpc_endpoint_hourly'] * 24 * 30 * len(services_used)
                savings = monthly_transfer_cost - vpc_endpoint_cost
                
                if savings > 20:
                    recommendations.append(NetworkOptimizationRecommendation(
                        resource_id=f"vpc-endpoints-{region}",
                        resource_type='vpc_endpoint',
                        current_cost=monthly_transfer_cost,
                        recommended_action='create_vpc_endpoints',
                        estimated_monthly_savings=savings,
                        confidence=0.8,
                        reason=f"High data transfer costs to AWS services: ${monthly_transfer_cost:.2f}/month",
                        risk_level='low',
                        implementation_steps=[
                            "1. Create Gateway VPC endpoints for S3 and DynamoDB (free)",
                            "2. Create Interface VPC endpoints for other AWS services",
                            "3. Update route tables to use VPC endpoints",
                            "4. Update security groups to allow endpoint traffic",
                            "5. Monitor data transfer costs reduction"
                        ],
                        affected_resources=list(services_used),
                        region=region
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to analyze VPC endpoint opportunities: {e}")
            
        return recommendations
    
    def _analyze_data_transfer_patterns(self, region: str) -> List[NetworkOptimizationRecommendation]:
        """Analyze data transfer patterns for optimization"""
        recommendations = []
        
        try:
            # Get VPC Flow Logs insights if available
            patterns = self._analyze_flow_logs(region)
            
            # Check for inter-AZ transfer
            if patterns.get('inter_az_gb', 0) > 1000:  # More than 1TB
                monthly_cost = patterns['inter_az_gb'] * self.NETWORK_PRICING['data_transfer_inter_az']
                
                recommendations.append(NetworkOptimizationRecommendation(
                    resource_id=f"inter-az-transfer-{region}",
                    resource_type='data_transfer',
                    current_cost=monthly_cost,
                    recommended_action='reduce_inter_az_transfer',
                    estimated_monthly_savings=monthly_cost * 0.5,  # Assume 50% reduction possible
                    confidence=0.7,
                    reason=f"High inter-AZ data transfer: {patterns['inter_az_gb']:.0f}GB/month",
                    risk_level='medium',
                    implementation_steps=[
                        "1. Identify top talkers using VPC Flow Logs",
                        "2. Co-locate frequently communicating resources in same AZ",
                        "3. Use placement groups for cluster compute",
                        "4. Consider using ElastiCache for frequently accessed data",
                        "5. Implement caching strategies"
                    ],
                    affected_resources=patterns.get('top_talkers', []),
                    region=region
                ))
            
            # Check for inter-region transfer
            if patterns.get('inter_region_gb', 0) > 500:
                monthly_cost = patterns['inter_region_gb'] * self.NETWORK_PRICING['data_transfer_inter_region']
                
                recommendations.append(NetworkOptimizationRecommendation(
                    resource_id=f"inter-region-transfer-{region}",
                    resource_type='data_transfer',
                    current_cost=monthly_cost,
                    recommended_action='optimize_inter_region_transfer',
                    estimated_monthly_savings=monthly_cost * 0.3,
                    confidence=0.6,
                    reason=f"High inter-region data transfer: {patterns['inter_region_gb']:.0f}GB/month",
                    risk_level='high',
                    implementation_steps=[
                        "1. Analyze if resources can be consolidated to single region",
                        "2. Use S3 Transfer Acceleration for large transfers",
                        "3. Consider AWS Direct Connect for predictable transfers",
                        "4. Implement regional caches",
                        "5. Use CloudFront for content distribution"
                    ],
                    affected_resources=patterns.get('regions_involved', []),
                    region=region
                ))
                
        except Exception as e:
            logger.error(f"Failed to analyze data transfer patterns: {e}")
            
        return recommendations
    
    def _analyze_flow_logs(self, region: str) -> Dict[str, Any]:
        """Analyze VPC Flow Logs for data transfer patterns"""
        # Simplified - in reality would query flow logs
        # This is a placeholder that returns estimated patterns
        return {
            'inter_az_gb': 1500,  # Placeholder
            'inter_region_gb': 300,  # Placeholder
            'top_talkers': ['i-1234567890abcdef0', 'i-0987654321fedcba0'],
            'regions_involved': ['us-east-1', 'us-west-2']
        }
    
    def calculate_nat_instance_sizing(self, bandwidth_mbps: float) -> Dict[str, Any]:
        """Calculate appropriate NAT instance sizing based on bandwidth needs"""
        # NAT instance bandwidth capabilities (approximate)
        instance_bandwidth = {
            't3.nano': 32,      # Mbps
            't3.micro': 64,
            't3.small': 128,
            't3.medium': 256,
            't3.large': 512,
            'm5.large': 1000,
            'm5.xlarge': 2000,
            'c5.large': 1000,
            'c5.xlarge': 2000,
            'c5n.large': 3000,
            'c5n.xlarge': 6000
        }
        
        # Instance costs per hour
        instance_costs = {
            't3.nano': 0.0052,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5n.large': 0.108,
            'c5n.xlarge': 0.216
        }
        
        # Find appropriate instance
        recommended_instance = None
        for instance_type, bandwidth in sorted(instance_bandwidth.items(), key=lambda x: x[1]):
            if bandwidth >= bandwidth_mbps * 1.2:  # 20% headroom
                recommended_instance = instance_type
                break
        
        if not recommended_instance:
            recommended_instance = 'c5n.xlarge'  # Maximum option
        
        monthly_cost = instance_costs[recommended_instance] * 24 * 30
        nat_gateway_cost = self.NETWORK_PRICING['nat_gateway_hourly'] * 24 * 30
        
        return {
            'recommended_instance': recommended_instance,
            'bandwidth_capacity': instance_bandwidth[recommended_instance],
            'monthly_cost': monthly_cost,
            'nat_gateway_cost': nat_gateway_cost,
            'monthly_savings': nat_gateway_cost - monthly_cost,
            'setup_complexity': 'medium' if recommended_instance.startswith('t3') else 'high'
        }
    
    def generate_optimization_report(self, recommendations: List[NetworkOptimizationRecommendation]) -> pd.DataFrame:
        """Generate detailed network optimization report"""
        data = []
        
        for rec in recommendations:
            data.append({
                'Resource ID': rec.resource_id,
                'Type': rec.resource_type,
                'Region': rec.region,
                'Current Cost': f"${rec.current_cost:.2f}",
                'Action': rec.recommended_action,
                'Monthly Savings': f"${rec.estimated_monthly_savings:.2f}",
                'Annual Savings': f"${rec.estimated_monthly_savings * 12:.2f}",
                'Confidence': f"{rec.confidence:.0%}",
                'Risk': rec.risk_level,
                'Reason': rec.reason
            })
        
        df = pd.DataFrame(data)
        
        # Summary by type
        summary = df.groupby('Type').agg({
            'Monthly Savings': lambda x: sum(float(s.replace('$', '').replace(',', '')) for s in x),
            'Resource ID': 'count'
        }).rename(columns={'Resource ID': 'Count'})
        
        total_monthly = sum(rec.estimated_monthly_savings for rec in recommendations)
        
        print(f"\nNetwork Optimization Summary:")
        print(f"Total Recommendations: {len(recommendations)}")
        print(f"Total Monthly Savings: ${total_monthly:,.2f}")
        print(f"Total Annual Savings: ${total_monthly * 12:,.2f}")
        print(f"\nSavings by Type:")
        print(summary)
        
        return df
    
    def export_recommendations(self,
                             recommendations: List[NetworkOptimizationRecommendation],
                             output_file: str = 'network_optimization_report.xlsx'):
        """Export recommendations to Excel"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self.generate_optimization_report(recommendations)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # NAT Gateway optimizations
            nat_recs = [r for r in recommendations if r.resource_type == 'nat_gateway']
            if nat_recs:
                nat_data = []
                for rec in nat_recs:
                    nat_data.append({
                        'NAT Gateway ID': rec.resource_id,
                        'Region': rec.region,
                        'Monthly Cost': f"${rec.current_cost:.2f}",
                        'Savings': f"${rec.estimated_monthly_savings:.2f}",
                        'Action': rec.recommended_action,
                        'Implementation': '\n'.join(rec.implementation_steps)
                    })
                pd.DataFrame(nat_data).to_excel(writer, sheet_name='NAT Gateways', index=False)
            
            # Data transfer optimizations
            transfer_recs = [r for r in recommendations if r.resource_type == 'data_transfer']
            if transfer_recs:
                transfer_data = []
                for rec in transfer_recs:
                    transfer_data.append({
                        'Transfer Type': rec.resource_id,
                        'Monthly Cost': f"${rec.current_cost:.2f}",
                        'Potential Savings': f"${rec.estimated_monthly_savings:.2f}",
                        'Strategy': '\n'.join(rec.implementation_steps[:3])
                    })
                pd.DataFrame(transfer_data).to_excel(writer, sheet_name='Data Transfer', index=False)
                
        logger.info(f"Exported network optimization report to {output_file}")