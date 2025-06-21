# AWS Cost Optimizer - API Reference

## Table of Contents

1. [EC2 Optimizer](#ec2-optimizer)
2. [Network Optimizer](#network-optimizer)
3. [RDS Optimizer](#rds-optimizer)
4. [S3 Optimizer](#s3-optimizer)
5. [Reserved Instance Analyzer](#reserved-instance-analyzer)
6. [Cost Anomaly Detector](#cost-anomaly-detector)
7. [Auto-Remediation Engine](#auto-remediation-engine)
8. [Orchestrator](#orchestrator)

---

## EC2 Optimizer

### Class: `EC2Optimizer`

The EC2 Optimizer analyzes EC2 instances for cost optimization opportunities.

#### Constructor

```python
EC2Optimizer(
    cpu_threshold: float = 10.0,
    memory_threshold: float = 20.0,
    network_threshold: float = 5.0,
    observation_days: int = 14,
    session: Optional[boto3.Session] = None
)
```

**Parameters:**
- `cpu_threshold` (float): CPU utilization percentage threshold for idle detection
- `memory_threshold` (float): Memory utilization percentage threshold for rightsizing
- `network_threshold` (float): Network I/O threshold in MB for idle detection
- `observation_days` (int): Number of days of metrics to analyze
- `session` (boto3.Session): Optional boto3 session

#### Methods

##### `analyze_all_instances(regions: List[str] = None) -> List[EC2OptimizationRecommendation]`

Analyzes all EC2 instances across specified regions.

**Parameters:**
- `regions` (List[str]): List of AWS regions to analyze. If None, analyzes all regions.

**Returns:**
- List of `EC2OptimizationRecommendation` objects

**Example:**
```python
optimizer = EC2Optimizer()
recommendations = optimizer.analyze_all_instances(['us-east-1', 'us-west-2'])
```

---

## Network Optimizer

### Class: `NetworkOptimizer`

Optimizes AWS networking costs including NAT Gateways, Elastic IPs, and data transfer.

#### Constructor

```python
NetworkOptimizer(session: Optional[boto3.Session] = None)
```

#### Methods

##### `analyze_all_network_costs(regions: List[str] = None) -> List[NetworkOptimizationRecommendation]`

Analyzes network resources for optimization opportunities.

**Example:**
```python
optimizer = NetworkOptimizer()
recommendations = optimizer.analyze_all_network_costs()
```

---

## RDS Optimizer

### Class: `RDSOptimizer`

Identifies RDS optimization opportunities including idle databases and rightsizing.

#### Constructor

```python
RDSOptimizer(
    connection_threshold: int = 7,
    cpu_threshold: float = 25.0,
    observation_days: int = 60,
    session: Optional[boto3.Session] = None
)
```

#### Methods

##### `analyze_all_databases(region_name: str = None) -> List[RDSOptimizationRecommendation]`

Analyzes all RDS instances for optimization.

---

## S3 Optimizer

### Class: `S3Optimizer`

Optimizes S3 storage through intelligent tiering and lifecycle policies.

#### Constructor

```python
S3Optimizer(
    size_threshold_gb: float = 1024,
    observation_days: int = 90,
    session: Optional[boto3.Session] = None
)
```

#### Methods

##### `analyze_all_buckets(region_name: str = None) -> List[S3OptimizationRecommendation]`

Analyzes all S3 buckets for optimization opportunities.

---

## Reserved Instance Analyzer

### Class: `ReservedInstanceAnalyzer`

Analyzes usage patterns and recommends optimal RI/SP purchases.

#### Constructor

```python
ReservedInstanceAnalyzer(
    lookback_days: int = 90,
    forecast_days: int = 365,
    minimum_savings_threshold: float = 100,
    session: Optional[boto3.Session] = None
)
```

#### Methods

##### `analyze_all_opportunities() -> Dict[str, Any]`

Analyzes all RI and Savings Plan opportunities.

**Returns:**
```python
{
    'ri_recommendations': List[RIRecommendation],
    'sp_recommendations': List[SavingsPlanRecommendation],
    'total_monthly_savings': float,
    'total_annual_savings': float,
    'purchase_strategy': Dict[str, Any]
}
```

---

## Cost Anomaly Detector

### Class: `CostAnomalyDetector`

Detects unusual AWS spending patterns using statistical analysis and ML.

#### Constructor

```python
CostAnomalyDetector(
    lookback_days: int = 90,
    anomaly_threshold: float = 2.5,
    min_daily_spend: float = 10,
    session: Optional[boto3.Session] = None
)
```

#### Methods

##### `detect_anomalies(real_time: bool = True, services_filter: List[str] = None) -> List[CostAnomaly]`

Detects cost anomalies across AWS services.

##### `send_alerts(anomalies: List[CostAnomaly], sns_topic_arn: str = None)`

Sends SNS alerts for detected anomalies.

---

## Auto-Remediation Engine

### Class: `AutoRemediationEngine`

Automatically executes optimization recommendations with safety controls.

#### Constructor

```python
AutoRemediationEngine(
    policy: RemediationPolicy,
    dry_run: bool = False,
    session: Optional[boto3.Session] = None
)
```

#### Methods

##### `create_remediation_task(...) -> RemediationTask`

Creates a new remediation task.

##### `execute_approved_tasks(max_concurrent: int = 5) -> Dict[str, Any]`

Executes all approved remediation tasks.

---

## Orchestrator

### Class: `CostOptimizationOrchestrator`

Main orchestrator that coordinates all optimization components.

#### Constructor

```python
CostOptimizationOrchestrator(
    session: Optional[boto3.Session] = None,
    config: Dict[str, Any] = None
)
```

#### Methods

##### `run_full_optimization(regions: List[str] = None, services: List[str] = None) -> OptimizationResult`

Runs comprehensive cost optimization analysis.

**Returns:**
```python
OptimizationResult(
    timestamp: datetime,
    total_monthly_savings: float,
    total_annual_savings: float,
    ec2_savings: float,
    network_savings: float,
    ri_savings: float,
    anomalies_detected: int,
    recommendations_count: int,
    auto_remediation_tasks: int,
    execution_time: float,
    details: Dict[str, Any]
)
```

**Example:**
```python
orchestrator = CostOptimizationOrchestrator(config=config)
result = orchestrator.run_full_optimization(
    regions=['us-east-1', 'us-west-2'],
    services=['EC2', 'RDS', 'S3']
)
print(f"Total monthly savings: ${result.total_monthly_savings:,.2f}")
```

---

## Data Classes

### EC2OptimizationRecommendation

```python
@dataclass
class EC2OptimizationRecommendation:
    instance_id: str
    instance_type: str
    region: str
    action: str  # 'stop', 'terminate', 'rightsize', 'schedule', 'migrate_to_spot'
    current_monthly_cost: float
    recommended_monthly_cost: float
    monthly_savings: float
    annual_savings: float
    confidence: float
    reason: str
    risk_level: str
    implementation_steps: List[str]
    rollback_plan: str
    tags: Dict[str, str]
```

### CostAnomaly

```python
@dataclass
class CostAnomaly:
    anomaly_id: str
    detection_date: datetime
    service: str
    region: Optional[str]
    anomaly_type: str
    severity: str
    current_daily_cost: float
    expected_daily_cost: float
    cost_impact: float
    percentage_increase: float
    confidence_score: float
    probable_causes: List[str]
    affected_resources: List[str]
    recommended_actions: List[str]
    alert_sent: bool = False
```

---

## Error Handling

All methods may raise the following exceptions:

- `ClientError`: AWS API errors
- `ValueError`: Invalid parameter values
- `ConnectionError`: Network connectivity issues

Always wrap API calls in try-except blocks:

```python
try:
    recommendations = optimizer.analyze_all_instances()
except ClientError as e:
    logger.error(f"AWS API error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```