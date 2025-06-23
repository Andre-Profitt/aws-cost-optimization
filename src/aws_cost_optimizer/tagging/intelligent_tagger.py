"""
Intelligent Auto-Tagging System

Uses ML and pattern recognition to automatically suggest and apply tags
based on resource usage patterns, relationships, and naming conventions.
"""

import re
import json
import boto3
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class TagCategory(Enum):
    """Categories of tags"""
    ENVIRONMENT = "environment"
    APPLICATION = "application"
    TEAM = "team"
    COST_CENTER = "cost_center"
    PROJECT = "project"
    COMPLIANCE = "compliance"
    LIFECYCLE = "lifecycle"
    AUTOMATION = "automation"


@dataclass
class TagSuggestion:
    """Represents a tag suggestion"""
    key: str
    value: str
    confidence: float
    category: TagCategory
    reasoning: str
    based_on: List[str] = field(default_factory=list)  # What the suggestion is based on
    
    
@dataclass
class TaggingRule:
    """Defines an automated tagging rule"""
    rule_id: str
    name: str
    condition_type: str  # 'pattern', 'relationship', 'usage', 'ml'
    condition: Dict[str, Any]
    tags_to_apply: Dict[str, str]
    priority: int = 0
    enabled: bool = True
    
    
@dataclass
class TagComplianceResult:
    """Result of tag compliance check"""
    resource_id: str
    resource_type: str
    compliant: bool
    missing_required: List[str]
    invalid_tags: List[Tuple[str, str]]  # (key, reason)
    suggestions: List[TagSuggestion]
    compliance_score: float  # 0-1


class IntelligentTagger:
    """Automatically suggest and apply tags based on patterns"""
    
    # Common environment patterns
    ENV_PATTERNS = {
        'production': [r'prod', r'prd', r'production', r'live'],
        'staging': [r'stag', r'stg', r'staging', r'stage', r'uat'],
        'development': [r'dev', r'develop', r'development', r'test'],
        'qa': [r'qa', r'test', r'testing'],
        'sandbox': [r'sandbox', r'sbx', r'poc', r'demo']
    }
    
    # Common application patterns
    APP_PATTERNS = {
        'web': [r'web', r'www', r'nginx', r'apache', r'frontend'],
        'api': [r'api', r'rest', r'graphql', r'backend'],
        'database': [r'db', r'database', r'mysql', r'postgres', r'mongo', r'redis'],
        'cache': [r'cache', r'redis', r'memcache', r'elasticache'],
        'queue': [r'queue', r'sqs', r'rabbit', r'kafka'],
        'analytics': [r'analytics', r'etl', r'data', r'warehouse', r'spark']
    }
    
    def __init__(self,
                 required_tags: List[str] = None,
                 tag_policies: Dict[str, Dict[str, Any]] = None,
                 tagging_rules: List[TaggingRule] = None,
                 ec2_client=None,
                 organizations_client=None):
        """
        Initialize intelligent tagger
        
        Args:
            required_tags: List of required tag keys
            tag_policies: Dictionary of tag policies
            tagging_rules: List of automated tagging rules
            ec2_client: Boto3 EC2 client
            organizations_client: Boto3 Organizations client
        """
        self.required_tags = required_tags or ['Environment', 'Owner', 'CostCenter', 'Application']
        self.tag_policies = tag_policies or {}
        self.tagging_rules = {r.rule_id: r for r in (tagging_rules or [])}
        
        # AWS clients
        self.ec2 = ec2_client or boto3.client('ec2')
        self.organizations = organizations_client or boto3.client('organizations')
        
        # ML models
        self.tag_classifier = None
        self.name_vectorizer = None
        self.relationship_model = None
        
        # Tag statistics
        self.tag_frequency = defaultdict(Counter)
        self.tag_relationships = defaultdict(set)
        
    def analyze_resource(self, 
                        resource_id: str,
                        resource_type: str,
                        resource_name: Optional[str] = None,
                        current_tags: Dict[str, str] = None,
                        usage_metrics: Dict[str, Any] = None) -> List[TagSuggestion]:
        """
        Analyze a resource and suggest tags
        
        Args:
            resource_id: Resource identifier
            resource_type: Type of resource
            resource_name: Resource name or identifier
            current_tags: Current tags on the resource
            usage_metrics: Usage metrics for ML-based suggestions
            
        Returns:
            List of tag suggestions
        """
        suggestions = []
        current_tags = current_tags or {}
        
        # 1. Name-based suggestions
        if resource_name:
            name_suggestions = self._suggest_from_name(resource_name, resource_type)
            suggestions.extend(name_suggestions)
        
        # 2. Relationship-based suggestions
        relationship_suggestions = self._suggest_from_relationships(
            resource_id, resource_type, current_tags
        )
        suggestions.extend(relationship_suggestions)
        
        # 3. Usage pattern suggestions
        if usage_metrics:
            usage_suggestions = self._suggest_from_usage(
                resource_id, resource_type, usage_metrics
            )
            suggestions.extend(usage_suggestions)
        
        # 4. ML-based suggestions
        if self.tag_classifier:
            ml_suggestions = self._suggest_from_ml(
                resource_id, resource_type, resource_name, usage_metrics
            )
            suggestions.extend(ml_suggestions)
        
        # 5. Rule-based suggestions
        rule_suggestions = self._apply_tagging_rules(
            resource_id, resource_type, resource_name, current_tags
        )
        suggestions.extend(rule_suggestions)
        
        # Deduplicate and prioritize suggestions
        suggestions = self._consolidate_suggestions(suggestions)
        
        # Filter out already applied tags
        suggestions = [s for s in suggestions if s.key not in current_tags 
                      or current_tags[s.key] != s.value]
        
        return suggestions
    
    def check_compliance(self,
                        resource_id: str,
                        resource_type: str,
                        current_tags: Dict[str, str]) -> TagComplianceResult:
        """
        Check if resource meets tagging compliance requirements
        
        Args:
            resource_id: Resource identifier
            resource_type: Type of resource
            current_tags: Current tags on the resource
            
        Returns:
            TagComplianceResult
        """
        missing_required = []
        invalid_tags = []
        
        # Check required tags
        for required_key in self.required_tags:
            if required_key not in current_tags:
                missing_required.append(required_key)
        
        # Check tag policies
        for key, value in current_tags.items():
            if key in self.tag_policies:
                policy = self.tag_policies[key]
                
                # Check allowed values
                if 'allowed_values' in policy:
                    if value not in policy['allowed_values']:
                        invalid_tags.append((key, f"Value '{value}' not in allowed values"))
                
                # Check pattern
                if 'pattern' in policy:
                    if not re.match(policy['pattern'], value):
                        invalid_tags.append((key, f"Value '{value}' doesn't match pattern"))
                
                # Check max length
                if 'max_length' in policy:
                    if len(value) > policy['max_length']:
                        invalid_tags.append((key, f"Value exceeds max length of {policy['max_length']}"))
        
        # Calculate compliance score
        total_checks = len(self.required_tags) + len(current_tags)
        failed_checks = len(missing_required) + len(invalid_tags)
        compliance_score = 1.0 - (failed_checks / total_checks) if total_checks > 0 else 0.0
        
        # Get suggestions for missing tags
        suggestions = self.analyze_resource(
            resource_id, resource_type, None, current_tags
        )
        
        # Filter suggestions to only missing required tags
        suggestions = [s for s in suggestions if s.key in missing_required]
        
        return TagComplianceResult(
            resource_id=resource_id,
            resource_type=resource_type,
            compliant=len(missing_required) == 0 and len(invalid_tags) == 0,
            missing_required=missing_required,
            invalid_tags=invalid_tags,
            suggestions=suggestions,
            compliance_score=compliance_score
        )
    
    def enforce_tagging_policies(self,
                               resources: List[Dict[str, Any]],
                               dry_run: bool = True) -> Dict[str, Any]:
        """
        Enforce tagging policies across resources
        
        Args:
            resources: List of resources to check
            dry_run: If True, only report what would be done
            
        Returns:
            Dictionary with enforcement results
        """
        results = {
            'total_resources': len(resources),
            'compliant': 0,
            'non_compliant': 0,
            'tags_added': 0,
            'tags_modified': 0,
            'errors': [],
            'actions': []
        }
        
        for resource in resources:
            resource_id = resource['resource_id']
            resource_type = resource['resource_type']
            current_tags = resource.get('tags', {})
            
            # Check compliance
            compliance = self.check_compliance(resource_id, resource_type, current_tags)
            
            if compliance.compliant:
                results['compliant'] += 1
            else:
                results['non_compliant'] += 1
                
                # Get suggested tags for missing required
                tags_to_add = {}
                for suggestion in compliance.suggestions:
                    if suggestion.confidence >= 0.8:  # High confidence only
                        tags_to_add[suggestion.key] = suggestion.value
                
                if tags_to_add and not dry_run:
                    try:
                        self._apply_tags(resource_id, resource_type, tags_to_add)
                        results['tags_added'] += len(tags_to_add)
                        results['actions'].append({
                            'resource_id': resource_id,
                            'action': 'add_tags',
                            'tags': tags_to_add
                        })
                    except Exception as e:
                        results['errors'].append({
                            'resource_id': resource_id,
                            'error': str(e)
                        })
                elif tags_to_add:
                    results['actions'].append({
                        'resource_id': resource_id,
                        'action': 'would_add_tags',
                        'tags': tags_to_add
                    })
        
        return results
    
    def train_ml_models(self, training_data: List[Dict[str, Any]]):
        """
        Train ML models for tag prediction
        
        Args:
            training_data: List of resources with tags for training
        """
        logger.info("Training ML models for intelligent tagging")
        
        # Prepare training data
        names = []
        environments = []
        applications = []
        
        for resource in training_data:
            name = resource.get('name', '')
            tags = resource.get('tags', {})
            
            if name and 'Environment' in tags:
                names.append(name)
                environments.append(tags['Environment'])
            
            if name and 'Application' in tags:
                applications.append(tags['Application'])
        
        # Train environment classifier
        if names and environments:
            self.name_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=1000
            )
            
            name_features = self.name_vectorizer.fit_transform(names)
            
            self.tag_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.tag_classifier.fit(name_features, environments)
            
            logger.info(f"Trained classifier on {len(names)} samples")
        
        # Build tag relationship graph
        self._build_tag_relationships(training_data)
    
    def _suggest_from_name(self, name: str, resource_type: str) -> List[TagSuggestion]:
        """Generate suggestions based on resource name"""
        suggestions = []
        name_lower = name.lower()
        
        # Check environment patterns
        for env, patterns in self.ENV_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    suggestions.append(TagSuggestion(
                        key='Environment',
                        value=env,
                        confidence=0.9,
                        category=TagCategory.ENVIRONMENT,
                        reasoning=f"Name contains '{pattern}' pattern",
                        based_on=['name_pattern']
                    ))
                    break
        
        # Check application patterns
        for app_type, patterns in self.APP_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    suggestions.append(TagSuggestion(
                        key='ApplicationType',
                        value=app_type,
                        confidence=0.85,
                        category=TagCategory.APPLICATION,
                        reasoning=f"Name suggests {app_type} application",
                        based_on=['name_pattern']
                    ))
                    break
        
        # Extract potential project/team from name
        # Look for patterns like "team-project-env-component"
        parts = re.split(r'[-_]', name)
        if len(parts) >= 2:
            # First part might be team
            suggestions.append(TagSuggestion(
                key='Team',
                value=parts[0],
                confidence=0.6,
                category=TagCategory.TEAM,
                reasoning="Extracted from name prefix",
                based_on=['name_structure']
            ))
            
            # Second part might be project/application
            if len(parts) >= 3:
                suggestions.append(TagSuggestion(
                    key='Project',
                    value=parts[1],
                    confidence=0.6,
                    category=TagCategory.PROJECT,
                    reasoning="Extracted from name structure",
                    based_on=['name_structure']
                ))
        
        return suggestions
    
    def _suggest_from_relationships(self,
                                  resource_id: str,
                                  resource_type: str,
                                  current_tags: Dict[str, str]) -> List[TagSuggestion]:
        """Generate suggestions based on resource relationships"""
        suggestions = []
        
        # Get related resources
        related = self._get_related_resources(resource_id, resource_type)
        
        if not related:
            return suggestions
        
        # Aggregate tags from related resources
        tag_votes = defaultdict(Counter)
        
        for related_resource in related:
            related_tags = related_resource.get('tags', {})
            relationship_type = related_resource.get('relationship_type', 'associated')
            
            # Weight based on relationship strength
            weight = 1.0
            if relationship_type == 'parent':
                weight = 2.0
            elif relationship_type == 'vpc':
                weight = 1.5
            
            for key, value in related_tags.items():
                if key not in current_tags:  # Only suggest missing tags
                    tag_votes[key][value] += weight
        
        # Generate suggestions from votes
        for key, value_counts in tag_votes.items():
            if value_counts:
                most_common_value, count = value_counts.most_common(1)[0]
                confidence = min(count / len(related), 1.0) * 0.8  # Cap at 0.8
                
                if confidence >= 0.5:
                    suggestions.append(TagSuggestion(
                        key=key,
                        value=most_common_value,
                        confidence=confidence,
                        category=self._categorize_tag_key(key),
                        reasoning=f"Common among {count} related resources",
                        based_on=['relationships']
                    ))
        
        return suggestions
    
    def _suggest_from_usage(self,
                          resource_id: str,
                          resource_type: str,
                          usage_metrics: Dict[str, Any]) -> List[TagSuggestion]:
        """Generate suggestions based on usage patterns"""
        suggestions = []
        
        # Analyze CPU usage patterns for environment detection
        if 'cpu_utilization' in usage_metrics:
            avg_cpu = usage_metrics['cpu_utilization'].get('average', 0)
            max_cpu = usage_metrics['cpu_utilization'].get('max', 0)
            
            if avg_cpu < 5 and max_cpu < 20:
                suggestions.append(TagSuggestion(
                    key='Environment',
                    value='development',
                    confidence=0.7,
                    category=TagCategory.ENVIRONMENT,
                    reasoning="Low CPU usage suggests non-production",
                    based_on=['usage_pattern']
                ))
            elif avg_cpu > 60:
                suggestions.append(TagSuggestion(
                    key='Environment',
                    value='production',
                    confidence=0.75,
                    category=TagCategory.ENVIRONMENT,
                    reasoning="High CPU usage suggests production workload",
                    based_on=['usage_pattern']
                ))
        
        # Analyze access patterns
        if 'access_pattern' in usage_metrics:
            pattern = usage_metrics['access_pattern']
            
            if pattern.get('business_hours_only', False):
                suggestions.append(TagSuggestion(
                    key='WorkloadType',
                    value='business_hours',
                    confidence=0.8,
                    category=TagCategory.LIFECYCLE,
                    reasoning="Access only during business hours",
                    based_on=['access_pattern']
                ))
            
            if pattern.get('periodic', False):
                suggestions.append(TagSuggestion(
                    key='WorkloadType',
                    value='batch_processing',
                    confidence=0.7,
                    category=TagCategory.APPLICATION,
                    reasoning="Periodic usage pattern detected",
                    based_on=['usage_pattern']
                ))
        
        return suggestions
    
    def _suggest_from_ml(self,
                       resource_id: str,
                       resource_type: str,
                       resource_name: Optional[str],
                       usage_metrics: Optional[Dict[str, Any]]) -> List[TagSuggestion]:
        """Generate suggestions using ML models"""
        suggestions = []
        
        if not self.tag_classifier or not self.name_vectorizer or not resource_name:
            return suggestions
        
        try:
            # Vectorize the name
            name_features = self.name_vectorizer.transform([resource_name])
            
            # Predict environment
            env_prediction = self.tag_classifier.predict(name_features)[0]
            env_proba = self.tag_classifier.predict_proba(name_features)[0].max()
            
            if env_proba >= 0.6:
                suggestions.append(TagSuggestion(
                    key='Environment',
                    value=env_prediction,
                    confidence=env_proba,
                    category=TagCategory.ENVIRONMENT,
                    reasoning="ML model prediction based on name patterns",
                    based_on=['ml_model']
                ))
        
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
        
        return suggestions
    
    def _apply_tagging_rules(self,
                           resource_id: str,
                           resource_type: str,
                           resource_name: Optional[str],
                           current_tags: Dict[str, str]) -> List[TagSuggestion]:
        """Apply configured tagging rules"""
        suggestions = []
        
        for rule_id, rule in self.tagging_rules.items():
            if not rule.enabled:
                continue
            
            # Check if rule applies
            applies = False
            
            if rule.condition_type == 'pattern':
                if resource_name and 'name_pattern' in rule.condition:
                    pattern = rule.condition['name_pattern']
                    if re.search(pattern, resource_name, re.IGNORECASE):
                        applies = True
            
            elif rule.condition_type == 'resource_type':
                if resource_type in rule.condition.get('types', []):
                    applies = True
            
            elif rule.condition_type == 'tag_exists':
                required_tag = rule.condition.get('tag_key')
                required_value = rule.condition.get('tag_value')
                
                if required_tag in current_tags:
                    if required_value is None or current_tags[required_tag] == required_value:
                        applies = True
            
            if applies:
                for key, value in rule.tags_to_apply.items():
                    if key not in current_tags:
                        suggestions.append(TagSuggestion(
                            key=key,
                            value=value,
                            confidence=0.95,
                            category=self._categorize_tag_key(key),
                            reasoning=f"Applied by rule: {rule.name}",
                            based_on=['tagging_rule']
                        ))
        
        return suggestions
    
    def _consolidate_suggestions(self, suggestions: List[TagSuggestion]) -> List[TagSuggestion]:
        """Consolidate and prioritize suggestions"""
        # Group by key
        by_key = defaultdict(list)
        for suggestion in suggestions:
            by_key[suggestion.key].append(suggestion)
        
        # For each key, pick the highest confidence suggestion
        consolidated = []
        for key, key_suggestions in by_key.items():
            # Sort by confidence
            key_suggestions.sort(key=lambda s: s.confidence, reverse=True)
            
            # Take the best one
            best = key_suggestions[0]
            
            # But combine the reasoning from multiple sources
            if len(key_suggestions) > 1:
                all_reasons = [s.reasoning for s in key_suggestions]
                all_based_on = list(set(sum([s.based_on for s in key_suggestions], [])))
                
                best.reasoning = "; ".join(all_reasons[:3])  # Top 3 reasons
                best.based_on = all_based_on
                
                # Boost confidence if multiple methods agree
                if len(set(s.value for s in key_suggestions)) == 1:
                    best.confidence = min(best.confidence * 1.2, 1.0)
            
            consolidated.append(best)
        
        # Sort by confidence
        consolidated.sort(key=lambda s: s.confidence, reverse=True)
        
        return consolidated
    
    def _get_related_resources(self, resource_id: str, resource_type: str) -> List[Dict[str, Any]]:
        """Get resources related to the given resource"""
        related = []
        
        try:
            if resource_type == 'ec2':
                # Get VPC, subnet, security groups
                response = self.ec2.describe_instances(InstanceIds=[resource_id])
                if response['Reservations']:
                    instance = response['Reservations'][0]['Instances'][0]
                    
                    # Add VPC
                    vpc_id = instance.get('VpcId')
                    if vpc_id:
                        vpc_tags = self._get_resource_tags('vpc', vpc_id)
                        related.append({
                            'resource_id': vpc_id,
                            'resource_type': 'vpc',
                            'relationship_type': 'vpc',
                            'tags': vpc_tags
                        })
                    
                    # Add security groups
                    for sg in instance.get('SecurityGroups', []):
                        sg_tags = self._get_resource_tags('security-group', sg['GroupId'])
                        related.append({
                            'resource_id': sg['GroupId'],
                            'resource_type': 'security-group',
                            'relationship_type': 'security',
                            'tags': sg_tags
                        })
            
            # Add more relationship logic for other resource types
            
        except Exception as e:
            logger.debug(f"Error getting related resources: {e}")
        
        return related
    
    def _get_resource_tags(self, resource_type: str, resource_id: str) -> Dict[str, str]:
        """Get tags for a specific resource"""
        tags = {}
        
        try:
            if resource_type in ['vpc', 'subnet', 'security-group', 'ec2']:
                response = self.ec2.describe_tags(
                    Filters=[
                        {'Name': 'resource-id', 'Values': [resource_id]}
                    ]
                )
                
                for tag in response.get('Tags', []):
                    tags[tag['Key']] = tag['Value']
        
        except Exception as e:
            logger.debug(f"Error getting tags for {resource_id}: {e}")
        
        return tags
    
    def _apply_tags(self, resource_id: str, resource_type: str, tags: Dict[str, str]):
        """Apply tags to a resource"""
        tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
        
        if resource_type in ['ec2', 'vpc', 'subnet', 'security-group']:
            self.ec2.create_tags(
                Resources=[resource_id],
                Tags=tag_list
            )
        
        # Add support for other resource types
    
    def _categorize_tag_key(self, key: str) -> TagCategory:
        """Categorize a tag key"""
        key_lower = key.lower()
        
        if key_lower in ['environment', 'env', 'stage']:
            return TagCategory.ENVIRONMENT
        elif key_lower in ['application', 'app', 'service']:
            return TagCategory.APPLICATION
        elif key_lower in ['team', 'owner', 'department']:
            return TagCategory.TEAM
        elif key_lower in ['costcenter', 'cost-center', 'billing']:
            return TagCategory.COST_CENTER
        elif key_lower in ['project', 'product']:
            return TagCategory.PROJECT
        elif key_lower in ['compliance', 'regulation']:
            return TagCategory.COMPLIANCE
        elif key_lower in ['lifecycle', 'retention', 'expiry']:
            return TagCategory.LIFECYCLE
        else:
            return TagCategory.AUTOMATION
    
    def _build_tag_relationships(self, training_data: List[Dict[str, Any]]):
        """Build tag relationship graph from training data"""
        for resource in training_data:
            tags = resource.get('tags', {})
            
            # Count tag frequencies
            for key, value in tags.items():
                self.tag_frequency[key][value] += 1
            
            # Build relationships
            tag_keys = list(tags.keys())
            for i, key1 in enumerate(tag_keys):
                for key2 in tag_keys[i+1:]:
                    self.tag_relationships[key1].add(key2)
                    self.tag_relationships[key2].add(key1)
    
    def generate_tagging_report(self, 
                              resources: List[Dict[str, Any]],
                              output_file: str = 'tagging_report.json') -> Dict[str, Any]:
        """Generate comprehensive tagging report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_resources': len(resources),
                'fully_tagged': 0,
                'partially_tagged': 0,
                'untagged': 0,
                'compliance_rate': 0.0
            },
            'missing_tags': defaultdict(int),
            'tag_coverage': defaultdict(int),
            'compliance_details': [],
            'improvement_opportunities': []
        }
        
        compliant_count = 0
        
        for resource in resources:
            resource_id = resource['resource_id']
            resource_type = resource['resource_type']
            current_tags = resource.get('tags', {})
            
            # Check compliance
            compliance = self.check_compliance(resource_id, resource_type, current_tags)
            
            if compliance.compliant:
                compliant_count += 1
                report['summary']['fully_tagged'] += 1
            elif len(current_tags) > 0:
                report['summary']['partially_tagged'] += 1
            else:
                report['summary']['untagged'] += 1
            
            # Track missing tags
            for missing in compliance.missing_required:
                report['missing_tags'][missing] += 1
            
            # Track tag coverage
            for tag_key in self.required_tags:
                if tag_key in current_tags:
                    report['tag_coverage'][tag_key] += 1
            
            # Add to detailed results
            report['compliance_details'].append({
                'resource_id': resource_id,
                'resource_type': resource_type,
                'compliance_score': compliance.compliance_score,
                'missing_required': compliance.missing_required,
                'suggestions': [
                    {
                        'key': s.key,
                        'value': s.value,
                        'confidence': s.confidence,
                        'reasoning': s.reasoning
                    }
                    for s in compliance.suggestions[:3]  # Top 3 suggestions
                ]
            })
        
        # Calculate overall compliance rate
        report['summary']['compliance_rate'] = compliant_count / len(resources) if resources else 0
        
        # Identify improvement opportunities
        for tag_key, missing_count in report['missing_tags'].items():
            if missing_count > 0:
                coverage_pct = (len(resources) - missing_count) / len(resources) * 100
                report['improvement_opportunities'].append({
                    'tag_key': tag_key,
                    'missing_count': missing_count,
                    'coverage_percentage': coverage_pct,
                    'impact': 'high' if tag_key in self.required_tags else 'medium'
                })
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Tagging report saved to {output_file}")
        
        return report