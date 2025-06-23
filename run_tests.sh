#!/bin/bash

# AWS Cost Optimizer - Test Runner Script

echo "🧪 Running AWS Cost Optimizer Tests..."
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run tests for a module
run_module_tests() {
    local module=$1
    local description=$2
    
    echo -e "\n${YELLOW}Testing: ${description}${NC}"
    python -m pytest tests/test_new_features.py::$module -v
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} tests passed${NC}"
    else
        echo -e "${RED}✗ ${description} tests failed${NC}"
        exit 1
    fi
}

# Run individual module tests
run_module_tests "TestPeriodicDetector" "Periodic Resource Detection"
run_module_tests "TestCostPredictor" "ML Cost Prediction"
run_module_tests "TestRealtimeController" "Real-time Cost Controls"
run_module_tests "TestIntelligentTagger" "Intelligent Tagging"
run_module_tests "TestSavingsTracker" "Savings Tracking"
run_module_tests "TestCLICommands" "CLI Commands"
run_module_tests "TestIntegration" "Cross-module Integration"

echo -e "\n${YELLOW}Running all existing tests...${NC}"
python -m pytest tests/ -v --cov=aws_cost_optimizer --cov-report=term-missing

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
else
    echo -e "\n${RED}❌ Some tests failed${NC}"
    exit 1
fi

# Generate coverage report
echo -e "\n${YELLOW}Generating coverage report...${NC}"
python -m pytest tests/ --cov=aws_cost_optimizer --cov-report=html

echo -e "${GREEN}📊 Coverage report generated in htmlcov/index.html${NC}"

# Run type checking
echo -e "\n${YELLOW}Running type checking...${NC}"
python -m mypy src/aws_cost_optimizer --ignore-missing-imports

# Run linting
echo -e "\n${YELLOW}Running code linting...${NC}"
python -m flake8 src/aws_cost_optimizer --max-line-length=120 --exclude=__pycache__

echo -e "\n${GREEN}🎉 Test suite complete!${NC}"