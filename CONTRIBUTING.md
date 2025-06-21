# Contributing to AWS Cost Optimizer

First off, thank you for considering contributing to AWS Cost Optimizer! It's people like you that make this tool better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to aws-cost-optimizer@your-org.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment (see below)
4. Create a branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, please include:

- A clear and descriptive title
- Exact steps to reproduce the problem
- Expected behavior vs actual behavior
- Your environment details (OS, Python version, AWS regions, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Any possible drawbacks or considerations
- Examples of how the enhancement would be used

### Contributing Code

#### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements needed

#### Pull Requests

1. **Small, focused PRs**: Keep pull requests small and focused on a single issue
2. **Follow the style guide**: Use the project's coding standards
3. **Write tests**: Add tests for new functionality
4. **Update documentation**: Update docs for any changed functionality
5. **Add yourself to CONTRIBUTORS**: If this is your first contribution

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for testing)
- AWS CLI configured with test credentials

### Local Development

1. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/aws-cost-optimizer.git
   cd aws-cost-optimizer
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Using Docker for Development

```bash
# Build development image
docker-compose build dev

# Run tests in container
docker-compose run test

# Start LocalStack for AWS service mocking
docker-compose up localstack
```

## Style Guidelines

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 120 characters
- Use type hints where possible
- Use f-strings for string formatting
- Sort imports with `isort`

### Code Formatting

We use `black` for automatic code formatting:

```bash
# Format all code
black src tests

# Check formatting without changes
black --check src tests
```

### Linting

We use `flake8` for linting:

```bash
# Run linter
flake8 src tests

# Run type checker
mypy src
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_savings(instance_type: str, hours: int) -> float:
    """Calculate potential savings for an instance type.
    
    Args:
        instance_type: The EC2 instance type (e.g., 't3.micro')
        hours: Number of hours the instance runs per month
        
    Returns:
        The potential monthly savings in USD
        
    Raises:
        ValueError: If instance_type is not recognized
    """
```

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Build process or auxiliary tool changes

Example:
```
feat: add S3 Intelligent-Tiering recommendations

- Analyze S3 bucket access patterns
- Recommend Intelligent-Tiering for buckets > 1TB
- Add safety checks for compliance-tagged buckets

Closes #123
```

## Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**:
   ```bash
   pytest
   ```

3. **Update documentation** if needed

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes

6. **Address review feedback**:
   - Make requested changes
   - Push additional commits
   - Reply to review comments

7. **Merge**:
   - Once approved, your PR will be merged
   - Delete your feature branch after merge

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/aws_cost_optimizer

# Run specific test file
pytest tests/unit/test_ec2_optimizer.py

# Run with verbose output
pytest -v
```

### Integration Tests

```bash
# Start test infrastructure
docker-compose up -d localstack

# Run integration tests
pytest tests/integration

# Clean up
docker-compose down
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use `pytest` fixtures for common setup
- Mock AWS services using `moto` or LocalStack
- Aim for >80% code coverage

Example test:

```python
def test_calculate_ec2_savings(mock_boto3):
    """Test EC2 savings calculation."""
    # Arrange
    optimizer = EC2Optimizer()
    instance = {
        'instance_id': 'i-1234567890',
        'instance_type': 't3.large',
        'cpu_avg': 5.0  # Below threshold
    }
    
    # Act
    recommendation = optimizer.analyze_instance_optimization(instance)
    
    # Assert
    assert recommendation.action == 'stop'
    assert recommendation.estimated_monthly_savings > 0
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include type hints for better IDE support
- Add inline comments for complex logic

### User Documentation

- Update README.md for significant changes
- Add examples for new features
- Update configuration documentation
- Include migration guides for breaking changes

### API Documentation

We use Sphinx for API documentation:

```bash
# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create a release PR
4. After merge, tag the release:
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push upstream v1.2.0
   ```

## Getting Help

- Check the [documentation](https://aws-cost-optimizer.readthedocs.io)
- Search [existing issues](https://github.com/your-org/aws-cost-optimizer/issues)
- Join our [Slack channel](https://your-org.slack.com/channels/aws-cost-optimizer)
- Email: aws-cost-optimizer@your-org.com

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to AWS Cost Optimizer!