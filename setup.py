from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aws-cost-optimizer",
    version="1.0.0",
    author="AWS Cost Optimizer Team",
    author_email="aws-cost-optimizer@your-org.com",
    description="A comprehensive tool for safely reducing AWS costs without breaking production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/aws-cost-optimizer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aws-cost-optimizer=aws_cost_optimizer.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "aws_cost_optimizer": ["config/*.yaml", "policies/*.yaml"],
    },
)