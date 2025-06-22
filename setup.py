from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aws-cost-optimizer",
    version="0.1.0",
    author="Your Organization",
    author_email="aws-cost-optimizer@your-org.com",
    description="AWS Cost Optimization Tool for identifying and implementing cost savings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/aws-cost-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "boto3>=1.26.0",
        "botocore>=1.29.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "pandas>=1.5.0",
        "openpyxl>=3.1.0",
        "tabulate>=0.9.0",
        "python-dateutil>=2.8.0",
        "colorama>=0.4.6",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "moto>=4.0.0",
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aws-cost-optimizer=aws_cost_optimizer.cli.main:cli",
        ],
    },
)