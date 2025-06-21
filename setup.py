from setuptools import setup, find_packages

setup(
    name='aws-cost-optimizer',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'boto3>=1.26.0',
        'click>=8.0.0',
        'pandas>=2.0.0',
        'pyyaml>=6.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'openpyxl>=3.1.0',
    ],
    entry_points={
        'console_scripts': [
            'aws-cost-optimizer=aws_cost_optimizer.cli:cli',
        ],
    },
)