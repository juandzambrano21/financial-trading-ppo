"""
Setup script for FSRPPO package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="fsrppo",
    version="1.0.0",
    author="Juan Diego Zambrano",
    author_email="juanzambrano@oriya.ai",
    description="Financial Signal Representation Proximal Policy Optimization for Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juandzambrano21/financial-trading-ppo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "fsrppo-train=fsrppo.cli.train:main",
            "fsrppo-backtest=fsrppo.cli.backtest:main",
            "fsrppo-demo=fsrppo.examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fsrppo": ["data/*.json", "configs/*.yaml"],
    },
    zip_safe=False,
)