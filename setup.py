"""
Setup script for CCUB2 Agent.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="ccub2-agent",
    version="0.1.0",
    description="Model-Agnostic Cultural Bias Mitigation System for Image Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Carnegie Mellon University",
    author_email="chans@andrew.cmu.edu",
    url="https://github.com/cmubig/ccub2-agent",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    keywords=[
        "cultural-bias",
        "image-generation",
        "ai-fairness",
        "model-agnostic",
        "text-to-image",
        "image-editing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cmubig/ccub2-agent/issues",
        "Source": "https://github.com/cmubig/ccub2-agent",
        "Documentation": "https://github.com/cmubig/ccub2-agent/blob/main/ARCHITECTURE.md",
        "Paper": "https://arxiv.org/abs/2510.20042",
    },
)
