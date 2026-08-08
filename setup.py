"""Setup file for editable install (pip install -e .)"""

from setuptools import find_packages, setup

setup(
    name="defect-classifier",
    version="1.0.0",
    description="Image-based defect classification ML system",
    packages=find_packages(),
    python_requires=">=3.10",
)
