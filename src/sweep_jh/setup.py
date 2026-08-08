"""Editable installation for the JH Isaac Lab sweep task."""

from setuptools import find_packages, setup

setup(
    name="sweep-jh",
    version="0.1.0",
    description="JH training configuration for the UR5e OSC sweep task",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
