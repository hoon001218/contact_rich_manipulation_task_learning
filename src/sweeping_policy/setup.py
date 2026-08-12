"""Editable installation for the shelf-sweeping Isaac Lab task."""

from setuptools import find_packages, setup


setup(
    name="sweeping-policy",
    version="0.1.0",
    description="Nucleus-backed UR5e/Robotiq shelf-sweeping task",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
