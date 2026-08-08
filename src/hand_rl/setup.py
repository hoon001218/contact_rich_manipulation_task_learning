"""Editable installation for the RH56 grasp task."""

from setuptools import find_packages, setup


setup(
    name="hand-rl",
    version="0.1.0",
    description="Isaac Lab UR5e and Inspire RH56 shelf-grasp task",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)

