"""Editable installation for the force-sensor hand grasp task."""

from setuptools import find_packages, setup


setup(
    name="hand-rl",
    version="0.1.0",
    description="Isaac Lab UR5e and ROAS force-sensor hand shelf-grasp task",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
