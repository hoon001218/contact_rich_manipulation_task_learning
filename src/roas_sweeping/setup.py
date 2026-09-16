"""Editable installation for the UR5e/ROAS-hand shelf-sweeping task."""

from setuptools import find_packages, setup


setup(
    name="roas-sweeping",
    version="0.1.0",
    description="Isaac Lab shelf-sweeping environment with a UR5e and ROAS force-sensor hand",
    packages=find_packages(),
    python_requires=">=3.10",
    zip_safe=False,
)
