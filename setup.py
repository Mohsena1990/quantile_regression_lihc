from pathlib import Path
from setuptools import find_packages, setup

# One dependency list: requirement.txt (kept in sync by construction, not by hand).
REQUIREMENTS = [l.strip() for l in (Path(__file__).parent / "requirement.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")]

setup(
    name="energy_poverty_identification",
    version="0.3",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.10",
)
