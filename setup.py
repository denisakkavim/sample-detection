from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sample-detection",
    packages=find_packages(),
    version="1.0.0",
    description="Detecting samples in music",
    author="Denis Akkavim",
    license="MIT",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sample-detection=sample_detection.cli:cli",
        ]
    },
)
