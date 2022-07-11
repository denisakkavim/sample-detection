from setuptools import find_packages, setup

setup(
    name="sample-detection",
    packages=find_packages(),
    version="0.1.0",
    description="Detecting samples in music",
    author="Denis Akkavim",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sample-detection=sample_detection.cli.cli:cli",
        ]
    },
)
