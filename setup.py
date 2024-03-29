#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "yt",
    "netcdf4",
    "h5py",
    "scipy",
    "geopandas>=0.11.1",
    "shapely",
    "xarray",
    "sklearn",
    "pygeos",
    "tslearn",
    "dask",
    "yt_xarray",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Chris Havlin",
    author_email="chris.havlin@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="some tools for analysis of 3d geophysical datasets",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="ytgeotools",
    name="ytgeotools",
    packages=find_packages(include=["ytgeotools", "ytgeotools.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/chrishavlin/ytgeotools",
    version="0.1.0",
    zip_safe=False,
)
