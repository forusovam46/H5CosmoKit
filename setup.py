from setuptools import setup, find_packages
import setuptools

# Load the version number from the H5CosmoKit/_version.py module

# version = open("H5CosmoKit/_version.py")
# version = version.readlines()[-1].split()[-1].strip("\"'")

version = "1.0.0"

# Load the readme as long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Load the package requirements from requirements.txt
with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()

setuptools.setup(
    name="H5CosmoKit",
    version=version,
    author="forusovam46",
    author_email="forusova@gmail.com",
    description="A Python package for analyzing cosmological simulation output",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/forusovam46/H5CosmoKit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/forusovam46/H5CosmoKit/",
    },
    python_requires=">=3.6",
    install_requires=required,
    extras_require={
        "testing": [
            "pytest",
            "coverage",
        ],
        "docs": [
            "sphinx",
            "readthedocs-sphinx-ext",
        ],
    },
    # Include any package data in your MANIFEST.in file, or specify them here
    # For example, if you have data files in your package directory
    package_data={"H5CosmoKit": ["data/*"]},  # Adjust the path to your data files
)
