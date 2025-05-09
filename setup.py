# Read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="quemb",
    version="0.1.0-alpha",
    description="QuEmb: A framework for efficient simulation of large molecules, "
    "surfaces, and solids via Bootstrap Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/troyvvgroup/quemb",
    url="https://vanvoorhisgroup.mit.edu/quemb",
    license="Apache 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<3.13",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.7.0",
        "pyscf>=2.0.0",
        "networkx",
        "matplotlib",
        "libdmet @ git+https://github.com/gkclab/libdmet_preview.git",
        "attrs",
        "cattrs",
        "pyyaml",
        # TODO: Remove the git dependency once chemcoord >= 2.2 is on PyPI
        "chemcoord @ git+https://github.com/mcocdawc/chemcoord.git",
        "numba",
        "ordered-set",
    ],
)
