# step 3: create a package
from setuptools import setup

setup(
    name="astra_rev1",
    version="0.0.1",
    packages=["astra_rev1", "astra_rev1.envs"],
    install_requires=["gym", "numpy"]

    # step 4: install package locally: in the \ASTRA-GeneralRepo directory, run pip install -e .
    # step 5: import statement + env variable 
    # step 6 (optional): use a wrapper to change the format/what types of observations you get from an environment instance
)