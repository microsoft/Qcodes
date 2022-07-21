import setuptools
from setuptools import setup
from versioningit import get_cmdclasses

if int(setuptools.__version__.split(".")[0]) < 61:
    raise RuntimeError(
        "At least setuptools 61 is required to install qcodes from source"
    )

if __name__ == "__main__":
    setup(
        cmdclass=get_cmdclasses(),
    )
