import setuptools
from setuptools import setup
from versioningit import get_cmdclasses

# this file does not contain configuration
# std configuration as defined by pep621
# is in pyproject.toml
# and setuptools specific config in setup.cfg

if int(setuptools.__version__.split(".")[0]) < 61:
    raise RuntimeError(
        "At least setuptools 61 is required to install qcodes from source"
    )

try:
    import pip

    if int(pip.__version__.split(".")[0]) < 19:
        raise RuntimeError("At least pip 19 is required to install qcodes from source")
except ImportError:
    # we are not being executed from pip so pip version is not important
    pass

if __name__ == "__main__":
    setup(
        cmdclass=get_cmdclasses(),
    )
