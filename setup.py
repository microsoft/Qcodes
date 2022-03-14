from setuptools import setup
from versioningit import get_cmdclasses

if __name__ == "__main__":
    setup(
        cmdclass=get_cmdclasses(),
    )
