import os
import sys

from setuptools import setup

# we need to import versioneer from the current dir
# once versioneer is pep517/518 compliant this can be removed
sys.path.append(os.path.dirname(__file__))
import versioneer

sys.path.pop()


if __name__ == "__main__":
    setup(
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
