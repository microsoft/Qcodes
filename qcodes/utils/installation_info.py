"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
from typing import Dict, List, Optional
import subprocess
from pip._vendor import pkg_resources
import importlib
import json
import logging

import qcodes


# sometimes a package is imported as something else than its package name
# _IMPORT_NAMES maps package name to import name
_IMPORT_NAMES = {'pyzmq': 'zmq'}
_PACKAGE_NAMES = {v: k for k, v in _IMPORT_NAMES.items()}

# sometimes we import non-versioned packages backported from the standard
# library (e.g. dataclasses for python 3.6). Those should be excluded from
# any version listing
_BACKPORTED_PACKAGES = ['dataclasses']


log = logging.getLogger(__name__)


def is_qcodes_installed_editably() -> Optional[bool]:
    """
    Try to ask pip whether QCoDeS is installed in editable mode and return
    the answer a boolean. Returns None if pip somehow did not respond as
    expected.
    """

    answer: Optional[bool]

    try:
        pipproc = subprocess.run(['pip', 'list', '-e', '--format=json'],
                                  stdout=subprocess.PIPE)
        e_pkgs = json.loads(pipproc.stdout.decode('utf-8'))
        answer = any([d["name"] == 'qcodes' for d in e_pkgs])
    except Exception as e:  # we actually do want a catch-all here
        log.warning('f{type(e)}: {str(e)}')
        answer = None

    return answer


def get_qcodes_version() -> str:
    """
    Get the version of the currently installed QCoDeS
    """
    return qcodes.version.__version__


def get_qcodes_requirements() -> List[str]:
    """
    Return a list of the names of the packages that QCoDeS requires
    """
    qc_pkg = pkg_resources.working_set.by_key['qcodes']

    requirements = [str(r) for r in qc_pkg.requires()]

    package_names = [n.split('>')[0].split('=')[0] for n in requirements]

    return package_names


def get_qcodes_requirements_versions() -> Dict[str, str]:
    """
    Return a dictionary of the currently installed versions of the packages
    that QCoDeS requires. The dict maps package name to version string.
    """

    req_names = get_qcodes_requirements()

    req_modules = []

    for req_name in req_names:
        if req_name in _BACKPORTED_PACKAGES:
            pass
        elif req_name in _IMPORT_NAMES:
            req_modules.append(_IMPORT_NAMES[req_name])
        else:
            req_modules.append(req_name)

    req_versions = {}

    for req_module in req_modules:
        mod = importlib.import_module(req_module)
        if req_module in _PACKAGE_NAMES:
            req_pkg = _PACKAGE_NAMES[req_module]
        else:
            req_pkg = req_module
        req_versions.update({req_pkg: mod.__version__})  # type: ignore

    return req_versions
