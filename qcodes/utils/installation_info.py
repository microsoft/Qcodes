"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
from typing import Dict, List
import subprocess
from pip._vendor import pkg_resources
import importlib

import qcodes


# sometimes a package is imported as something else than its package name
# _IMPORT_NAMES maps package name to import name
_IMPORT_NAMES = {'pyzmq': 'zmq'}
_PACKAGE_NAMES = {v: k for k, v in _IMPORT_NAMES.items()}

# sometimes we import non-versioned packages backported from the standard
# library (e.g. dataclasses for python 3.6). Those should be excluded from
# any version listing
_BACKPORTED_PACKAGES = ['dataclasses']


def _pip_list_parser(pip_list_raw: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse the raw output of pip list into a dict with package name keys and
    a dict {'version': ..., 'location': ...} values

    Args:
        pip_list_raw: a list of strings, one line per list element
    """
    out: Dict[str, Dict[str, str]] = {}

    # The first two lines are the header and header-content separator
    for line in pip_list_raw[2:]:
        fields = line.split()
        pkg, ver = fields[0], fields[1]
        if len(fields) == 3:
            loc = fields[2]
        else:
            loc = ''

        out.update({pkg: {'version': ver, 'location': loc}})

    return out


def _get_lines_from_pip_list_e() -> List[str]:
    """
    Return the output of `pip list -e` as a list of lines
    """
    pipproc = subprocess.run(['pip', 'list', '-e'], stdout=subprocess.PIPE)
    lines = pipproc.stdout.decode('utf-8').split('\r\n')[:-1]

    return lines


def is_qcodes_installed_editably() -> bool:
    """
    Return a boolean with the answer to the question 'is QCoDeS installed in
    editable mode?'
    """
    lines = _get_lines_from_pip_list_e()

    editable_packages = _pip_list_parser(lines)

    return 'qcodes' in editable_packages


def get_qcodes_version() -> str:
    """
    Get the version of the currently installed QCoDeS
    """
    return qcodes.version.__version__  # type: ignore


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
