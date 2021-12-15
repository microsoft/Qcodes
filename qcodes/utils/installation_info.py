"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
import json
import logging
import subprocess
import sys
from typing import Dict, List, Optional

import pkg_resources
import requirements

if sys.version_info >= (3, 8):
    from importlib.metadata import PackageNotFoundError, distribution, version
else:
    # 3.7 and earlier
    from importlib_metadata import distribution, version, PackageNotFoundError

import qcodes

log = logging.getLogger(__name__)


def is_qcodes_installed_editably() -> Optional[bool]:
    """
    Try to ask pip whether QCoDeS is installed in editable mode and return
    the answer a boolean. Returns None if pip somehow did not respond as
    expected.
    """

    answer: Optional[bool]

    try:
        pipproc = subprocess.run(['python', '-m', 'pip', 'list', '-e', '--no-index',
                                  '--format=json'],
                                 check=True,
                                 stdout=subprocess.PIPE)
        e_pkgs = json.loads(pipproc.stdout.decode('utf-8'))
        answer = any([d["name"] == 'qcodes' for d in e_pkgs])
    except Exception as e:  # we actually do want a catch-all here
        log.warning(f'{type(e)}: {str(e)}')
        answer = None

    return answer


def get_qcodes_version() -> str:
    """
    Get the version of the currently installed QCoDeS
    """
    return qcodes.__version__


def get_qcodes_requirements() -> List[str]:
    """
    Return a list of the names of the packages that QCoDeS requires
    """
    qc_pkg = distribution('qcodes').requires
    if qc_pkg is None:
        return []
    package_names = [list(requirements.parse(req))[0].name for req in qc_pkg]

    return package_names


def get_qcodes_requirements_versions() -> Dict[str, str]:
    """
    Return a dictionary of the currently installed versions of the packages
    that QCoDeS requires. The dict maps package name to version string.
    If an (optional) dependency is not installed the name maps to "Not installed".
    """

    req_names = get_qcodes_requirements()

    req_versions = {}

    for req in req_names:
        try:
            req_versions[req] = version(req)
        except PackageNotFoundError:
            req_versions[req] = "Not installed"

    return req_versions


def get_all_installed_package_versions() -> Dict[str, str]:
    """
    Return a dictionary of the currently installed packages and their versions.
    """
    packages = pkg_resources.working_set
    return {i.key: i.version for i in packages}
