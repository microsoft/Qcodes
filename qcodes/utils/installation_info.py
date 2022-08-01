"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
import json
import logging
import subprocess
import sys
from typing import Dict, Optional

if sys.version_info >= (3, 10):
    # distribution.name used below became part of the
    # official api in 3.10
    from importlib.metadata import distributions
else:
    # 3.9 and earlier
    from importlib_metadata import distributions

from qcodes.utils.deprecate import deprecate

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


@deprecate("function 'get_qcodes_version'", alternative="qcodes.__version__")
def get_qcodes_version() -> str:
    """
    Get the version of the currently installed QCoDeS
    """
    from qcodes._version import __version__
    return __version__


def get_all_installed_package_versions() -> Dict[str, str]:
    """
    Return a dictionary of the currently installed packages and their versions.
    """
    return {d.name: d.version for d in distributions()}


def convert_legacy_version_to_supported_version(ver: str) -> str:
    """
    Convert a legacy version str containing single chars rather than
    numbers to a regular version string. This is done by replacing a char
    by its ASCII code (using ``ord``). This assumes that the version number
    only uses at most a single char per level and only ASCII chars.
    """

    temp_list = []
    for v in ver:
        if v.isalpha():
            temp_list.append(str(ord(v.upper())))
        else:
            temp_list.append(v)
    return "".join(temp_list)
