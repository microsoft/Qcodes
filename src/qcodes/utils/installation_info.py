"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""

import json
import logging
import subprocess
from importlib.metadata import distributions

log = logging.getLogger(__name__)


def is_qcodes_installed_editably() -> bool | None:
    """
    Try to ask pip whether QCoDeS is installed in editable mode and return
    the answer a boolean. Returns None if pip somehow did not respond as
    expected.
    """

    answer: bool | None

    try:
        pipproc = subprocess.run(
            ["python", "-m", "pip", "list", "-e", "--no-index", "--format=json"],
            check=True,
            stdout=subprocess.PIPE,
        )
        e_pkgs = json.loads(pipproc.stdout.decode("utf-8"))
        answer = any([d["name"] == "qcodes" for d in e_pkgs])
    except Exception as e:  # we actually do want a catch-all here
        log.warning(f"{type(e)}: {e!s}")
        answer = None

    return answer


def get_all_installed_package_versions() -> dict[str, str]:
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

    It also splits off anything that comes after the first ``-`` in the version str.

    This is meant to pass versions like ``'A.02.17-02.40-02.17-00.52-04-01'``
    primarily used by Keysight instruments.
    """

    temp_list = []
    for v in ver:
        if v.isalpha():
            temp_list.append(str(ord(v.upper())))
        else:
            temp_list.append(v)
    temp_str = "".join(temp_list)
    return temp_str.split("-")[0]
