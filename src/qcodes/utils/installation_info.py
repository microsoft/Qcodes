"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""

import json
import logging
from importlib.metadata import distribution, distributions

log = logging.getLogger(__name__)


def is_qcodes_installed_editably() -> bool | None:
    """
    Check whether QCoDeS is installed in editable mode and return the answer
    as a boolean. Returns None if the installation metadata could not be read
    or understood.

    The answer is read from the ``direct_url.json`` file that installers such
    as pip and uv write as part of the package metadata, as specified by
    :pep:`610`. This means that no installer needs to be available at runtime.
    """

    try:
        # On Python 3.13+ ``Distribution.origin`` parses this file for us, but it
        # is typed as ``SimpleNamespace`` so everything below it becomes ``Any``,
        # and ``dir_info`` is missing entirely for VCS installs. Parsing the json
        # ourselves keeps this typed and works on all supported Python versions.
        direct_url = distribution("qcodes").read_text("direct_url.json")
        if direct_url is None:
            # No direct_url.json means that QCoDeS was installed from an index
            # (e.g. PyPI) and therefore not in editable mode.
            return False
        # A non-editable install from a local directory writes an empty
        # "dir_info", and a VCS or archive install writes none at all, so the
        # "editable" key is absent rather than false in those cases.
        dir_info = json.loads(direct_url).get("dir_info", {})
        return dir_info.get("editable", False) is True
    except Exception:
        log.exception("Could not determine if QCoDeS is installed editably")
        return None


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
