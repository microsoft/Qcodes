"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

if sys.version_info >= (3, 8):
    from importlib.metadata import PackageNotFoundError, distribution, version
else:
    # 3.7 and earlier
    from importlib_metadata import PackageNotFoundError, distribution, version

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
    import qcodes
    package_name = "qcodes"

    qcodes_path = Path(qcodes.__file__).parent
    if _has_pyproject_toml_and_is_git_repo(qcodes_path.parent):
        log.info(
            f"QCoDeS seems to be installed editably trying to look up version "
            f"from git repo in {qcodes_path}"
        )

        import versioningit

        __version__ = versioningit.get_version(project_dir=qcodes_path.parent)
    else:
        __version__ = version(package_name)
    return __version__


def get_qcodes_requirements() -> List[str]:
    """
    Return a list of the names of the packages that QCoDeS requires
    """
    import pkg_resources
    qc_pkg = distribution('qcodes').requires
    if qc_pkg is None:
        return []
    package_names = [pkg_resources.Requirement.parse(req).unsafe_name for req in qc_pkg]

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
    import pkg_resources
    packages = pkg_resources.working_set
    return {i.key: i.version for i in packages}


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


def _has_pyproject_toml_and_is_git_repo(path: Path) -> bool:
    has_pyproject_toml = (path / "pyproject.toml").exists()
    is_git_repo = (path / ".git").exists()
    return has_pyproject_toml and is_git_repo
