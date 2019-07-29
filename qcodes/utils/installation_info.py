"""
This module contains helper functions that provide information about how
QCoDeS is installed and about what other packages are installed along with
QCoDeS
"""
from typing import Dict, List
import subprocess


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


def is_qcodes_installed_editably() -> bool:
    """
    Return a boolean with the answer to the question 'is QCoDeS installed in
    editable mode?'
    """
    pipproc = subprocess.run(['pip', 'list', '-e'], stdout=subprocess.PIPE)
    lines = pipproc.stdout.decode('utf-8').split('\r\n')[:-1]

    editable_packages = _pip_list_parser(lines)

    return 'qcodes' in editable_packages
