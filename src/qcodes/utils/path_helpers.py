import os

# files added to stdlib in 3.9
from importlib.resources import files
from pathlib import Path

QCODES_USER_PATH_ENV = "QCODES_USER_PATH"


def get_qcodes_path(*subfolder: str) -> str:
    """
    Return full file path of the QCoDeS module. Additional arguments will be
    appended as subfolder.

    """
    return str(files("qcodes").joinpath(*subfolder)) + os.sep


def get_qcodes_user_path(*file_parts: str) -> str:
    """
    Get ``~/.qcodes`` path or if defined the path defined in the
    ``QCODES_USER_PATH`` environment variable.

    Returns:
        path to the user qcodes directory

    """
    path = os.environ.get(QCODES_USER_PATH_ENV, os.path.join(Path.home(), ".qcodes"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return os.path.join(path, *file_parts)
