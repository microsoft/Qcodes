import os
import re
import shutil
from contextlib import contextmanager
from fnmatch import fnmatch

from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.data.io import ALLOWED_OPEN_MODES, DiskIO
except ImportError as e:
    raise ImportError(
        "qcodes.data.io is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning("qcodes.data.io module", alternative="qcodes_loop.data.io")
