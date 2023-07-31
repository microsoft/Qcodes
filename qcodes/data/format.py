import logging
from collections import namedtuple
from operator import attrgetter
from traceback import format_exc
from typing import Set

from qcodes.utils import issue_deprecation_warning

log = logging.getLogger(__name__)

try:
    from qcodes_loop.data.format import ArrayGroup, Formatter
except ImportError as e:
    raise ImportError(
        "qcodes.data.format is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.data.format module", alternative="qcodes_loop.data.format"
)
