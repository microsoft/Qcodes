from datetime import datetime
from typing import Optional, Sequence

from qcodes.metadatable import Metadatable
from qcodes.parameters import Parameter
from qcodes.utils import full_class, issue_deprecation_warning

try:
    from qcodes_loop.actions import _actions_snapshot
    from qcodes_loop.loops import Loop
    from qcodes_loop.measure import Measure
except ImportError as e:
    raise ImportError(
        "qcodes.measure is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning("qcodes.measure module", alternative="qcodes_loop.measure")
