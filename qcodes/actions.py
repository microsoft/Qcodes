"""Actions, mainly to be executed in measurement Loops."""
import time

from qcodes.utils import is_function, issue_deprecation_warning, thread_map

try:
    from qcodes_loop.actions import (
        BreakIf,
        Task,
        UnsafeThreadingException,
        Wait,
        _actions_snapshot,
        _Measure,
        _Nest,
        _QcodesBreak,
    )

except ImportError as e:
    raise ImportError(
        "qcodes.actions is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e

issue_deprecation_warning("qcodes.actions module", alternative="qcodes_loop.actions")
