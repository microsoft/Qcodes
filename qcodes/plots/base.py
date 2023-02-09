"""
Deprecated
"""
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.plots.base import BasePlot
except ImportError as e:
    raise ImportError(
        "qcodes.plots.base is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning("qcodes.plots.base", alternative="qcodes_loop.plots.base")
