from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.utils.qt_helpers import foreground_qt_window
except ImportError as e:
    raise ImportError(
        "qcodes.utils.qt_helpers is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e

issue_deprecation_warning(
    "qcodes.utils.qt_helpers module", alternative="qcodes_loop.utils.qt_helpers"
)
