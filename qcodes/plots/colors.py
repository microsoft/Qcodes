from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.plots.colors import (
        color_cycle,
        colorscales,
        colorscales_raw,
        make_rgba,
        one_rgba,
    )
except ImportError as e:
    raise ImportError(
        "qcodes.plots.colors is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning("qcodes.plots.colors", alternative="qcodes_loop.plots.colors")
