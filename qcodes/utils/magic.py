# get_ipython is part of the public api but IPython does
# not use __all__ to mark this
from IPython import get_ipython  # type: ignore[attr-defined]
from IPython.core.magic import Magics, line_cell_magic, magics_class

from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.utils.magic import QCoDeSMagic, register_magic_class
except ImportError as e:
    raise ImportError(
        "qcodes.utils.magic is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.utils.magic module", alternative="qcodes_loop.utils.magic"
)
