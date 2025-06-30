import warnings

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.utils.dataset` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
