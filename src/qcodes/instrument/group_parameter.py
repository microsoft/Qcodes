import warnings

from qcodes.parameters import Group, GroupParameter
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument.group_parameter` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
