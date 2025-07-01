import warnings

from qcodes.parameters import ElapsedTimeParameter
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument.specialized_parameters` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
