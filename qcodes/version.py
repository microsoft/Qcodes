import warnings

import qcodes
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

__version__ = qcodes.__version__

warnings.warn(
    "The qcodes.version module is deprecated and will be removed, Please use `qcodes.__version__` to "
    "get the QCoDeS version at runtime",
    QCoDeSDeprecationWarning,
)
