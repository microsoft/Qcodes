import warnings

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

warnings.warn(
    "The qcodes.version module is deprecated and will be removed, Please use `qcodes.__version__` to "
    "get the QCoDeS version at runtime",
    QCoDeSDeprecationWarning,
)
