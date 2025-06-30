import warnings

from qcodes.parameters.command import Command, NoCommandError, Output, ParsedOutput
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.utils.command` module is deprecated. Command is no longer part of the public API of QCoDeS. ",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
