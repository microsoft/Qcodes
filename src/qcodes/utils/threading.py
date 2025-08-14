"""Module left for backwards compatibility. Please do not import
in new code"""

import warnings

from qcodes.dataset.threading import (
    SequentialParamsCaller,
    ThreadPoolParamsCaller,
    _call_params,
    _instrument_to_param,
    _ParamCaller,
    _ParamsCallerProtocol,
    call_params_threaded,
    process_params_meas,
)
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .threading_utils import RespondingThread, thread_map

warnings.warn(
    "The `qcodes.utils.threading` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
