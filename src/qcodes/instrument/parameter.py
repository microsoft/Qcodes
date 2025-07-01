import warnings

from qcodes.parameters import (
    ArrayParameter,
    CombinedParameter,
    DelegateParameter,
    InstrumentRefParameter,
    ManualParameter,
    MultiParameter,
    ParamDataType,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
    ScaledParameter,
    SweepFixedValues,
    combine,
    expand_setpoints_helper,
    invert_val_mapping,
)
from qcodes.parameters import ParameterBase as _BaseParameter
from qcodes.parameters.parameter_base import GetLatest
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument.parameter` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
