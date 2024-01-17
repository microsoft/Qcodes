from qcodes.dataset.dond.do_0d import do0d
from qcodes.dataset.dond.do_1d import do1d
from qcodes.dataset.dond.do_2d import do2d
from qcodes.dataset.dond.do_nd import (
    ParameterGroup,
    _parse_dond_arguments,
    dond,
)
from qcodes.dataset.dond.do_nd_utils import (
    ActionsT,
    AxesTuple,
    AxesTupleList,
    AxesTupleListWithDataSet,
    BreakConditionInterrupt,
    BreakConditionT,
    MeasInterruptT,
    MultiAxesTupleListWithDataSet,
    ParamMeasT,
    _handle_plotting,
    _register_actions,
    _register_parameters,
    _set_write_period,
    catch_interrupts,
)
from qcodes.dataset.dond.sweeps import AbstractSweep, ArraySweep, LinSweep, LogSweep
from qcodes.dataset.plotting import plot_and_save_image as plot

# todo enable warning once new api is in release
# warnings.warn(
#     "qcodes.utils.dataset.doNd module is deprecated. "
#     "Please update to import from qcodes.dataset"
# )


class UnsafeThreadingException(Exception):
    pass


__all__ = [
    "AbstractSweep",
    "ActionsT",
    "ArraySweep",
    "AxesTuple",
    "AxesTupleList",
    "AxesTupleListWithDataSet",
    "BreakConditionInterrupt",
    "BreakConditionT",
    "LinSweep",
    "LogSweep",
    "MeasInterruptT",
    "MultiAxesTupleListWithDataSet",
    "ParamMeasT",
    "ParameterGroup",
    "UnsafeThreadingException",
    "catch_interrupts",
    "_handle_plotting",
    "_parse_dond_arguments",
    "_register_actions",
    "_register_parameters",
    "_set_write_period",
    "do0d",
    "do1d",
    "do2d",
    "dond",
    "plot",
]
