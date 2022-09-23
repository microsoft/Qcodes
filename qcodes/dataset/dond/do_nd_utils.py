from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.colorbar

from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_and_save_image
from qcodes.parameters import ParameterBase

ActionsT = Sequence[Callable[[], None]]
BreakConditionT = Callable[[], bool]

ParamMeasT = Union[ParameterBase, Callable[[], None]]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[
    List[matplotlib.axes.Axes], List[Optional[matplotlib.colorbar.Colorbar]]
]
AxesTupleListWithDataSet = Tuple[
    DataSetProtocol,
    List[matplotlib.axes.Axes],
    List[Optional[matplotlib.colorbar.Colorbar]],
]
MultiAxesTupleListWithDataSet = Tuple[
    Tuple[DataSetProtocol, ...],
    Tuple[List[matplotlib.axes.Axes], ...],
    Tuple[List[Optional[matplotlib.colorbar.Colorbar]], ...],
]


class BreakConditionInterrupt(Exception):
    pass


MeasInterruptT = Union[KeyboardInterrupt, BreakConditionInterrupt, None]


def _register_parameters(
    meas: Measurement,
    param_meas: Sequence[ParamMeasT],
    setpoints: Sequence[ParameterBase] | None = None,
    shapes: Shapes = None,
) -> None:
    for parameter in param_meas:
        if isinstance(parameter, ParameterBase):
            meas.register_parameter(parameter, setpoints=setpoints)
    meas.set_shapes(shapes=shapes)


def _set_write_period(meas: Measurement, write_period: float | None = None) -> None:
    if write_period is not None:
        meas.write_period = write_period


def _handle_plotting(
    data: DataSetProtocol,
    do_plot: bool = True,
    interrupted: MeasInterruptT = None,
) -> AxesTupleListWithDataSet:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        data: a dataset to generate plots from
            as plot.
        do_plot: Should a plot be produced

    """
    if do_plot:
        res = plot_and_save_image(data)
    else:
        res = data, [None], [None]

    if interrupted:
        raise interrupted

    return res


def _register_actions(
    meas: Measurement, enter_actions: ActionsT, exit_actions: ActionsT
) -> None:
    for action in enter_actions:
        # this omits the possibility of passing
        # argument to enter and exit actions.
        # Do we want that?
        meas.add_before_run(action, ())
    for action in exit_actions:
        meas.add_after_run(action, ())


@contextmanager
def _catch_interrupts() -> Iterator[Callable[[], MeasInterruptT]]:
    interrupt_exception = None

    def get_interrupt_exception() -> MeasInterruptT:
        nonlocal interrupt_exception
        return interrupt_exception

    try:
        yield get_interrupt_exception
    except (KeyboardInterrupt, BreakConditionInterrupt) as e:
        interrupt_exception = e
