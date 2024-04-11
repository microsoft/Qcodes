from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace

from qcodes import config
from qcodes.dataset import (
    DataSetDefinition,
    LinSweep,
    LinSweeper,
    datasaver_builder,
    dond_into,
)

LOG = logging.getLogger(__name__)
TRACER = trace.get_tracer(__name__)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.dond.do_nd_utils import ActionsT
    from qcodes.dataset.experiment_container import Experiment
    from qcodes.parameters import ParameterBase


@TRACER.start_as_current_span("qcodes.dataset.do2d")
def do2d_retrace(
    param_set1: ParameterBase,
    start1: float,
    stop1: float,
    num_points1: int,
    delay1: float,
    param_set2: ParameterBase,
    start2: float,
    stop2: float,
    num_points2: int,
    delay2: float,
    *param_meas: ParameterBase,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    before_inner_actions: ActionsT = (),
    after_inner_actions: ActionsT = (),
    measurement_name: str = "",
    exp: Experiment | None = None,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
    show_progress: bool | None = None,
) -> tuple[DataSetProtocol, DataSetProtocol]:
    """
    Perform a 1D scan of ``param_set1`` from ``start1`` to ``stop1`` in
    ``num_points1`` and ``param_set2`` from ``start2`` to ``stop2`` in
    ``num_points2`` measuring param_meas at each step.

    Args:
        param_set1: The QCoDeS parameter to sweep over in the outer loop
        start1: Starting point of sweep in outer loop
        stop1: End point of sweep in the outer loop
        num_points1: Number of points to measure in the outer loop
        delay1: Delay after setting parameter in the outer loop
        param_set2: The QCoDeS parameter to sweep over in the inner loop
        start2: Starting point of sweep in inner loop
        stop2: End point of sweep in the inner loop
        num_points2: Number of points to measure in the inner loop
        delay2: Delay after setting parameter before measurement is performed
        param_meas: Parameter(s) to measure at each step or functions that
          will be called at each step. The function should take no arguments.
          The parameters and functions are called in the order they are
          supplied.
        enter_actions: A list of functions taking no arguments that will be
            called before the measurements start
        exit_actions: A list of functions taking no arguments that will be
            called after the measurements ends
        before_inner_actions: Actions executed before each run of the inner loop
        after_inner_actions: Actions executed after each run of the inner loop
        measurement_name: Name of the measurement. This will be passed down to
            the dataset produced by the measurement. If not given, a default
            value of 'results' is used for the dataset.
        exp: The experiment to use for this measurement.
        additional_setpoints: A list of setpoint parameters to be registered in
            the measurement but not scanned.
        show_progress: should a progress bar be displayed during the
            measurement. If None the setting will be read from ``qcodesrc.json``

    Returns:
        The QCoDeS dataset.
    """

    if show_progress is None:
        show_progress = config.dataset.dond_show_progress

    dataset_definition = [
        DataSetDefinition(
            name=measurement_name,
            independent=[param_set1, param_set2],
            dependent=param_meas,
            experiment=exp,
        ),
        DataSetDefinition(
            name=f"retrace {measurement_name}",
            independent=[param_set1, param_set2],
            dependent=param_meas,
            experiment=exp,
        ),
    ]

    with datasaver_builder(dataset_definition) as datasavers:
        for action in enter_actions:
            action()
        for _ in LinSweeper(param_set1, start1, stop1, num_points1, delay1):
            sweep_up = LinSweep(
                param_set2,
                start2,
                stop2,
                num_points2,
                delay2,
                post_actions=after_inner_actions,
            )
            sweep_down = LinSweep(
                param_set2,
                stop2,
                start2,
                num_points2,
                delay2,
                post_actions=after_inner_actions,
            )
            for action in before_inner_actions:
                action()
            dond_into(
                datasavers[0],
                sweep_up,
                *param_meas,
                additional_setpoints=additional_setpoints,
            )
            for action in before_inner_actions:
                action()
            dond_into(
                datasavers[1],
                sweep_down,
                *param_meas,
                additional_setpoints=additional_setpoints,
            )
            for action in exit_actions:
                action()
    datasets = (datasavers[0].dataset, datasavers[1].dataset)
    return datasets
