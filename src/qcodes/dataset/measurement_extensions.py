from __future__ import annotations

import time
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from qcodes.dataset.dond.do_nd import _Sweeper
from qcodes.dataset.dond.do_nd_utils import ParamMeasT, catch_interrupts
from qcodes.dataset.dond.sweeps import AbstractSweep, LinSweep, TogetherSweep
from qcodes.dataset.measurements import DataSaver, Measurement
from qcodes.dataset.threading import process_params_meas
from qcodes.parameters.parameter_base import ParameterBase

if TYPE_CHECKING:
    from qcodes.dataset.experiment_container import Experiment

TRACER = trace.get_tracer(__name__)


@dataclass
class DataSetDefinition:
    """
    A specification for the creation of a Dataset or Measurement object
    """

    name: str
    """The name to be assigned to the Measurement and dataset"""
    independent: Sequence[ParameterBase]
    """A sequence of independent parameters in the Measurement and dataset"""
    dependent: Sequence[ParameterBase]
    """A sequence of dependent parameters in the Measurement and dataset
    Note: All dependent parameters will depend on all independent parameters"""
    experiment: Experiment | None = None
    """An optional argument specifying which Experiment this dataset should be
    written to"""


def setup_measurement_instances(
    dataset_definitions: Sequence[DataSetDefinition],
    override_experiment: Experiment | None = None,
) -> list[Measurement]:
    """Creates a set of Measurement instances and registers parameters

    Args:
        dataset_definitions: A set of DataSetDefinitions to create and register
            parameters for
        override_experiment: The Experiment all Measurement objects will be part of

    Returns:
        A list of Measurement objects
    """
    measurements: list[Measurement] = []
    for ds_def in dataset_definitions:
        ds_experiment = (
            override_experiment
            if override_experiment is not None
            else ds_def.experiment
        )
        meas = Measurement(name=ds_def.name, exp=ds_experiment)
        for param in ds_def.independent:
            meas.register_parameter(param)
        for param in ds_def.dependent:
            meas.register_parameter(param, setpoints=ds_def.independent)
        measurements.append(meas)
    return measurements


@contextmanager
def datasaver_builder(
    dataset_definitions: Sequence[DataSetDefinition],
    *,
    override_experiment: Experiment | None = None,
) -> Generator[list[DataSaver], Any, None]:
    """
    A utility context manager intended to simplify the creation of datasavers

    The datasaver builder can be used to streamline the creation of multiple datasavers
    where all dependent parameters depend on all independent parameters.

    Args:
        dataset_definitions: A set of DataSetDefinitions to create and register
            parameters for
        override_experiment: Sets the Experiment for all datasets to be written
            to. This argument overrides any experiments provided in the
            DataSetDefinition

    Yields:
        A list of generated datasavers with parameters registered
    """

    measurement_instances = setup_measurement_instances(
        dataset_definitions, override_experiment
    )
    with (
        TRACER.start_as_current_span("qcodes.dataset.datasaver_builder"),
        catch_interrupts() as _,
        ExitStack() as stack,
    ):
        datasaver_builder_span = trace.get_current_span()
        datasavers = [
            stack.enter_context(measurement.run(parent_span=datasaver_builder_span))
            for measurement in measurement_instances
        ]
        yield datasavers


def parse_dond_into_args(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
) -> tuple[list[AbstractSweep], list[ParamMeasT]]:
    """
    Parse supplied arguments into sweeps and measurement parameters

    Measurement parameters may include Callables which are executed in order

    Args:
        params: Instances of n sweep classes and m measurement parameters or
            callables

    Returns:

    """
    sweep_instances: list[AbstractSweep] = []
    params_meas: list[ParamMeasT] = []
    for par in params:
        if isinstance(par, AbstractSweep):
            sweep_instances.append(par)
        elif isinstance(par, TogetherSweep):
            raise ValueError("dond_into does not support TogetherSweeps")
        elif isinstance(par, Sequence):
            raise ValueError("dond_into does not support multiple datasets")
        elif isinstance(par, ParameterBase) and par.gettable:
            params_meas.append(par)
        elif callable(par):
            params_meas.append(par)
    return sweep_instances, params_meas


def dond_into(
    datasaver: DataSaver,
    *params: AbstractSweep | ParamMeasT,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
) -> None:
    """
    A doNd-like utility function that writes gridded data to the supplied DataSaver

    dond_into accepts AbstractSweep objects and measurement parameters or callables.
    It executes the specified Sweeps, reads the measurement parameters, and stores
    the resulting data in the datasaver.

    Args:
        datasaver: The datasaver to write data to
        params: Instances of n sweep classes and m measurement parameters,
            e.g. if linear sweep is considered:

            .. code-block:: python

                LinSweep(param_set_1, start_1, stop_1, num_points_1, delay_1), ...,
                LinSweep(param_set_n, start_n, stop_n, num_points_n, delay_n),
                param_meas_1, param_meas_2, ..., param_meas_m

        additional_setpoints: A list of setpoint parameters to be registered in the
            measurement but not scanned/swept-over.
    """
    # at this stage multiple measurement context managers may be in run state
    # as datasavers. Here we ensure we bind the parent span to the correct
    # datasaver.
    if datasaver._span is not None:
        context = trace.set_span_in_context(datasaver._span)
    else:
        context = None

    with TRACER.start_as_current_span("qcodes.dataset.dond_into", context=context):
        sweep_instances, params_meas = parse_dond_into_args(*params)
        sweeper = _Sweeper(sweep_instances, additional_setpoints)
        for set_events in sweeper:
            results: dict[ParameterBase, Any] = {}
            additional_setpoints_data = process_params_meas(additional_setpoints)
            for set_event in set_events:
                if set_event.should_set:
                    set_event.parameter(set_event.new_value)
                    for act in set_event.actions:
                        act()
                    time.sleep(set_event.delay)

                if set_event.get_after_set:
                    results[set_event.parameter] = set_event.parameter()
                else:
                    results[set_event.parameter] = set_event.new_value

            meas_value_pair = process_params_meas(params_meas)
            for meas_param, value in meas_value_pair:
                results[meas_param] = value

            datasaver.add_result(
                *list(results.items()),
                *additional_setpoints_data,
            )


class LinSweeper(LinSweep):
    """
    An iterable version of the LinSweep class

    Iterations of this object, set the next setpoint and then wait the delay time
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._setpoints = self.get_setpoints()
        self._iter_index = 0

    def __iter__(self) -> LinSweeper:
        return self

    def __next__(self) -> float:
        if self._iter_index < self._num_points:
            set_val = self._setpoints[self._iter_index]
            self._param(set_val)
            time.sleep(self._delay)
            self._iter_index += 1
            return set_val
        else:
            raise StopIteration
