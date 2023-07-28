from __future__ import annotations
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Generator, List, Sequence, Callable

from qcodes.dataset.dond.do_nd import _Sweeper
from qcodes.dataset.dond.do_nd_utils import ParamMeasT, catch_interrupts
from qcodes.dataset.dond.sweeps import AbstractSweep, LinSweep, TogetherSweep
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.measurements import DataSaver, Measurement
from qcodes.dataset.threading import process_params_meas
from qcodes.parameters.parameter_base import ParameterBase


@dataclass
class DataSetDefinition:
    name: str
    independent: Sequence[ParameterBase]
    dependent: Sequence[ParameterBase]


def setup_measurement_instances(
    dataset_definitions: Sequence[DataSetDefinition], experiment: Experiment
) -> List[Measurement]:
    measurements: List[Measurement] = []
    for ds_def in dataset_definitions:
        meas = Measurement(name=ds_def.name, exp=experiment)
        for param in ds_def.independent:
            meas.register_parameter(param)
        for param in ds_def.dependent:
            meas.register_parameter(param, setpoints=ds_def.independent)
        measurements.append(meas)
    return measurements


@contextmanager
def datasaver_builder(
    dataset_definitions: Sequence[DataSetDefinition], experiment: Experiment
) -> Generator[list[DataSaver], Any, None]:
    measurement_instances = setup_measurement_instances(dataset_definitions, experiment)
    with catch_interrupts() as interrupted, ExitStack() as stack:
        datasavers = [
            stack.enter_context(measurement.run())
            for measurement in measurement_instances
        ]
        try:
            yield datasavers
        except Exception:
            raise Exception


def parse_dond_core_args(
    *params: AbstractSweep | TogetherSweep | ParamMeasT | Sequence[ParamMeasT],
) -> tuple[list[AbstractSweep], list[ParamMeasT]]:
    """
    Parse supplied arguments into sweep objects and measurement parameters
    and their callables.
    """
    sweep_instances: list[AbstractSweep] = []
    params_meas: list[ParamMeasT] = []
    for par in params:
        if isinstance(par, AbstractSweep):
            sweep_instances.append(par)
        elif isinstance(par, TogetherSweep):
            raise ValueError("dond_core does not support TogetherSweeps")
        elif isinstance(par, Sequence):
            raise ValueError("dond_core does not support multiple datasets")
        elif isinstance(par, ParameterBase) and par.gettable:
            params_meas.append(par)
        elif isinstance(par, Callable):
            params_meas.append(par)
    return sweep_instances, params_meas


def dond_core(
    datasaver: DataSaver,
    *params: ParameterBase,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
) -> None:
    sweep_instances, params_meas = parse_dond_core_args(*params)
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

        filtered_results_list = [(param, value) for param, value in results.items()]
        datasaver.add_result(
            *filtered_results_list,
            *additional_setpoints_data,
        )


class LinSweeper(LinSweep):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._setpoints = self.get_setpoints()
        self._iter_index = 0

    def __iter__(self) -> "LinSweeper":
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
