from typing import List, Any, Dict, Generator, Sequence, TypedDict
import time

from contextlib import ExitStack, contextmanager


from qcodes.parameters.parameter_base import ParameterBase
from qcodes.dataset.dond.do_nd_utils import _catch_interrupts
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.measurements import Measurement, DataSaver
from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.dataset.dond.do_nd import _parse_dond_arguments, _Sweeper
from qcodes.dataset.threading import _call_params, process_params_meas


class DataSetDefinition(TypedDict):
    name: str
    independent: Sequence[ParameterBase]
    dependent: Sequence[ParameterBase]


def setup_measurement_instances(
    dataset_definition: DataSetDefinition, experiment: Experiment
) -> List[Measurement]:
    measurements: List[Measurement] = []
    for dataset_name in dataset_definition.keys():
        meas = Measurement(name=dataset_name, exp=experiment)
        indep_params = dataset_definition[dataset_name]["independent"]
        dep_params = dataset_definition[dataset_name]["dependent"]
        for param in indep_params:
            meas.register_parameter(param)
        for param in dep_params:
            meas.register_parameter(param, setpoints=indep_params)
        measurements.append(meas)
    return measurements


@contextmanager
def complex_measurement_context(
    dataset_definition: Dict[str, Any], experiment: Experiment
) -> Generator[list[DataSaver], Any, None]:
    measurement_instances = setup_measurement_instances(dataset_definition, experiment)
    with _catch_interrupts() as interrupted, ExitStack() as stack:
        datasavers = [
            stack.enter_context(measurement.run())
            for measurement in measurement_instances
        ]
        try:
            yield datasavers
        except Exception:
            raise Exception


def dond_core(
    datasaver: DataSaver,
    *params: ParameterBase,
    additional_setpoints: Sequence[ParameterBase] = tuple(),
) -> None:
    sweep_instances, params_meas = _parse_dond_arguments(*params)
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

        meas_value_pair = _call_params(params_meas)
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
