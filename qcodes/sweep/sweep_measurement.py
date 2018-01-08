"""
The sweep measurement class that is designed to be used with sweep objects. Since with sweep objects we can
automatically detect dependent and independent parameters, there is no need to register these explicitly

Intended usage:

>>> from qcodes import new_experiment
>>> from qcodes.sweep import sweep, SweepMeasurement
>>> experiment = new_experiment()
>>> with SweepMeasurement(experiment).run() as data_saver:
>>>     for result_dict in sweep(obj, values):
>>>         data_saver.add(result_dict)
>>>


"""
from typing import List

from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.measurements import DataSaver, Measurement, Runner
from qcodes.dataset.data_set import DataSet


class SweepDataSaver(DataSaver):
    def __init__(self, dataset: DataSet, write_period: float,
                 known_parameters: List[str]) -> None:

        super().__init__(dataset, write_period, known_parameters)
        self._parameters = set()  # Parameter names encountered during the run

    def _register_parameters_in_result_dict(self, result_dict):
        # Make sure all parameters in a data line are registered in the QCoDeS data set
        # first find all independent parameters in the data line and make sure they are registered
        independent_parameters = []
        dependent_parameters = []

        for parameter_name in result_dict.keys():
            if result_dict[parameter_name]["independent_parameter"]:
                independent_parameters.append(parameter_name)

                if parameter_name not in self._parameters:
                    unit = result_dict[parameter_name]["unit"]
                    self._add_to_data_set(parameter_name, unit, [])
            else:
                dependent_parameters.append(parameter_name)

        # Then process all dependent parameters
        for parameter_name in dependent_parameters:
            if parameter_name not in self._parameters:
                unit = result_dict[parameter_name]["unit"]
                self._add_to_data_set(parameter_name, unit, independent_parameters)

    def _add_to_data_set(self, parameter_name, unit, depends_on):
        ty = "number"  # for now
        param_spec = ParamSpec(parameter_name, ty, depends_on=depends_on, unit=unit)
        self._dataset.add_parameters([param_spec])
        self._parameters.add(parameter_name)

    def addResult(self, result_dict):  # In the sweep version of the data saver we expect dictionaries
        self._register_parameters_in_result_dict(result_dict)

        result_list = [(name, result_dict[name]["value"]) for name in result_dict.keys()]
        super().add_result(*result_list)


class SweepMeasurement(Measurement):
    def run(self):
        """
        Returns the context manager for the experimental run
        """
        return Runner(
            self.enteractions, self.exitactions,
            self.experiment, parameters=self.parameters,
            saver_class=SweepDataSaver
        )
