"""
A measurement subclass which is designed to work specifically with sweep
objects. It simply adds a method to register parameters implicitly defined
in sweep objects.
"""

from collections import OrderedDict
import itertools

from qcodes.dataset.measurements import Measurement
from qcodes import ParamSpec


class SweepMeasurement(Measurement):
    def register_sweep(self, sweep_object):

        independent_parameters = []
        dependent_parameters = []

        for parameters_dict in sweep_object.parameter_table.table_list:
            local_independents = [
                ParamSpec(
                    name=name,
                    paramtype='numeric',
                    unit=unit
                )
                for name, unit in parameters_dict["independent_parameters"]
            ]

            dependent_parameters.extend([
                ParamSpec(
                    name=name,
                    paramtype='numeric',
                    unit=unit,
                    depends_on=local_independents
                )
                for name, unit in parameters_dict["dependent_parameters"]
            ])

            independent_parameters.extend(local_independents)

        if len(set(dependent_parameters)) != len(dependent_parameters):
            raise RuntimeError("Duplicate dependent parameters detected!")

        self.parameters = OrderedDict({
            spec.name: spec for spec in itertools.chain(
                independent_parameters, dependent_parameters
            )
        })
