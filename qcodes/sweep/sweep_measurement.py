"""
A measurement subclass which is designed to work specifically with sweep
objects. It simply adds a method to register parameters implicitly defined
in sweep objects.
"""

from collections import OrderedDict, defaultdict
import itertools

from qcodes.dataset.measurements import Measurement
from qcodes import ParamSpec


class SweepMeasurement(Measurement):

    @staticmethod
    def _make_param_spec_list(symbols_list, inferred_parameters,
                              depends_on=None):

        param_spec_list = {}

        if depends_on is None:
            depends_on = []

        inferred_or_not = [
            it[0] in inferred_parameters.keys() for it in symbols_list
        ]

        sorted_symbols = sorted(
            symbols_list,
            key=lambda v: inferred_or_not[symbols_list.index(v)]
        )

        # We have sorted the symbols such that those that are not inferred
        # are encountered first in the loop
        for name, unit in sorted_symbols:

            param_spec_list[name] = ParamSpec(
                name=name,
                paramtype='numeric',
                unit=unit,
                depends_on=depends_on,
                inferred_from=[
                    param_spec_list[n]
                    for n in inferred_parameters.get(name, [])
                ]
            )

        return list(param_spec_list.values())

    def register_sweep(self, sweep_object):

        table = sweep_object.parameter_table
        inferred_parameters = table.inferred_from_dict

        independent_parameters = []
        dependent_parameters = []

        for table_list in table.table_list:

            local_independents = self._make_param_spec_list(
                table_list["independent_parameters"],
                inferred_parameters
            )

            # Independent parameters that are used to infer other independent
            # parameters never occur in the "depends_on" list
            # Thus if a voltage x is inferred from xmv, then a current y
            # which is measured is said to depend on x only, even though
            # xmv is also independent
            dependency_black_list = []
            for param in inferred_parameters.values():
                dependency_black_list += param

            dependents = self._make_param_spec_list(
                table_list["dependent_parameters"],
                inferred_parameters,
                depends_on=[i for i in local_independents
                            if i.name not in dependency_black_list]
            )

            independent_parameters.extend(local_independents)
            dependent_parameters.extend(dependents)

        if len(set(dependent_parameters)) != len(dependent_parameters):
            raise RuntimeError("Duplicate dependent parameters detected!")

        self.parameters = OrderedDict({
            spec.name: spec for spec in itertools.chain(
                independent_parameters, dependent_parameters
            )
        })

