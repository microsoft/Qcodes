"""
A measurement subclass which is designed to work specifically with sweep
objects. It simply adds a method to register parameters implicitly defined
in sweep objects.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict
import itertools

from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
from qcodes.sweep.base_sweep import BaseSweepObject
from qcodes import ParamSpec


class SweepMeasurement(Measurement):

    @staticmethod
    def _make_param_spec_list(
            symbols_list: List[Tuple],
            inferred_parameters: Dict,
            depends_on: List=None
    )->List[ParamSpec]:
        """
        Args:
            symbols_list (list):
                A list of tuples (<name>, <unit>) where the name and unit are
                strings.
            inferred_parameters (dict):
                The keys in the dictionary are the symbols which are inferred.
                The values are lists of symbol names from which we perform the
                inference (e.g. {A: [B, C]} means A is inferred from B and C)
            depends_on (list):
                A list of symbol names which all symbols in the symbols_list
                depend on

        Returns:
            A list of ParamSpec objects
        """
        param_spec_dict:Dict[str, ParamSpec] = {}

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

            param_spec_dict[name] = ParamSpec(
                name=name,
                paramtype='numeric',
                unit=unit,
                depends_on=depends_on,
                inferred_from=[
                    param_spec_dict[n]
                    for n in inferred_parameters.get(name, [])
                ]
            )

        return list(param_spec_dict.values())

    def register_sweep(self, sweep_object: BaseSweepObject)->None:
        """
        Args:
            sweep_object:
                Make and add param specs from the parameters table in the
                sweep object
        """

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
            dependency_black_list: List[str] = []
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


class _DataExtractor:
    """
    A convenience class to quickly extract data from a data saver instance
    """
    def __init__(self, datasaver):
        self._run_id = datasaver.run_id
        self._dataset = datasaver.dataset

    def __getitem__(self, layout):

        def is_subset(smaller, larger):
            return smaller == larger[:len(smaller)]

        layout = sorted(layout.split(","))
        all_data = get_data_by_id(self._run_id)
        data_layouts = [sorted([d["name"] for d in ad]) for ad in all_data]

        i = np.array(
            [is_subset(layout, data_layout) for data_layout in data_layouts]
        )

        ind = np.flatnonzero(i)
        if len(ind) == 0:
            raise ValueError(f"No such layout {layout}")

        data = all_data[ind[0]]
        return {d["name"]: d["data"] for d in data}

    def plot(self):
        plot_by_id(self._run_id)

    @property
    def run_id(self):
        return self._run_id


def run(setup, sweep_object, cleanup, experiment=None, station=None):

    meas = SweepMeasurement(exp=experiment, station=station)
    meas.register_sweep(sweep_object)

    for f, args in setup:
        meas.add_before_run(f, args)

    for f, args in cleanup:
        meas.add_after_run(f, args)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(*data.items())

    return _DataExtractor(datasaver)
