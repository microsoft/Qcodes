from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_base import (get_dependencies, get_dependents,
                                        get_layout)

DB = qc.config["core"]["db_location"]


def flatten_data_for_plot(rawdata: List[List[Any]]) -> np.ndarray:
    """
    Cast the return value of the dataset get_data function to
    a numpy array
    """
    dataarray = np.array(rawdata)
    shape = np.shape(dataarray)
    dataarray = dataarray.reshape(np.product(shape))

    return dataarray


def plot_by_id(run_id: int) -> None:
    """
    Construct all plots for a given run

    Only 1D plots implemented
    """

    conn = DataSet(DB).conn

    data = qc.load_by_id(run_id)
    deps = get_dependents(conn, run_id)

    for dep in deps:
        recipe = get_dependencies(conn, dep)

        if len(recipe) == 1:
            # get plotting info
            first_axis_layout = get_layout(conn, recipe[0][0])
            first_axis_name = first_axis_layout['name']
            first_axis_label = first_axis_layout['label']
            first_axis_unit = first_axis_layout['unit']
            first_axis_data = flatten_data_for_plot(data.get_data(first_axis_name))

            second_axis_layout = get_layout(conn, dep)
            second_axis_name = second_axis_layout['name']
            second_axis_label = second_axis_layout['label']
            second_axis_unit = second_axis_layout['unit']
            second_axis_data = flatten_data_for_plot(data.get_data(second_axis_name))

            # perform plotting
            fig, ax = plt.subplots()

            ax.plot(first_axis_data, second_axis_data)

            if first_axis_label == '':
                lbl = first_axis_name
            else:
                lbl = first_axis_label
            if first_axis_unit == '':
                unit = ''
            else:
                unit = f'({first_axis_unit})'
            ax.set_xlabel(f'{lbl} {unit}')

            if second_axis_label == '':
                lbl = second_axis_name
            else:
                lbl = second_axis_label
            if second_axis_unit == '':
                unit = ''
            else:
                unit = f'({second_axis_unit})'
            ax.set_ylabel(f'{lbl} {unit}')
