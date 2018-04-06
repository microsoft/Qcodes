"""
This module defines convenience functions so that end users can easily create
sweep objects.
"""

import numpy as np
from typing import Callable, Iterable, Union, Sized

import qcodes
from qcodes import Parameter
from qcodes.sweep.base_sweep import ParametersTable, BaseSweepObject, \
    FunctionSweep, ParameterSweep, Nest, wrap_objects, TimeTrace, Chain, Zip


sw_objects = Union[Callable, Parameter, BaseSweepObject]

def _infer_axis_properties(axis, length_only=False):
    class _Dict(dict):
        def dset(self, name, value):
            self[name] = value
            return self

    properties = dict(ParametersTable.default_axis_properties)

    if not hasattr(axis, "__len__"):
        return [properties]

    properties["length"] = len(axis)

    array = np.array(axis)
    if len(array.shape) == 1:
        array = array[:, None]

    properties = [_Dict(properties) for _ in array.T]

    if array.dtype == np.dtype("O") or length_only:
        return properties

    axis_min = np.min(array, axis=0)
    axis_max = np.max(array, axis=0)

    steps = [
        np.unique(i)
        for i in zip(*np.diff(array, axis=0).round(decimals=10))
    ]

    steps = [step[0] if len(step) == 1 else "?" for step in steps]

    return [
        p.dset("min", mn).dset("max", mx).dset("steps", s)
        for p, mn, mx, s in zip(properties, axis_min, axis_max, steps)
    ]


def sweep(
        obj: Union[Parameter, Callable],
        sweep_points: Union[Sized, Iterable, Callable]
)->BaseSweepObject:
    """
    A convenience function to create a 1D sweep object

    Args:
        obj (Parameter or callable):
            If callable, a function decorated with setter
        sweep_points (iterable or callable):
            If callable, it shall be a callable of no parameters

    Returns:
        FunctionSweep or ParameterSweep
    """

    if not callable(sweep_points):
        point_function = lambda: sweep_points
    else:
        point_function = sweep_points
    so: BaseSweepObject
    if not isinstance(obj, qcodes.Parameter):
        if not callable(obj):
            raise ValueError(
                "The object to sweep over needs to either be a QCoDeS "
                "parameter or a function"
            )

        so = FunctionSweep(obj, point_function)
    else:
        so = ParameterSweep(obj, point_function)

    ind = so.parameter_table.get_independents(exclude_inferees=True)
    has_inferred = True if len(so.parameter_table.inferred_symbols_list()) \
        else False

    axis_properties = _infer_axis_properties(
        sweep_points, length_only=has_inferred
    )

    axis_properties = {
        name: props for name, props in zip(ind, axis_properties)
    }
    so.parameter_table.set_axis_info(axis_properties)

    return so


def nest(*objects: BaseSweepObject)->BaseSweepObject:
    """
    Convenience function to create a nested sweep

    Args:
        objects (list): List of Sweep object to nest

    Returns:
        Nested sweep object
    """
    return Nest(wrap_objects(*objects))


def chain(*objects:  BaseSweepObject)->BaseSweepObject:
    """
    Convenience function to create a chained sweep

    Args:
        objects (list): List of Sweep object to chain

    Returns:
        Chained sweep object
    """
    return Chain(wrap_objects(*objects))


def szip(*objects:  sw_objects)->BaseSweepObject:
    """
    A plausible scenario for using szip is the following:

    >>> szip(therometer.t, sweep(source.voltage, [0, 1, 2]))

    The idea is to measure the temperature *before* going to each voltage set
    point. The parameter "thermometer.t" needs to be wrapped by the
    ParameterWrapper in such a way that the get method of the parameter is
    called repeatedly. An infinite loop is prevented because
    "sweep(source.voltage, [0, 1, 2])"  has a finite length and the
    Zip operator loops until the shortest sweep object is exhausted

    Args:
        objects (list): List of Sweep object to zip

    Returns:
        Zipped sweep object
    """
    repeat = False
    if any([isinstance(i, BaseSweepObject) for i in objects]):
        repeat = True

    return Zip(wrap_objects(*objects, repeat=repeat))


def time_trace(
        measurement_object: sw_objects,
        interval_time: float,
        total_time: float
):
    """
    Make time trace sweep object to monitor the return value of the measurement
    object over a certain time period.

    Args:
        measurement_object:
            This can be; a function decorated with the getter decorator,
            a QCoDeS parameter, or another sweep object
        interval_time (float)
        total_time (float)
    """
    tt_sweep = TimeTrace(interval_time, total_time)
    return szip(measurement_object, tt_sweep)
