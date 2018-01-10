import qcodes
from .sweep import *


def sweep(obj, sweep_points):
    """
    A convenience function to create a 1D sweep object

    Parameters
    ----------
    obj: qcodes.StandardParameter or callable
        If callable, it shall be a callable of three parameters: station, namespace, set_value and shall return a
        dictionary
    sweep_points: iterable or callable returning a iterable
        If callable, it shall be a callable of two parameters: station, namespace and shall return an iterable

    Returns
    -------
    FunctionSweep or ParameterSweep
    """

    if not callable(sweep_points):
        point_function = lambda: sweep_points
    else:
        point_function = sweep_points

    if not isinstance(obj, qcodes.Parameter):
        if not callable(obj):
            raise ValueError("The object to sweep over needs to either be a QCoDeS parameter or a function")

        return FunctionSweep(obj, point_function)
    else:
        return ParameterSweep(obj, point_function)


def wrap_objects(*objects, repeat=False):

    def wrapper(obj):
        if isinstance(obj, qcodes.Parameter):
            new_obj = ParameterWrapper(obj, repeat=repeat)
        elif callable(obj):
            new_obj = FunctionWrapper(obj, repeat=repeat)
        else:
            new_obj = obj

        return new_obj

    return [wrapper(obj) for obj in objects]


def nest(*objects):
    return Nest(wrap_objects(*objects))


def chain(*objects):
    return Chain(wrap_objects(*objects))


def szip(*objects):
    """
    A plausible scenario for using szip is the following:

    >>> szip(therometer.t, sweep(source.voltage, [0, 1, 2]))

    The idea is to measure the temperature *before* going to each voltage set point. The parameter "thermometer.t" needs
    to be wrapped by the ParameterWrapper in such a way that the get method of the parameter is called repeatedly. An
    infinite loop is prevented because the sweep object "sweep(source.voltage, [0, 1, 2])"  has a finite length and the
    Zip operator loops until the shortest sweep object is exhausted
    """
    repeat = False
    if any([isinstance(i, BaseSweepObject) for i in objects]):  # If any of the objects is a sweep object it (probably)
        # has a finite length and therefore wrapping functions and parameters with repeat=True is save. These parameters
        # and functions will be called as often as the length of the sweep object.
        repeat = True
    return Zip(wrap_objects(*objects, repeat=repeat))
