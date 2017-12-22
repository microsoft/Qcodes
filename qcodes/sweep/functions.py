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


def wrap_objects(*objects):

    def wrapper(obj):
        if isinstance(obj, qcodes.Parameter):
            new_obj = ParameterWrapper(obj)
        elif callable(obj):
            new_obj = FunctionWrapper(obj)
        else:
            new_obj = obj

        return new_obj

    return [wrapper(obj) for obj in objects]


def nest(*objects):
    return Nest(wrap_objects(*objects))


def chain(*objects):
    return Chain(wrap_objects(*objects))


def szip(*objects):
    return Zip(wrap_objects(*objects))
