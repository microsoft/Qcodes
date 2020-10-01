from typing import Union, Sequence, Dict, Any, Tuple, Optional, Sized
from numbers import Integral
from collections import abc

import numpy as np

from qcodes.instrument.parameter import (_BaseParameter,
                                         ArrayParameter,
                                         MultiParameter,
                                         ParameterWithSetpoints)
from qcodes.utils.validators import Arrays


def get_shape_of_measurement(
        parameters: Sequence[_BaseParameter],
        steps: Union[Sequence[int], Sequence[Sized]] = ()
) -> Dict[str, Tuple[int, ...]]:
    """
    Construct the shape of a measurement of a dependent parameter from the
    parameter and the axes it is to be sweept over.

    Args:
        parameters: The dependent parameters to construct the shape for
        *steps: Zero or more dimensions that the parameter is to be sweept over.
           These can be given either as the length of the dimension or as
           a sequence of numbers to sweep over.

    Returns:
        A dictionary from the parameter name to a tuple of integers describing
        the shape.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}

    for param in parameters:

        if isinstance(param, MultiParameter):
            shapes.update(_get_shapes_of_multi_parameter(param=param))
        elif _param_is_array_like(param):
            shapes[param.full_name] = _get_shape_of_arrayparam(param)
        else:
            shapes[param.full_name] = ()

    for step in steps:
        for name in shapes.keys():
            shapes[name] = shapes[name] + (_get_shape_of_step(step),)

    return shapes


def _get_shape_of_step(
        step: Union[int, np.integer, Sized, np.ndarray]
) -> int:
    if isinstance(step, Integral):
        return int(step)
    elif isinstance(step, np.ndarray):
        # we explicitly check for numpy arrays
        # even if the numpy array is sized
        # to verify that they are one dimensional
        if not len(step.shape) == 1:
            raise TypeError("A step must be a one dimensional array")
        return int(step.shape[0])
    elif isinstance(step, abc.Sized):
        return len(step)
    else:
        raise TypeError(f"get_shape_of_step takes "
                        f"either an integer or a sized sequence "
                        f"not: {type(step)}")


def _param_is_array_like(meas_param: _BaseParameter) -> bool:
    if isinstance(meas_param, (ArrayParameter, ParameterWithSetpoints)):
        return True
    elif isinstance(meas_param.vals, Arrays):
        return True
    return False


def _get_shape_of_arrayparam(param: _BaseParameter) -> Tuple[int, ...]:

    if isinstance(param, ArrayParameter):
        return tuple(param.shape)
    elif isinstance(param, ParameterWithSetpoints):
        if not isinstance(param.vals, Arrays):
            raise TypeError("ParameterWithSetpoints must have an"
                            " array type validator.")
        shape = param.vals.shape
        if shape is None:
            raise TypeError("Cannot infer shape of a ParameterWithSetpoints "
                            "without an unknown shape in its validator.")
        return shape
    else:
        raise TypeError(f"Invalid parameter type: Expected an array like "
                        f"parameter got: {type(param)}")


def _get_shapes_of_multi_parameter(
        param: MultiParameter
) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}

    for i, name in enumerate(param.full_names):
        shapes[name] = tuple(param.shapes[i])

    return shapes
