from __future__ import annotations

from collections.abc import Sequence, Sized
from numbers import Integral
from typing import Any

import numpy as np
import numpy.typing as npt

from qcodes.parameters import (
    ArrayParameter,
    MultiParameter,
    ParameterBase,
    ParameterWithSetpoints,
)
from qcodes.validators import Arrays


def detect_shape_of_measurement(
    parameters: Sequence[ParameterBase],
    steps: Sequence[int] | Sequence[Sized] = (),
) -> dict[str, tuple[int, ...]]:
    """
    Construct the shape of a measurement of a dependent parameter from the
    parameter and the axes it is to be sweept over. Note that the
    steps should be given in the order from outer to inner loop and
    the shape will be returned in C order with the inner loop at the
    end.

    Args:
        parameters: The dependent parameters to construct the shape for
        steps: Zero or more dimensions that the parameter is to be sweept over.
           These can be given either as the length of the dimension or as
           a sequence of numbers to sweep over.

    Returns:
        A dictionary from the parameter name to a tuple of integers describing
        the shape.

    Raises:
        TypeError: If the shape cannot be detected due to incorrect types of
            parameters or steps supplied.

    """

    loop_shape: list[int] = []

    for step in steps:
        loop_shape.append(_get_shape_of_step(step))

    array_shapes: dict[str, tuple[int, ...]] = {}

    for param in parameters:
        if isinstance(param, MultiParameter):
            array_shapes.update(_get_shapes_of_multi_parameter(param=param))
        elif _param_is_array_like(param):
            array_shapes[param.register_name] = _get_shape_of_arrayparam(param)
        else:
            array_shapes[param.register_name] = ()

    shapes: dict[str, tuple[int, ...]] = {}

    for param_name, param_shape in array_shapes.items():
        total_shape = tuple(loop_shape) + param_shape
        if total_shape == ():
            total_shape = (1,)
        shapes[param_name] = total_shape

    return shapes


def _get_shape_of_step(step: int | np.integer[Any] | Sized | npt.NDArray) -> int:
    if isinstance(step, Integral):
        return int(step)
    elif isinstance(step, np.ndarray):
        # we explicitly check for numpy arrays
        # even if the numpy array is sized
        # to verify that they are one dimensional
        if not len(step.shape) == 1:
            raise TypeError("A step must be a one dimensional array")
        return int(step.shape[0])
    elif isinstance(step, Sized):
        return len(step)
    else:
        raise TypeError(
            f"get_shape_of_step takes "
            f"either an integer or a sized sequence "
            f"not: {type(step)}"
        )


def _param_is_array_like(meas_param: ParameterBase) -> bool:
    if isinstance(meas_param, (ArrayParameter, ParameterWithSetpoints)):
        return True
    elif isinstance(meas_param.vals, Arrays):
        return True
    return False


def _get_shape_of_arrayparam(param: ParameterBase) -> tuple[int, ...]:
    if isinstance(param, ArrayParameter):
        return tuple(param.shape)
    elif isinstance(param, ParameterWithSetpoints):
        if not isinstance(param.vals, Arrays):
            raise TypeError(
                "ParameterWithSetpoints must have an "
                "array type validator in order "
                "to detect it's shape."
            )
        shape = param.vals.shape
        if shape is None:
            raise TypeError(
                "Cannot infer shape of a ParameterWithSetpoints "
                "with an unknown shape in its validator."
            )
        return shape
    elif isinstance(param.vals, Arrays):
        shape = param.vals.shape
        if shape is None:
            raise TypeError(
                "Cannot infer shape of a Parameter "
                "with Array validator without a known shape."
            )
        return shape
    else:
        raise TypeError(
            f"Invalid parameter type: Expected an array like "
            f"parameter got: {type(param)}"
        )


def _get_shapes_of_multi_parameter(param: MultiParameter) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    for i, name in enumerate(param.full_names):
        shapes[name] = tuple(param.shapes[i])

    return shapes
