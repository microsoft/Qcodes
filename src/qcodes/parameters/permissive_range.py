from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, SupportsAbs

import numpy as np

if TYPE_CHECKING:
    from qcodes.utils.types import NumberType


# could use numpy.arange here, but
# I'd like to be more flexible with the sign of step
def permissive_range(
    start: NumberType, stop: NumberType, step: SupportsAbs[NumberType]
) -> list[NumberType]:
    """
    Returns a range (as a list of values) with floating point steps.
    Always starts at start and moves toward stop, regardless of the
    sign of step.

    Args:
        start: The starting value of the range.
        stop: The end value of the range.
        step: Spacing between the values.

    """
    # Converting numpy integers to Python ints below would also drop the
    # promotion numpy applies when they meet a numpy float (int64 with
    # float32 gives float64), so cast numpy floats to that type first.
    numpy_args = [arg for arg in (start, stop, step) if isinstance(arg, np.generic)]
    if any(isinstance(arg, np.integer) for arg in numpy_args):
        float_type = np.result_type(*numpy_args).type
        if isinstance(start, np.floating):
            start = float_type(start)
        if isinstance(stop, np.floating):
            stop = float_type(stop)
        if isinstance(step, np.floating):
            step = float_type(step)
    # Do integer arithmetic with Python ints. Fixed width integers such as
    # numpy.uint16 or numpy.int8 silently wrap around on overflow, and an
    # unsigned integer cannot represent the negative step of a descending range.
    if isinstance(start, numbers.Integral):
        start = int(start)
    if isinstance(stop, numbers.Integral):
        stop = int(stop)
    if isinstance(step, numbers.Integral):
        step = int(step)
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]
