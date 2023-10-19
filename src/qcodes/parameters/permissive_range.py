from __future__ import annotations

import math
from typing import SupportsAbs


# could use numpy.arange here, but
# I'd like to be more flexible with the sign of step
def permissive_range(
    start: float, stop: float, step: SupportsAbs[float]
) -> list[float]:
    """
    Returns a range (as a list of values) with floating point steps.
    Always starts at start and moves toward stop, regardless of the
    sign of step.

    Args:
        start: The starting value of the range.
        stop: The end value of the range.
        step: Spacing between the values.
    """
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]
