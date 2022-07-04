"""
This file holds plotting utility functions that are
independent of the dataset from which to be plotted.
For the current dataset see :class:`qcodes.dataset.plot_dataset`
For the legacy dataset see :class:`qcodes.plots`
"""

from .auto_range import auto_range_iqr
from .axis_labels import find_scale_and_prefix
from .matplotlib_helpers import (
    apply_auto_color_scale,
    apply_color_scale_limits,
    auto_color_scale_from_config,
)

__all__ = [
    "apply_auto_color_scale",
    "apply_color_scale_limits",
    "auto_color_scale_from_config",
    "auto_range_iqr",
    "find_scale_and_prefix",
]
