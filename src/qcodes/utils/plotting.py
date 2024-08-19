"""
Module left for backwards compatibility. Please do not depend on this
in new code.
"""

from qcodes.plotting import (
    apply_auto_color_scale,
    apply_color_scale_limits,
    auto_color_scale_from_config,
    auto_range_iqr,
    find_scale_and_prefix,
)
from qcodes.plotting.axis_labels import _ENGINEERING_PREFIXES, _UNITS_FOR_RESCALING
from qcodes.plotting.matplotlib_helpers import _set_colorbar_extend
