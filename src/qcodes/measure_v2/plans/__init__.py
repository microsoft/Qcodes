"""Plan-builders for common scan patterns.

Plan-builders are functions that return plan generators. They never yield
``OpenRun``/``CloseRun`` themselves — wrap with :py:func:`qcodes.measure_v2.run`
to open the run.
"""

from qcodes.measure_v2.plans.scan import scan_1d

__all__ = ["scan_1d"]
