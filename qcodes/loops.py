"""
Data acquisition loops.

The general scheme is:

1. create a (potentially nested) Loop, which defines the sweep setpoints and
delays

2. activate the loop (which changes it to an ActiveLoop object),

3. run it with the .run method, which creates a DataSet to hold the data,
and defines how and where to save the data.

Some examples:

- 1D sweep, using the default measurement set

>>> Loop(sweep_values, delay).run()

- 2D sweep, using the default measurement set sv1 is the outer loop, sv2 is the
  inner.

>>> Loop(sv1, delay1).loop(sv2, delay2).run()

- 1D sweep with specific measurements to take at each point

>>> Loop(sv, delay).each(param4, param5).run()

- Multidimensional sweep: 1D measurement of param6 on the outer loop, and another
  measurement in an inner loop.

>>> Loop(sv1, delay).each(param6, Loop(sv2, delay).each(sv3, delay)).run()

Supported commands to .each are:

    - Parameter: anything with a .get method and .name or .names see
      parameter.py for options
    - ActiveLoop
    - Task: any callable that does not generate data
    - Wait: a delay
"""
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np

from qcodes.data.data_array import DataArray
from qcodes.data.data_set import new_data
from qcodes.metadatable import Metadatable
from qcodes.station import Station
from qcodes.utils import full_class, issue_deprecation_warning

issue_deprecation_warning("qcodes.loops", alternative="qcodes_loop.loops")

try:
    from qcodes_loop.actions import (
        BreakIf,
        Task,
        Wait,
        _actions_snapshot,
        _Measure,
        _Nest,
        _QcodesBreak,
    )
    from qcodes_loop.loops import (
        ActiveLoop,
        Loop,
        active_data_set,
        active_loop,
        tprint,
        wait_secs,
    )
except ImportError as e:
    raise ImportError(
        "qcodes.loops is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
