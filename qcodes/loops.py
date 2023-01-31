"""
Deprecated
"""
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np

from qcodes.metadatable import Metadatable
from qcodes.station import Station
from qcodes.utils import full_class, issue_deprecation_warning

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
    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.data.data_set import new_data
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
issue_deprecation_warning("qcodes.loops module", alternative="qcodes_loop.loops")
