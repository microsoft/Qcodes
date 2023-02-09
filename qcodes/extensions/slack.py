import inspect
import logging
import os
import tempfile
import threading
import traceback
import warnings
from functools import partial
from time import sleep

from requests.exceptions import ConnectTimeout, HTTPError, ReadTimeout
from slack_sdk import WebClient
from urllib3.exceptions import ReadTimeoutError

from qcodes import config as qc_config
from qcodes.parameters import ParameterBase
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.extensions.slack import Slack, SlackTimeoutWarning, convert_command
    from qcodes_loop.loops import active_data_set, active_loop
    from qcodes_loop.plots.base import BasePlot
except ImportError as e:
    raise ImportError(
        "qcodes.utils.slack is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e



__all__ = ["Slack", "SlackTimeoutWarning", "convert_command"]
issue_deprecation_warning(
    "qcodes.utils.slack module", alternative="qcodes_loop.extensions.slack"
)
