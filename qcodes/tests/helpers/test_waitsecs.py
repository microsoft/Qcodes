import time
from datetime import datetime

import pytest
from qcodes.logger.logger import LogCapture
from qcodes.utils.helpers import wait_secs


def test_bad_calls():
    bad_args = [None, datetime.now()]
    for arg in bad_args:
        with pytest.raises(TypeError):
            wait_secs(arg)


def test_warning():
    with LogCapture() as logs:
        secs_out = wait_secs(time.perf_counter() - 1)
    assert secs_out == 0

    assert logs.value.count('negative delay') == 1, logs.value
