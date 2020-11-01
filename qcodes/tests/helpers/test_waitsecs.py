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


def test_good_calls():
    for secs in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        finish_clock = time.perf_counter() + secs
        secs_out = wait_secs(finish_clock)
        assert secs_out > secs - 3e-4
        # add a tiny offset as this test may fail if
        # otherwise if the two calls to perf_counter are close
        # enough to return the same result as a + b - a cannot
        # in general be assumed to be <= b in floating point
        # math (here a is perf_counter() and b is the wait time
        assert secs_out <= secs + 1e-14


def test_warning():
    with LogCapture() as logs:
        secs_out = wait_secs(time.perf_counter() - 1)
    assert secs_out == 0

    assert logs.value.count('negative delay') == 1, logs.value
