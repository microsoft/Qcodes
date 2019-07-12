"""
Tests for the utils.time module
"""

from time import sleep

import pytest

from qcodes.utils.time import ElapsedTimeParameter


def test_elapsed_time_parameter_init():

    tp1 = ElapsedTimeParameter('time1')
    sleep(0.01)
    tp2 = ElapsedTimeParameter('time2')

    assert tp1() > tp2()


def test_elapsed_time_parameter_monotonic():

    tp = ElapsedTimeParameter('time')

    times = [tp() for _ in range(25)]

    assert sorted(times) == times


def test_elapsed_time_parameter_reset_clock():

    tp = ElapsedTimeParameter('time')

    sleep(0.01)
    t1 = tp()

    tp.reset_clock()
    t2 = tp()

    assert t1 > t2


def test_elapsed_time_parameter_not_settable():

    tp = ElapsedTimeParameter('time')

    with pytest.raises(NotImplementedError):
        tp(0)
