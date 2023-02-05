"""
Tests for the specialized_parameters module
"""

from time import sleep

import pytest

from qcodes.parameters import ElapsedTimeParameter


def test_elapsed_time_parameter_init() -> None:
    tp1 = ElapsedTimeParameter("time1")
    sleep(0.01)
    tp2 = ElapsedTimeParameter("time2")

    assert tp1() > tp2()


def test_elapsed_time_parameter_monotonic() -> None:
    tp = ElapsedTimeParameter("time")

    times = [tp() for _ in range(25)]

    assert sorted(times) == times


def test_elapsed_time_parameter_reset_clock() -> None:
    tp = ElapsedTimeParameter("time")

    sleep(0.01)
    t1 = tp()

    tp.reset_clock()
    t2 = tp()

    assert t1 > t2


def test_elapsed_time_parameter_not_settable() -> None:
    tp = ElapsedTimeParameter("time")

    with pytest.raises(NotImplementedError):
        tp(0)


def test_elapsed_time_parameter_forbidden_kwargs() -> None:
    forbidden_kwargs = ["unit", "get_cmd", "set_cmd"]

    for fb_kwarg in forbidden_kwargs:
        match = f'Can not set "{fb_kwarg}" for an ElapsedTimeParameter'
        with pytest.raises(ValueError, match=match):
            ElapsedTimeParameter("time", **{fb_kwarg: None})  # type: ignore[arg-type]
