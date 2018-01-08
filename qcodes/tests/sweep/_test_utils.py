"""
This module provides convenience functions for testing the sweep classes
"""

from typing import AnyStr, Callable

from hypothesis import strategies as st
from qcodes import ManualParameter


class MockIO:
    """
    A simple class to mock IO. Strings are stored in a buffer.
    """
    def __init__(self)->None:
        self._buffer = ""

    def write(self, string: AnyStr)->None:
        self._buffer += "\n" + string

    def __repr__(self)->str:
        return self._buffer

    def flush(self)->None:
        self._buffer = ""

mock_io = MockIO()


class TestParameter(ManualParameter):
    """
    A Manual Parameter subclass to test against
    """
    def set_raw(self, value: AnyStr)->None:
        mock_io.write("Setting {} to {}".format(self.name, str(value)))

    def get_raw(self)->int:
        return 0


def equivalence_test(test: Callable, compare: Callable)->None:
    """
    Assert that two test functions produce the same output on the stdout
    """
    test()
    test_value = str(mock_io)
    mock_io.flush()

    compare()
    compare_value = str(mock_io)
    mock_io.flush()

    assert test_value == compare_value


def parameter_list(list_size: int)->st.lists:
    """
    Return a list of parameters useful for testing through the hypothesis module.
    """
    a_to_z = [chr(i) for i in range(ord("a"), ord("z"))]

    return st.lists(
        st.builds(
            TestParameter,
            name=st.text(alphabet=a_to_z, min_size=4, max_size=4),
            unit=st.text(alphabet=a_to_z, min_size=4, max_size=4)
        ),
        min_size=list_size,
        max_size=list_size,
        unique_by=lambda p: p.name
    )


def sweep_values_list(list_size: int)->st.lists:
    """
    Return a list of sweep values useful for testing through the hypothesis module. Sweep values are themselves lists
    of floats.
    """
    return st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=3, max_size=6,
            unique=True
        ),
        min_size=list_size, max_size=list_size
    )
