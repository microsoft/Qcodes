"""
This module provides convenience functions for testing the sweep classes
"""

from typing import AnyStr, Callable

from hypothesis import strategies as st
from qcodes import ManualParameter

from qcodes.sweep import getter, setter


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
    def __init__(self, name: str, unit: str, independent_parameter: bool=False)->None:
        super().__init__(name, unit=unit)
        self._independent_parameter = independent_parameter

    def set_raw(self, value: AnyStr)->None:
        self._save_val(value)
        mock_io.write("Setting {} to {}".format(self.name, str(value)))

    def get_raw(self)->int:
        if self._independent_parameter:
            raw_value = hash(str(mock_io))
            mock_io.write("Current value of {} is {}".format(self.name, str(raw_value)))
        else:
            raw_value = self.raw_value
        return raw_value


class TestMeasureFunction:
    """
    We can use measurement function instead of qcodes parameters to measure dependent parameters.
    """
    def __init__(self, name: str)->None:
        self._name = name

    @property
    def name(self)->str:
        return self._name

    def caller(self):
        hs = hash(str(mock_io))
        mock_io.write("{} returns {}".format(self._name, hs))

    def __call__(self)->dict:
        return getter([(self._name, "hash")])(self.caller)()


class TestSetFunction:
    """
    We can use set functions instead of qcodes parameters to set independent parameters
    """
    def __init__(self, name: str)->None:
        self._name = name

    @property
    def name(self)->str:
        return self._name

    def caller(self, value)->None:
        mock_io.write("Setting {} to {}".format(self._name, value))

    def __call__(self):
        return setter([(self._name, "none")])(self.caller)()


def equivalence_test(test: Callable, compare: Callable)->None:
    """
    Assert that two test functions produce the same output on the stdout
    """
    mock_io.flush()
    test()
    test_value = str(mock_io)
    mock_io.flush()

    compare()
    compare_value = str(mock_io)
    mock_io.flush()

    assert test_value == compare_value


def parameter_list(list_size: int)->st.lists:
    """
    Return a list of independent parameters useful for testing through the hypothesis module.
    """
    a_to_z = [chr(i) for i in range(ord("a"), ord("z"))]

    return st.lists(
        st.builds(
            TestParameter,
            name=st.text(alphabet=a_to_z, min_size=4, max_size=4),
            unit=st.text(alphabet=a_to_z, min_size=1, max_size=1)
        ),
        min_size=list_size,
        max_size=list_size,
        unique_by=lambda p: p.name
    )


def measurement_parameter_list(list_size: int)->st.lists:
    """
    Return a list of dependent parameters useful for testing through the hypothesis module. The difference between
    dependent and independent parameters in this context is that the latter will be verbose on getting the current
    parameter value
    """
    a_to_z = [chr(i) for i in range(ord("a"), ord("z"))]

    return st.lists(
        st.builds(
            TestParameter,
            name=st.text(alphabet=a_to_z, min_size=4, max_size=4),
            unit=st.text(alphabet=a_to_z, min_size=1, max_size=1),
            independent_parameter=st.sampled_from([True])
        ),
        min_size=list_size,
        max_size=list_size,
        unique_by=lambda p: p.name
    )


def measure_function_list(list_size: int)->st.lists:
    """
    Return a list of measurement functions useful for testing through the hypothesis module.
    """
    a_to_z = [chr(i) for i in range(ord("a"), ord("z"))]

    return st.lists(
        st.builds(
            TestMeasureFunction,
            name=st.text(alphabet=a_to_z, min_size=4, max_size=4)
        ),
        min_size=list_size,
        max_size=list_size,
        unique_by=lambda p: p.name
    )


def set_function_list(list_size: int)->st.lists:
    """
    Return a list of set functions useful for testing through the hypothesis module.
    """
    a_to_z = [chr(i) for i in range(ord("a"), ord("z"))]

    return st.lists(
        st.builds(
            TestSetFunction,
            name=st.text(alphabet=a_to_z, min_size=4, max_size=4)
        ),
        min_size=list_size,
        max_size=list_size,
        unique_by=lambda p: p.name
    )


def sweep_values_list(list_size: int, sweep_value_sizes=(3, 6))->st.lists:
    """
    Return a list of sweep values useful for testing through the hypothesis module. Sweep values are themselves lists
    of floats.
    """
    return st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=sweep_value_sizes[0], max_size=sweep_value_sizes[1],
            unique=True
        ),
        min_size=list_size, max_size=list_size
    )
