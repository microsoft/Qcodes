"""
This module tests the convenience functions to create sweep objects. We test that the function create the expected
sweep objects.
"""

from hypothesis import given

from qcodes.instrument.parameter import ManualParameter

from qcodes.sweep.sweep import (
    ParameterSweep,
    FunctionSweep,
    Nest,
    FunctionWrapper,
    ParameterWrapper
)

from qcodes.sweep import sweep, nest, szip

from ._test_utils import (
    parameter_list,
    measurement_parameter_list,
    measure_function_list,
    set_function_list,
    sweep_values_list,
    equivalence_test
)


@given(parameter_list(1), sweep_values_list(1))
def test_parameters_sweep(parameters, sweep_values):
    """
    The "sweep" function should detect whether a ParameterSweep object should be created or a FunctionSweep object
    """
    p = parameters[0]
    v = sweep_values[0]
    assert list(sweep(p, v)) == list(ParameterSweep(p, lambda: v))


@given(set_function_list(1), sweep_values_list(1))
def test_setfunction_sweep(set_functions, sweep_values):
    """
    The "sweep" function should detect whether a ParameterSweep object should be created or a FunctionSweep object
    """
    f = set_functions[0]
    v = sweep_values[0]
    assert list(sweep(f, v)) == list(FunctionSweep(f, lambda: v))


@given(parameter_list(1), sweep_values_list(1), measurement_parameter_list(1))
def test_wrap_parameters(parameters, sweep_values, measurements):
    """
    The "nest" operator should detect how to wrap objects in its arguments list to create valid sweep objects. For
    instance, a QCoDeS parameter should be wrapped with "ParameterWrapper".
    """
    p = parameters[0]
    v = sweep_values[0]
    m = measurements[0]

    def test():
        list(nest(sweep(p, v), m))

    def compare():
        list(Nest([ParameterSweep(p, lambda: v), ParameterWrapper(m)]))

    equivalence_test(test, compare)


@given(parameter_list(1), sweep_values_list(1), measure_function_list(1))
def test_wrap_callable(parameters, sweep_values, measurements):
    """
    The "nest" operator should detect how to wrap objects in its arguments list to create valid sweep objects. For
    instance, a callable should be wrapped with "FunctionWrapper".
    """
    p = parameters[0]
    v = sweep_values[0]
    m = measurements[0]

    def test():
        list(nest(sweep(p, v), m))

    def compare():
        list(Nest([ParameterSweep(p, lambda: v), FunctionWrapper(m)]))

    equivalence_test(test, compare)

# Since the chain operator use the same wrapping function as nest, so we will not test this separately.


def test_szip_measure_prior_to_set():
    """
    We can use szip to perform a measurement before setting sweep set points.  Test this scenario
    """
    x = ManualParameter("x")
    v = range(1, 10)
    m = ManualParameter("m")
    m.get = lambda: 2 * x()

    x(0)
    count = 0
    previous_x = x()

    for count, i in enumerate(szip(m, sweep(x, v))):
        assert i["m"] == 2 * previous_x  # Note that at this point, x should already have been incremented
        assert count < len(v)
        previous_x = x()

    assert count == len(v) - 1


def test_szip_finiteness():
    """
    Test that if only parameters and/or functions are given to szip, we do not end up in infinite loops but instead
    iterate once returning the value of the parameter/function
    """
    x = ManualParameter("x")
    y = ManualParameter("y")

    x(0)
    y(1)

    for count, i in enumerate(szip(x, y)):
        assert i["x"] == x()
        assert i["y"] == y()
        assert count == 0
