"""
This module tests the convenience functions to create sweep objects. We test that the function create the expected
sweep objects.
"""

from hypothesis import given

from qcodes.sweep.sweep import (
    ParameterSweep,
    FunctionSweep,
    Nest,
    Chain,
    FunctionWrapper,
    ParameterWrapper
)

from qcodes.sweep import sweep, nest, chain

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

# Since the zip and chain operators use the same wrapping function as nest, so we will not test these separately.
