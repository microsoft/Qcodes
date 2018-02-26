"""
Tests in this module have the basic structure:

def test():
    ....

def compare():
    ....

equivalence_test(test, compare)  # Test that the two produce the same results

We test that looping with the sweep module produces the same result as using "raw" python loops.

NB: For debugging, use the decorator "@settings(use_coverage=False)"
"""
from hypothesis import given

from qcodes.sweep.sweep import (
    ParameterSweep,
    FunctionSweep,
    Nest,
    Chain,
    Zip,
    FunctionWrapper,
    ParameterWrapper
)

from ._test_utils import (
    parameter_list,
    measurement_parameter_list,
    measure_function_list,
    set_function_list,
    sweep_values_list,
    equivalence_test
)


@given(parameter_list(1), sweep_values_list(1))
def test_parameter_sweep(parameters, sweep_values):
    """
    Basic sanity test. Sweeping with a sweep object should produce the same
    result as looping over the sweep values and
    manually setting parameters
    """
    def test():
        p = parameters[0]
        v = sweep_values[0]

        sweep_object = ParameterSweep(p, lambda: v)
        parameter_table = sweep_object.parameter_table

        assert parameter_table.table_list[0]["independent_parameters"][0] == (
            p.full_name, p.unit
        )

        for i in ParameterSweep(p, lambda: v):
            assert i[p.name] == p()

    def compare():
        p = parameters[0]
        v = sweep_values[0]

        for value in v:
            p.set(value)

    equivalence_test(test, compare)


@given(parameter_list(3), sweep_values_list(3))
def test_nesting(parameters, sweep_values):
    """
    Test the nesting functionality
    """

    def test():
        # Recasting a sweep object to list unrolls the sweep object
        sweep_object = Nest([
            ParameterSweep(p, lambda v=v: v)
            for p, v in zip(parameters, sweep_values)
        ])

        assert all([
            sweep_object.parameter_table.table_list[0][
                "independent_parameters"] == [(p.full_name, p.unit) for p in
                                              parameters]
        ])

        list(sweep_object)

    def compare():

        for v0 in sweep_values[0]:
            parameters[0].set(v0)
            for v1 in sweep_values[1]:
                parameters[1].set(v1)
                for v2 in sweep_values[2]:
                    parameters[2].set(v2)

    equivalence_test(test, compare)


@given(set_function_list(1), sweep_values_list(1))
def test_set_function(set_functions, sweep_values):
    """
    Test that we can use set functions to set independent parameters
    """
    set_function = set_functions[0]
    values = sweep_values[0]

    def test():
        sweep_object = FunctionSweep(set_function, lambda: values)
        assert sweep_object.parameter_table.table_list[0]["independent_parameters"][0] == (set_function.name, "none")
        list(sweep_object)

    def compare():
        for v in values:
            set_function()[0](v)

    equivalence_test(test, compare)


@given(parameter_list(2), sweep_values_list(2))
def test_chain(parameters, sweep_values):
    """
    Test the chaining functionality
    """
    def test():
        sweep_object = Chain([ParameterSweep(p, lambda v=v: v) for p, v in zip(parameters, sweep_values)])

        assert sweep_object.parameter_table.table_list[0]["independent_parameters"][0] == (
            parameters[0].full_name, parameters[0].unit
        )

        assert sweep_object.parameter_table.table_list[1]["independent_parameters"][0] == (
            parameters[1].full_name, parameters[1].unit
        )

        list(sweep_object)

    def compare():

        for value0 in sweep_values[0]:
            parameters[0].set(value0)

        for value1 in sweep_values[1]:
            parameters[1].set(value1)

    equivalence_test(test, compare)


@given(parameter_list(2), measurement_parameter_list(1), sweep_values_list(2))
def test_measure_parameter(parameters, measure_param, sweep_values):
    """
    Test that we can measure a parameter at an arbitrary locations in a nested sweep
    """
    p0, p1 = parameters
    m = measure_param[0]
    v0, v1 = sweep_values

    def test():
        sweep_object = Nest([ParameterSweep(p0, lambda: v0), ParameterWrapper(m), ParameterSweep(p1, lambda: v1)])
        parameter_table = sweep_object.parameter_table

        assert parameter_table.table_list[0]["independent_parameters"] == [
            (p0.full_name, p0.unit), (p1.full_name, p1.unit)
        ]
        assert parameter_table.table_list[0]["dependent_parameters"][0] == (m.full_name, m.unit)

        list(sweep_object)

    def compare():

        for value0 in v0:
            p0.set(value0)
            m()
            for value1 in v1:
                p1.set(value1)

    equivalence_test(test, compare)


@given(parameter_list(2), measure_function_list(1), sweep_values_list(2))
def test_measurement_function(parameters, measurements, sweep_values):
    """
    Test that we can nest a measurement function at an arbitrary location
    """
    p0, p1 = parameters
    m = measurements[0]
    v0, v1 = sweep_values

    def test():
        sweep_object = Nest([ParameterSweep(p0, lambda: v0), FunctionWrapper(m), ParameterSweep(p1, lambda: v1)])
        parameter_table = sweep_object.parameter_table
        assert parameter_table.table_list[0]["independent_parameters"] == [
            (p0.full_name, p0.unit), (p1.full_name, p1.unit)
        ]
        assert parameter_table.table_list[0]["dependent_parameters"][0] == (m.name, "hash")

        list(sweep_object)

    def compare():

        for value0 in v0:
            p0.set(value0)
            m()[0]()
            for value1 in v1:
                p1.set(value1)

    equivalence_test(test, compare)


@given(parameter_list(2), sweep_values_list(2, sweep_value_sizes=(4, 4)))
def test_zip(parameters, sweep_values):
    """
    Test the Zip operator
    """
    def test():
        sweep_object = Zip([ParameterSweep(p, lambda v=v: v) for p, v in zip(parameters, sweep_values)])
        parameter_table = sweep_object.parameter_table
        assert parameter_table.table_list[0]["independent_parameters"] == [
            (parameters[0].full_name, parameters[0].unit),
            (parameters[1].full_name, parameters[1].unit)
        ]

        list(sweep_object)

    def compare():

        for value0, value1 in zip(*sweep_values):
            parameters[0].set(value0)
            parameters[1].set(value1)

    equivalence_test(test, compare)
