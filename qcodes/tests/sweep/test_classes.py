"""
Tests in this module have the basic structure:

def test():
    ....

def compare():
    ....

equivalence_test(test, compare)  # Test that the two produce the same results

We test that looping with the sweep module produces the same result as using "raw" python loops
"""

from hypothesis import given
from qcodes.sweep.sweep import ParameterSweep, Nest

from ._test_utils import parameter_list, sweep_values_list, equivalence_test


@given(parameter_list(1), sweep_values_list(1))  # This test requires one parameter
def test_sanity(parameters, sweep_values):
    """
    Basic sanity test. Sweeping with a sweep object should produce the same result as looping over the sweep values and
    manually setting parameters
    """
    def test():
        p = parameters[0]
        v = sweep_values[0]

        for i, value in zip(ParameterSweep(p, lambda: v), v):
            assert i[p.name]["independent_parameter"]
            assert i[p.name]["value"] == value
            assert i[p.name]["unit"] == p.unit

    def compare():
        p = parameters[0]
        v = sweep_values[0]

        for value in v:
            p.set(value)

    equivalence_test(test, compare)


@given(parameter_list(3), sweep_values_list(3))  # This test requires three parameters
def test_nesting(parameters, sweep_values):
    """
    Test the nesting functionality
    """

    def test():
        _ = list(Nest([ParameterSweep(p, lambda v=v: v) for p, v in zip(parameters, sweep_values)]))

    def compare():

        for v0 in sweep_values[0]:
            parameters[0].set(v0)
            for v1 in sweep_values[1]:
                parameters[1].set(v1)
                for v2 in sweep_values[2]:
                    parameters[2].set(v2)

    equivalence_test(test, compare)