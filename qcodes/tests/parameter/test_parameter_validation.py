from qcodes.instrument.parameter import Parameter

from .conftest import BookkeepingValidator


def test_number_of_validations():
    p = Parameter('p', set_cmd=None, initial_value=0,
                  vals=BookkeepingValidator())
    # in the set wrapper the final value is validated
    # and then subsequently each step is validated.
    # in this case there is one step so the final value
    # is validated twice.
    assert p.vals.values_validated == [0, 0]

    p.step = 1
    p.set(10)
    assert p.vals.values_validated == [0, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_number_of_validations_for_set_cache():
    p = Parameter('p', set_cmd=None,
                  vals=BookkeepingValidator())
    assert p.vals.values_validated == []

    p.cache.set(1)
    assert p.vals.values_validated == [1]

    p.cache.set(4)
    assert p.vals.values_validated == [1, 4]

    p.step = 1
    p.cache.set(10)
    assert p.vals.values_validated == [1, 4, 10]
