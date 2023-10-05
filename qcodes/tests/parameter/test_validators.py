import hypothesis.strategies as hst
import pytest
from hypothesis import given

from qcodes import Parameter
from qcodes.validators import Ints


def test_add_ints_validator_to_parameter():
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    p.add_validator(Ints(min_value=0, max_value=10))
    assert len(p.validators) == 1


def test_remove_ints_validator_from_parameter():
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    p.add_validator(Ints(min_value=0, max_value=10))
    p.remove_validator()
    assert len(p.validators) == 0


@given(
    min_max_values=hst.lists(
        hst.tuples(hst.integers(max_value=0), hst.integers(min_value=1)),
        min_size=1,
        max_size=10,
    ),
    value_to_validate=hst.integers(),
    add_validator_via_constructor=hst.booleans(),
)
def test_multiple_ints_validators(
    min_max_values: list[tuple[int, int]],
    value_to_validate: int,
    add_validator_via_constructor: bool,
) -> None:
    validators = []
    for min_val, max_val in min_max_values:
        validator = Ints(min_value=min_val, max_value=max_val)
        validators.append(validator)
    n_validators = len(validators)

    max_min_value = max([min_val for min_val, _ in min_max_values])
    min_max_value = min([max_val for _, max_val in min_max_values])

    if add_validator_via_constructor:
        p = Parameter("test_param", set_cmd=None, get_cmd=None, vals=validators[0])

        for validator in validators[1:]:
            p.add_validator(validator)
    else:
        p = Parameter("test_param", set_cmd=None, get_cmd=None)

        for validator in validators:
            p.add_validator(validator)

    assert len(p.validators) == n_validators

    if value_to_validate >= max_min_value and value_to_validate <= min_max_value:
        p.validate(value_to_validate)
    else:
        with pytest.raises(ValueError):
            p.validate(value_to_validate)

    while len(p.validators) > 0:
        p.remove_validator()

    assert len(p.validators) == 0

    p.validate(value_to_validate)
