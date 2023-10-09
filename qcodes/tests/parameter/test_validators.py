import hypothesis.strategies as hst
import pytest
from hypothesis import given

from qcodes.parameters import Parameter
from qcodes.validators import Ints


def test_add_ints_validator_to_parameter():
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    ints_val = Ints(min_value=0, max_value=10)
    p.add_validator(ints_val)
    assert len(p.validators) == 1
    assert p.validators[0] is ints_val
    assert p.vals is ints_val


def test_remove_ints_validator_from_parameter():
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    p.add_validator(Ints(min_value=0, max_value=10))
    p.remove_validator()
    assert len(p.validators) == 0
    assert p.vals is None


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
    if len(validators) > 0:
        assert p.vals is validators[0]

    if value_to_validate >= max_min_value and value_to_validate <= min_max_value:
        p.validate(value_to_validate)
    else:
        with pytest.raises(ValueError):
            p.validate(value_to_validate)

    while len(p.validators) > 0:
        p.remove_validator()
        if len(p.validators) > 0:
            assert p.vals is p.validators[0]

    assert len(p.validators) == 0
    assert p.vals is None

    p.validate(value_to_validate)


@given(
    min_val=hst.integers(max_value=0),
    max_val=hst.integers(min_value=1),
    value_to_validate=hst.integers(),
)
def test_validator_context(min_val: int, max_val: int, value_to_validate: int) -> None:
    p = Parameter("test_param", set_cmd=None, get_cmd=None)

    with p.extra_validator(Ints(min_value=min_val, max_value=max_val)):
        assert len(p.validators) == 1

        if value_to_validate >= min_val and value_to_validate <= max_val:
            p.validate(value_to_validate)
        else:
            with pytest.raises(ValueError):
                p.validate(value_to_validate)

    assert len(p.validators) == 0

    p.validate(value_to_validate)


def test_validator_doc() -> None:
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    p.add_validator(Ints(min_value=0, max_value=10))
    p.add_validator(Ints(min_value=3, max_value=7))
    assert p.__doc__ is not None
    assert "vals` <Ints 0<=v<=10>" in p.__doc__
    assert "vals` <Ints 3<=v<=7>" in p.__doc__
    p.remove_validator()
    assert "vals` <Ints 0<=v<=10>" in p.__doc__
    assert "vals` <Ints 3<=v<=7>" not in p.__doc__
    p.remove_validator()
    assert "vals` <Ints 0<=v<=10>" not in p.__doc__
    assert "vals` <Ints 3<=v<=7>" not in p.__doc__

    p.vals = Ints(min_value=4, max_value=6)
    assert "vals` <Ints 0<=v<=10>" not in p.__doc__
    assert "vals` <Ints 3<=v<=7>" not in p.__doc__
    assert "vals` <Ints 4<=v<=6>" in p.__doc__


def test_validator_snapshot() -> None:
    p = Parameter("test_param", set_cmd=None, get_cmd=None)
    p.add_validator(Ints(min_value=0, max_value=10))
    p.add_validator(Ints(min_value=3, max_value=7))
    snapshot = p.snapshot()
    assert "<Ints 0<=v<=10>" in snapshot["validators"]
    assert "<Ints 3<=v<=7>" in snapshot["validators"]
    assert snapshot["vals"] == "<Ints 0<=v<=10>"
    p.remove_validator()
    snapshot = p.snapshot()
    assert "<Ints 0<=v<=10>" in snapshot["validators"]
    assert "<Ints 3<=v<=7>" not in snapshot["validators"]
    assert snapshot["vals"] == "<Ints 0<=v<=10>"
    p.remove_validator()
    snapshot = p.snapshot()
    assert "<Ints 0<=v<=10>" not in snapshot["validators"]
    assert "<Ints 3<=v<=7>" not in snapshot["validators"]
    assert "vals" not in snapshot.keys()
    p.vals = Ints(min_value=4, max_value=6)
    snapshot = p.snapshot()
    assert "<Ints 0<=v<=10>" not in snapshot["validators"]
    assert "<Ints 3<=v<=7>" not in snapshot["validators"]
    assert "<Ints 4<=v<=6>" in snapshot["validators"]
    assert "<Ints 4<=v<=6>" == snapshot["vals"]
