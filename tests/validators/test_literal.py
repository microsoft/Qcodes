from typing import Literal

import pytest

from qcodes.validators.validators import LiteralValidator


def test_literal_validator() -> None:
    A123 = Literal[1, 2, 3]

    A123Val = LiteralValidator[A123]

    a123_val = A123Val()

    a123_val.validate(1)

    with pytest.raises(ValueError, match="5 is not a member of "):
        a123_val.validate(5, context="Outside range")  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="some_str is not a member of "):
        a123_val.validate("some_str", context="Wrong type")  # pyright: ignore[reportArgumentType]


def test_literal_validator_repr() -> None:
    A123 = Literal[1, 2, 3]

    A123Val = LiteralValidator[A123]

    a123_val = A123Val()

    assert repr(a123_val) == "<Literal[1, 2, 3]>"


def test_valid_values() -> None:
    A123 = Literal[1, 2, 3]

    A123Val = LiteralValidator[A123]

    a123_val = A123Val()

    assert a123_val.valid_values == (1, 2, 3)


def test_missing_generic_arg_raises_at_runtime():
    wrong_validator = LiteralValidator()

    with pytest.raises(
        TypeError, match="Cannot find valid literal members for Validator"
    ):
        wrong_validator.validate(1)
