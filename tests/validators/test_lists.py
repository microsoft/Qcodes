from typing import TYPE_CHECKING, Any

import pytest

from qcodes.validators import Ints, Lists

if TYPE_CHECKING:
    import numpy as np


def test_type() -> None:
    list_validator: Lists[Any] = Lists()
    v1 = ["a", "b", 5]
    list_validator.validate(v1)

    v2 = 234
    with pytest.raises(TypeError):
        list_validator.validate(v2)  # type: ignore[arg-type]


def test_elt_vals() -> None:
    list_validator = Lists(Ints(max_value=10))
    v1: list[int | np.integer | bool] = [0, 1, 5]
    list_validator.validate(v1)

    v2 = [0, 1, 11]
    with pytest.raises(ValueError):
        list_validator.validate(v2)  # type: ignore[arg-type]


def test_valid_values() -> None:
    val = Lists(Ints(max_value=10))
    for vval in val.valid_values:
        val.validate(vval)
