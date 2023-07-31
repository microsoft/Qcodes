from __future__ import annotations

from typing import Any

import pytest

import qcodes.validators as vals


def test_dict() -> None:
    d = vals.Dict()
    my_dict: dict[Any, Any] = {}
    d.validate(my_dict)
    my_int = 5
    with pytest.raises(TypeError):
        d.validate(my_int)  # type: ignore[arg-type]


def test_valid_values() -> None:
    val = vals.Dict()
    for vval in val.valid_values:
        val.validate(vval)
