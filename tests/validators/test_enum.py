from __future__ import annotations

from typing import Any

import pytest

import qcodes.validators as vals

enums: list[list[Any]] = [
    [True, False],
    [1, 2, 3],
    [True, None, 1, 2.3, 'Hi!', b'free', (1, 2), float("inf")]
]

# enum items must be immutable - tuple OK, list bad.
not_enums: list[list[list[int]]] = [[], [[1, 2], [3, 4]]]


def test_good() -> None:
    for enum in enums:
        e = vals.Enum(*enum)

        for v in enum:
            e.validate(v)

        for v in [22, 'bad data', [44, 55]]:
            with pytest.raises((ValueError, TypeError)):
                e.validate(v)

        assert repr(e) == f"<Enum: {set(enum)!r}>"

        # Enum is never numeric, even if its members are all numbers
        # because the use of is_numeric is for sweeping
        assert not e.is_numeric


def test_bad() -> None:
    for enum in not_enums:
        with pytest.raises(TypeError):
            vals.Enum(*enum)  # type: ignore[arg-type]


def test_valid_values() -> None:
    for enum in enums:
        e = vals.Enum(*enum)
        for val in e._valid_values:
            e.validate(val)
