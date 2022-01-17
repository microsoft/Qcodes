import pytest

from qcodes.utils.validators import Enum

enums = [
    [True, False],
    [1, 2, 3],
    [True, None, 1, 2.3, 'Hi!', b'free', (1, 2), float("inf")]
]

# enum items must be immutable - tuple OK, list bad.
not_enums = [[], [[1, 2], [3, 4]]]


def test_good():
    for enum in enums:
        e = Enum(*enum)

        for v in enum:
            e.validate(v)

        for v in [22, 'bad data', [44, 55]]:
            with pytest.raises((ValueError, TypeError)):
                e.validate(v)

        assert repr(e) == f"<Enum: {repr(set(enum))}>"

        # Enum is never numeric, even if its members are all numbers
        # because the use of is_numeric is for sweeping
        assert not e.is_numeric


def test_bad():
    for enum in not_enums:
        with pytest.raises(TypeError):
            Enum(*enum)


def test_valid_values():
    for enum in enums:
        e = Enum(*enum)
        for val in e._valid_values:
            e.validate(val)
