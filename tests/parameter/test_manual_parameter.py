import pytest

from qcodes.parameters import ManualParameter, Parameter


def test_manual_parameter_set_get_raw() -> None:
    myparam = Parameter("myparam", set_cmd=None, get_cmd=None)

    value = 23
    value2 = 132

    assert myparam.get() is None
    assert myparam.get_raw() is None

    myparam.set_raw(value)
    assert myparam.get() == value
    assert myparam.get_raw() == value

    myparam.set(value2)
    assert myparam.get() == value2
    assert myparam.get_raw() == value2


def test_manual_parameter_forbidden_kwargs() -> None:
    forbidden_kwargs = ["get_cmd", "set_cmd"]

    for fk in forbidden_kwargs:
        match = f'It is not allowed to set "{fk}" for a ManualParameter'
        with pytest.raises(ValueError, match=match):
            ManualParameter("test", **{fk: None})  # type: ignore[arg-type]
