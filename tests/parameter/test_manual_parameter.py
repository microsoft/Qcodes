from qcodes.parameters import Parameter


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
