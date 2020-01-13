import re
import pytest
from typing import Optional

from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes import Instrument


@pytest.fixture(autouse=True)
def close_all_instruments():
    """Makes sure that after startup and teardown all instruments are closed"""
    Instrument.close_all()
    yield
    Instrument.close_all()


class Dummy(Instrument):
    def __init__(self, name: str,
                 initial_a: Optional[int] = None,
                 initial_b: Optional[int] = None) -> None:
        super().__init__(name)

        self._a = 0
        self._b = 0

        self.add_parameter(
            "a",
            get_parser=int,
            parameter_class=GroupParameter,
            docstring="Some succinct description",
            label="label",
            unit="SI",
            initial_value=initial_a
        )

        self.add_parameter(
            "b",
            get_parser=int,
            parameter_class=GroupParameter,
            docstring="Some succinct description",
            label="label",
            unit="SI",
            initial_value=initial_b
        )

        Group(
            [self.a, self.b],
            set_cmd="CMD {a}, {b}",
            get_cmd="CMD?"
        )

    def write(self, cmd: str) -> None:
        result = re.search("CMD (.*), (.*)", cmd)
        assert result is not None
        self._a, self._b = [int(i) for i in result.groups()]

    def ask(self, cmd: str) -> str:
        assert cmd == "CMD?"
        return ",".join([str(i) for i in [self._a, self._b]])


def test_sanity():
    """
    Test that we can individually address parameters "a" and "b", which belong
    to the same group.
    """
    dummy = Dummy("dummy")

    assert dummy.a() == 0
    assert dummy.b() == 0

    dummy.a(3)
    dummy.b(6)

    assert dummy.a() == 3
    assert dummy.b() == 6

    dummy.b(10)
    assert dummy.a() == 3
    assert dummy.b() == 10


def test_raise_on_get_set_cmd():

    for arg in ["set_cmd", "get_cmd"]:
        kwarg = {arg: ""}

        with pytest.raises(ValueError) as e:
            GroupParameter(name="a", **kwarg)

        assert str(e.value) == "A GroupParameter does not use 'set_cmd' or " \
                               "'get_cmd' kwarg"


def test_raises_on_get_set_without_group():
    param = GroupParameter(name='b')

    with pytest.raises(RuntimeError) as e:
        param.get()
    assert str(e.value) == "('Trying to get Group value but no group defined', 'getting b')"

    with pytest.raises(RuntimeError) as e:
        param.set(1)
    assert str(e.value) == "('Trying to set Group value but no group defined', 'setting b to 1')"


def test_initial_values():
    initial_a = 42
    initial_b = 43
    dummy = Dummy("dummy", initial_a=initial_a, initial_b=initial_b)

    assert dummy.a() == initial_a
    assert dummy.b() == initial_b


def test_raise_on_not_all_initial_values():
    expected_err_msg = (r'Either none or all of the parameters in a group '
                        r'should have an initial value. Found initial values '
                        r'for \[.*\] but not for \[.*\].')
    with pytest.raises(ValueError, match=expected_err_msg):
        dummy = Dummy("dummy", initial_a=42)
