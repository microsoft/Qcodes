import re
import pytest

from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes import Instrument


class Dummy(Instrument):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self._a = 0
        self._b = 0

        self.add_parameter(
            "a",
            get_parser=int,
            parameter_class=GroupParameter,
            docstring="Some succinct description",
            label="label",
            unit="SI"
        )

        self.add_parameter(
            "b",
            get_parser=int,
            parameter_class=GroupParameter,
            docstring="Some succinct description",
            label="label",
            unit="SI"
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
