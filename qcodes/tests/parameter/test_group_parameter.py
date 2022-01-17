import re
from datetime import datetime
from typing import Optional

import pytest

from qcodes import Instrument
from qcodes.instrument.group_parameter import Group, GroupParameter


@pytest.fixture(autouse=True)
def close_all_instruments():
    """Makes sure that after startup and teardown all instruments are closed"""
    Instrument.close_all()
    yield
    Instrument.close_all()


class Dummy(Instrument):
    def __init__(self, name: str,
                 initial_a: Optional[int] = None,
                 initial_b: Optional[int] = None,
                 scale_a: Optional[float] = None,
                 get_cmd: Optional[str] = "CMD?") -> None:
        super().__init__(name)

        self._a = 0
        self._b = 0
        self._get_cmd = get_cmd

        self.add_parameter(
            "a",
            get_parser=int,
            parameter_class=GroupParameter,
            docstring="Some succinct description",
            label="label",
            unit="SI",
            initial_value=initial_a,
            scale=scale_a
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

        self.group = Group(
            [self.a, self.b],
            set_cmd="CMD {a}, {b}",
            get_cmd=get_cmd
        )

    def write(self, cmd: str) -> None:
        result = re.search("CMD (.*), (.*)", cmd)
        assert result is not None
        self._a, self._b = (int(i) for i in result.groups())

    def ask(self, cmd: str) -> str:
        assert cmd == self._get_cmd
        return ",".join(str(i) for i in [self._a, self._b])


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


def test_raises_runtime_error_on_update_if_get_cmd_is_none():
    dummy = Dummy("dummy", get_cmd=None)
    msg = ("Cannot update values in the group with "
           "parameters - dummy_a, dummy_b since it "
           "has no `get_cmd` defined.")
    with pytest.raises(RuntimeError, match=msg):
        dummy.group.update()

def test_raises_runtime_error_if_set_parameters_called_with_empty_dict():
    dummy = Dummy("dummy")
    parameters_dict = dict()
    msg = ("Provide at least one group parameter and its value to be set.")

    with pytest.raises(RuntimeError, match=msg):
        dummy.group.set_parameters(parameters_dict)

def test_set_parameters_called_for_one_parameter():
    dummy = Dummy("dummy")
    parameters_dict = {"a": 7}

    dummy.group.set_parameters(parameters_dict)
    assert dummy.a() == 7
    assert dummy.b() == 0

def test_set_parameters_called_for_more_than_one_parameters():
    dummy = Dummy("dummy")
    parameters_dict = {"a": 10, "b": 57}

    dummy.group.set_parameters(parameters_dict)
    assert dummy.a() == 10
    assert dummy.b() == 57

def test_set_parameters_when_parameter_value_not_equal_to_raw_value():
    dummy = Dummy("dummy", scale_a=10)
    parameters_dict = {"a": 7}

    dummy.group.set_parameters(parameters_dict)
    assert dummy.a.cache.get(get_if_invalid=False) == 7
    assert dummy.a.cache.raw_value == 70
    assert dummy.a() == 7

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


def test_update_group_parameter_reflected_in_cache_of_all_params():
    dummy = Dummy("dummy")
    group = dummy.a.group

    assert dummy.a.cache.timestamp is None
    assert dummy.b.cache.timestamp is None
    assert dummy.a.cache.get(get_if_invalid=False) is None
    assert dummy.b.cache.get(get_if_invalid=False) is None

    before = datetime.now()
    group.update()
    after = datetime.now()

    assert before <= dummy.a.cache.timestamp
    assert after >= dummy.a.cache.timestamp

    assert before <= dummy.b.cache.timestamp
    assert after >= dummy.b.cache.timestamp

    assert dummy.a.cache.get(get_if_invalid=False) == 0
    assert dummy.b.cache.get(get_if_invalid=False) == 0


def test_get_group_param_updates_cache_of_other_param():
    dummy = Dummy("dummy")

    assert dummy.a.cache.timestamp is None
    assert dummy.b.cache.timestamp is None
    assert dummy.a.cache.get(get_if_invalid=False) is None
    assert dummy.b.cache.get(get_if_invalid=False) is None

    before = datetime.now()
    assert dummy.a.get() == 0
    after = datetime.now()

    assert before <= dummy.a.cache.timestamp
    assert after >= dummy.a.cache.timestamp

    assert before <= dummy.b.cache.timestamp
    assert after >= dummy.b.cache.timestamp

    assert dummy.a.cache.get(get_if_invalid=False) == 0
    assert dummy.b.cache.get(get_if_invalid=False) == 0


def test_set_group_param_updates_cache_of_other_param():
    dummy = Dummy("dummy")

    assert dummy.a.cache.timestamp is None
    assert dummy.b.cache.timestamp is None
    assert dummy.a.cache.get(get_if_invalid=False) is None
    assert dummy.b.cache.get(get_if_invalid=False) is None

    before = datetime.now()
    dummy.a.set(10)
    after = datetime.now()

    assert before <= dummy.a.cache.timestamp
    assert after >= dummy.a.cache.timestamp

    assert before <= dummy.b.cache.timestamp
    assert after >= dummy.b.cache.timestamp

    assert dummy.a.cache.get(get_if_invalid=False) == 10
    assert dummy.b.cache.get(get_if_invalid=False) == 0


def test_group_param_scale_is_handled():
    dummy = Dummy("dummy", scale_a=10, initial_a=1, initial_b=5)

    assert dummy.a.cache.get(get_if_invalid=False) == 1
    assert dummy.a.cache.raw_value == 10
    assert dummy.a.get() == 1

    dummy.a.set(10)

    assert dummy.a.cache.get(get_if_invalid=False) == 10
    assert dummy.a.cache.raw_value == 100
