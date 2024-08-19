from typing import TYPE_CHECKING, Any

import pytest

import qcodes.validators as vals
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import Parameter, ParamRawDataType

if TYPE_CHECKING:
    from collections.abc import Generator


class DummyTrackingInstrument(DummyInstrument):
    def __init__(self, name: str):
        super().__init__(name)
        self.add_parameter("a", set_cmd=None, get_cmd=None)

        # These two parameters mock actual instrument parameters; when first
        # connecting to the instrument, they have the _latest["value"] None.
        # We must call get() on them to get a valid value that we can set
        # them to in the __exit__ method of the context manager
        self.add_parameter(
            "validated_param",
            initial_cache_value=None,
            set_cmd=self._vp_setter,
            get_cmd=self._vp_getter,
            vals=vals.Enum("foo", "bar"),
        )

        self.add_parameter(
            "parsed_param",
            initial_cache_value=None,
            set_cmd=self._pp_setter,
            get_cmd=self._pp_getter,
            set_parser=int,
        )

        # A parameter that is not initialized and whose cache value does not
        # pass the validator or the set parser
        self.add_parameter(
            "uninitialized_param",
            initial_cache_value=None,
            set_cmd=None,
            set_parser=int,
            vals=vals.Enum(2),
        )

        # A parameter that counts the number of times it has been set
        self.add_parameter(
            "counting_parameter", set_cmd=self._cp_setter, get_cmd=self._cp_getter
        )

        # the mocked instrument state values of validated_param and
        # parsed_param
        self._vp_value = "foo"
        self._pp_value = 42

        # the counter value for counting_parameter
        self._cp_counter = 0
        self._cp_get_counter = 0

    def _vp_getter(self) -> str:
        return self._vp_value

    def _vp_setter(self, value: str) -> None:
        self._vp_value = value

    def _pp_getter(self) -> ParamRawDataType:
        return self._pp_value

    def _pp_setter(self, value: ParamRawDataType) -> None:
        self._pp_value = value

    def _cp_setter(self, value: ParamRawDataType) -> None:
        self._cp_counter += 1

    def _cp_getter(self) -> ParamRawDataType:
        self._cp_get_counter += 1
        return self.counting_parameter.cache._value


@pytest.fixture(name="instrument")
def _make_instrument() -> "Generator[DummyTrackingInstrument, None, None]":
    instrument = DummyTrackingInstrument("dummy_holder")
    try:
        yield instrument
    finally:
        instrument.close()


def test_set_to_none_when_parameter_is_not_captured_yet(
    instrument: DummyTrackingInstrument,
) -> None:
    counting_parameter = instrument.counting_parameter
    # Pre-conditions:
    assert instrument._cp_counter == 0
    assert instrument._cp_get_counter == 0
    assert counting_parameter.cache._value is None
    assert counting_parameter.get_latest.get_timestamp() is None

    with counting_parameter.set_to(None):
        # The value should not change
        assert counting_parameter.cache._value is None
        # The timestamp of the latest value should not be None anymore
        assert counting_parameter.get_latest.get_timestamp() is not None
        # Set method is not called
        assert instrument._cp_counter == 0
        # Get method is called once
        assert instrument._cp_get_counter == 1

    # The value should not change
    assert counting_parameter.cache._value is None
    # The timestamp of the latest value should still not be None
    assert counting_parameter.get_latest.get_timestamp() is not None
    # Set method is still not called
    assert instrument._cp_counter == 0
    # Get method is still called once
    assert instrument._cp_get_counter == 1


def test_set_to_none_for_not_captured_parameter_but_instrument_has_value() -> None:
    # representing instrument here
    instr_value = "something"
    set_counter = 0

    def set_instr_value(value: Any) -> None:
        nonlocal instr_value, set_counter
        instr_value = value
        set_counter += 1

    # make a parameter that is linked to an instrument
    p = Parameter(
        "p",
        set_cmd=set_instr_value,
        get_cmd=lambda: instr_value,
        val_mapping={"foo": "something", None: "nothing"},
    )

    # pre-conditions
    assert p.cache._value is None  # type: ignore[attr-defined]
    assert p.cache.raw_value is None
    assert p.cache.timestamp is None
    assert set_counter == 0

    with p.set_to(None):
        # assertions after entering the context
        assert set_counter == 1
        assert instr_value == "nothing"
        assert p.cache._value is None  # type: ignore[attr-defined]
        assert p.cache.raw_value == "nothing"
        assert p.cache.timestamp is not None

    # assertions after exiting the context
    assert set_counter == 2
    assert instr_value == "something"
    assert p.cache._value == "foo"  # pyright: ignore
    assert p.cache.raw_value == "something"
    assert p.cache.timestamp is not None


def test_none_value(instrument: DummyTrackingInstrument) -> None:
    with instrument.a.set_to(3):
        assert instrument.a.get_latest.get_timestamp() is not None
        assert instrument.a.get() == 3
    assert instrument.a.get() is None
    assert instrument.a.get_latest.get_timestamp() is not None


def test_context(instrument: DummyTrackingInstrument) -> None:
    instrument.a.set(2)

    with instrument.a.set_to(3):
        assert instrument.a.get() == 3
    assert instrument.a.get() == 2

    # The cached value does not pass the validator (since it was not
    # initialized). Make sure no exception is raised when __exit__ing the
    # context
    with instrument.uninitialized_param.set_to(2):
        assert instrument.uninitialized_param.get() == 2


def test_validated_param(instrument: DummyTrackingInstrument) -> None:
    assert instrument.parsed_param.cache._value is None
    assert instrument.validated_param.get_latest() == "foo"
    with instrument.validated_param.set_to("bar"):
        assert instrument.validated_param.get() == "bar"
    assert instrument.validated_param.get_latest() == "foo"
    assert instrument.validated_param.get() == "foo"


def test_parsed_param(instrument: DummyTrackingInstrument) -> None:
    assert instrument.parsed_param.cache._value is None
    assert instrument.parsed_param.get_latest() == 42
    with instrument.parsed_param.set_to(1):
        assert instrument.parsed_param.get() == 1
    assert instrument.parsed_param.get_latest() == 42
    assert instrument.parsed_param.get() == 42


def test_number_of_set_calls(instrument: DummyTrackingInstrument) -> None:
    """
    Test that with param.set_to(X) does not perform any calls to set if
    the parameter already had the value X
    """
    assert instrument._cp_counter == 0
    instrument.counting_parameter(1)
    assert instrument._cp_counter == 1

    with instrument.counting_parameter.set_to(2):
        pass
    assert instrument._cp_counter == 3

    with instrument.counting_parameter.set_to(1):
        pass
    assert instrument._cp_counter == 3


def test_value_modified_between_context_create_and_enter(
    instrument: DummyTrackingInstrument,
) -> None:
    p = instrument.a
    p.set(2)
    ctx = p.set_to(5)
    # the parameter value is changed after the context has been created
    # this is the value it should return to after exit.
    p.set(3)
    with ctx:
        assert p() == 5
    assert p() == 3


def test_disallow_changes(instrument: DummyTrackingInstrument) -> None:
    instrument.a.set(2)

    with instrument.a.set_to(3, allow_changes=False):
        assert instrument.a() == 3
        assert not instrument.a.settable
        with pytest.raises(TypeError):
            instrument.a.set(5)

    assert instrument.a.settable
    assert instrument.a() == 2


def test_allow_changes(instrument: DummyTrackingInstrument) -> None:
    p = instrument.a
    p.set(2)
    with p.set_to(3, allow_changes=True):
        assert p.settable
        assert p() == 3
        p.set(5)
        assert p() == 5

    assert p.settable
    assert p() == 2

    # check that the value gets restored even if entering the context
    # with the current value
    with instrument.a.set_to(2, allow_changes=True):
        assert p() == 2
        p(5)
        assert p() == 5

    assert p.settable
    assert p() == 2


def test_reset_at_exit(instrument: DummyTrackingInstrument) -> None:
    p = instrument.a
    p.set(2)
    with p.restore_at_exit():
        p.set(5)
    assert p() == 2


def test_reset_at_exit_with_allow_changes_false(
    instrument: DummyTrackingInstrument,
) -> None:
    p = instrument.a
    p.set(2)
    with p.restore_at_exit(allow_changes=False):
        with pytest.raises(TypeError):
            p.set(5)
    assert p() == 2
