import logging
from typing import TYPE_CHECKING, Any

import pytest

import qcodes.validators as vals
from qcodes.instrument import Instrument
from qcodes.parameters import Function, Parameter, ParameterBase, ParamRawDataType

from .conftest import (
    GettableParam,
    VirtualParameter,
    blank_instruments,
    named_instrument,
)

if TYPE_CHECKING:
    from collections.abc import Generator

_LOG = logging.getLogger(__name__)


class LoggingInstrument(Instrument):
    def ask(self, cmd: str) -> Any:
        _LOG.info(f"Received ask str {cmd}")
        return 1

    def write(self, cmd: str) -> None:
        _LOG.info(f"Received write str {cmd}")


@pytest.fixture(name="logging_instrument")
def _make_logging_instrument() -> "Generator[LoggingInstrument, None, None]":
    inst = LoggingInstrument("logging_instr")
    yield inst
    inst.close()


def test_no_name() -> None:
    with pytest.raises(TypeError):
        Parameter()  # type: ignore[call-arg]


def test_default_attributes() -> None:
    # Test the default attributes, providing only a name
    name = "repetitions"
    p = GettableParam(name, vals=vals.Numbers())
    assert p.name == name
    assert p.label == name
    assert p.unit == ""
    assert str(p) == name

    # default validator is all numbers
    p.validate(-1000)
    with pytest.raises(TypeError):
        p.validate("not a number")

    # docstring exists, even without providing one explicitly
    assert p.__doc__ is not None
    assert name in p.__doc__

    # test snapshot_get by looking at _get_count
    # by default, snapshot_get is True, hence we expect ``get`` to be called
    assert p._get_count == 0
    snap = p.snapshot(update=True)
    assert p._get_count == 1
    snap_expected = {
        "name": name,
        "label": name,
        "unit": "",
        "value": 42,
        "raw_value": 42,
        "vals": repr(vals.Numbers()),
    }
    for k, v in snap_expected.items():
        assert snap[k] == v
    assert snap["ts"] is not None


def test_explicit_attributes() -> None:
    # Test the explicit attributes, providing everything we can
    name = "volt"
    label = "Voltage"
    unit = "V"
    docstring = "DOCS!"
    metadata = {"gain": 100}
    p = GettableParam(
        name,
        label=label,
        unit=unit,
        vals=vals.Numbers(5, 10),
        docstring=docstring,
        snapshot_get=False,
        metadata=metadata,
    )

    assert p.name == name
    assert p.label == label
    assert p.unit == unit
    assert str(p) == name

    with pytest.raises(ValueError):
        p.validate(-1000)
    p.validate(6)
    with pytest.raises(TypeError):
        p.validate("not a number")

    assert p.__doc__ is not None
    assert name in p.__doc__
    assert docstring in p.__doc__

    # test snapshot_get by looking at _get_count
    assert p._get_count == 0
    # Snapshot should not perform get since snapshot_get is False
    snap = p.snapshot(update=True)
    assert p._get_count == 0
    snap_expected = {
        "name": name,
        "label": label,
        "unit": unit,
        "vals": repr(vals.Numbers(5, 10)),
        "value": None,
        "raw_value": None,
        "ts": None,
        "metadata": metadata,
    }
    for k, v in snap_expected.items():
        assert snap[k] == v

    # attributes only available in MultiParameter
    for attr in [
        "names",
        "labels",
        "setpoints",
        "setpoint_names",
        "setpoint_labels",
        "full_names",
    ]:
        assert not hasattr(p, attr)


def test_has_set_get() -> None:
    # Create parameter that has no set_cmd, and get_cmd returns last value
    gettable_parameter = Parameter("one", set_cmd=False, get_cmd=None)
    assert hasattr(gettable_parameter, "get")
    assert gettable_parameter.gettable
    assert not hasattr(gettable_parameter, "set")
    assert not gettable_parameter.settable
    with pytest.raises(NotImplementedError):
        gettable_parameter(1)
    # Initial value is None if not explicitly set
    assert gettable_parameter() is None
    # Assert the ``cache.set`` still works for non-settable parameter
    gettable_parameter.cache.set(1)
    assert gettable_parameter() == 1

    # Create parameter that saves value during set, and has no get_cmd
    settable_parameter = Parameter("two", set_cmd=None, get_cmd=False)
    assert not hasattr(settable_parameter, "get")
    assert not settable_parameter.gettable
    assert hasattr(settable_parameter, "set")
    assert settable_parameter.settable
    with pytest.raises(NotImplementedError):
        settable_parameter()
    settable_parameter(42)

    settable_gettable_parameter = Parameter("three", set_cmd=None, get_cmd=None)
    assert hasattr(settable_gettable_parameter, "set")
    assert settable_gettable_parameter.settable
    assert hasattr(settable_gettable_parameter, "get")
    assert settable_gettable_parameter.gettable
    assert settable_gettable_parameter() is None
    settable_gettable_parameter(22)
    assert settable_gettable_parameter() == 22


def test_str_representation() -> None:
    # three cases where only name gets used for full_name
    for instrument in blank_instruments:
        p = Parameter(name="fred")
        p._instrument = instrument  # type: ignore[assignment]
        assert str(p) == "fred"

    # and finally an instrument that really has a name
    p = Parameter(name="wilma")
    p._instrument = named_instrument  # type: ignore[assignment]
    assert str(p) == "astro_wilma"


def test_bad_name() -> None:
    with pytest.raises(ValueError):
        Parameter("p with space")
    with pytest.raises(ValueError):
        Parameter("⛄")
    with pytest.raises(ValueError):
        Parameter("1")


def test_set_via_function() -> None:
    # not a use case we want to promote, but it's there...
    p = Parameter("test", get_cmd=None, set_cmd=None)

    def doubler(x: float) -> None:
        p.set(x * 2)

    f = Function("f", call_cmd=doubler, args=[vals.Numbers(-10, 10)])

    f(4)
    assert p.get() == 8
    with pytest.raises(ValueError):
        f(20)


def test_parameter_call() -> None:
    p = Parameter("test", get_cmd=None, set_cmd=None)

    p(1)

    assert p() == 1

    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        p(1, 2)  # type: ignore[call-overload]

    p(value=2)

    assert p() == 2

    with pytest.raises(TypeError, match="got multiple values for argument"):
        p(2, value=2)  # type: ignore[call-overload]


def test_parameter_set_extra_kwargs() -> None:
    class ParameterWithSetKwargs(Parameter):
        def set_raw(
            self,
            value: ParamRawDataType,
            must_be_set_true: bool = False,
        ) -> None:
            if must_be_set_true is True:
                self.cache._set_from_raw_value(value)
            else:
                raise ValueError("must_be_set_true must be True")

    p = ParameterWithSetKwargs("testparam", get_cmd=None)

    with pytest.raises(ValueError, match="must_be_set_true must be True"):
        p(1)

    with pytest.raises(ValueError, match="must_be_set_true must be True"):
        p.set(1)

    p(1, must_be_set_true=True)

    assert p() == 1

    p.set(2, must_be_set_true=True)

    assert p() == 2


def test_unknown_args_to_baseparameter_raises() -> None:
    """
    Passing an unknown kwarg to ParameterBase should trigger a TypeError
    """
    with pytest.raises(TypeError):
        _ = ParameterBase(
            name="Foo",
            instrument=None,
            snapshotable=False,  # type: ignore[call-arg]
        )


def test_underlying_instrument_for_virtual_parameter() -> None:
    p = GettableParam("base_param", vals=vals.Numbers())
    p._instrument = named_instrument  # type: ignore[assignment]
    vp = VirtualParameter("test_param", param=p)

    assert vp.underlying_instrument is named_instrument  # type: ignore[comparison-overlap]


def test_get_cmd_str_no_instrument_raises() -> None:
    with pytest.raises(
        TypeError, match="Cannot use a str get_cmd without binding to an instrument."
    ):
        Parameter(name="test", instrument=None, get_cmd="get_me")


def test_set_cmd_str_no_instrument_raises() -> None:
    with pytest.raises(
        TypeError, match="Cannot use a str set_cmd without binding to an instrument."
    ):
        Parameter(name="test", instrument=None, set_cmd="set_me")


def test_str_get_set(
    logging_instrument: LoggingInstrument, caplog: pytest.LogCaptureFixture
) -> None:
    logging_instrument.add_parameter(
        "my_param", get_cmd="my_param?", set_cmd="my_param{}"
    )

    caplog.clear()

    with caplog.at_level(logging.INFO):
        logging_instrument.my_param.get()

    assert caplog.records[0].message == "Received ask str my_param?"

    caplog.clear()

    with caplog.at_level(logging.INFO):
        logging_instrument.my_param.set(100)

    assert caplog.records[0].message == "Received write str my_param100"


def test_str_set_multi_arg(
    logging_instrument: LoggingInstrument, caplog: pytest.LogCaptureFixture
) -> None:
    logging_instrument.add_parameter("my_param", get_cmd="my_param?", set_cmd="{}{}")

    with caplog.at_level(logging.INFO):
        logging_instrument.my_param.get()

    assert caplog.records[0].message == "Received ask str my_param?"

    caplog.clear()

    with caplog.at_level(logging.INFO):
        logging_instrument.my_param.set("this_command", 2344)

    assert caplog.records[0].message == "Received write str this_command2344"
