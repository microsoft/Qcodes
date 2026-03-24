"""Extended tests for qcodes.parameters.function.Function."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from qcodes.parameters import Function
from qcodes.validators import Numbers, Strings


def test_function_with_callable_cmd_no_args() -> None:
    """Function with a callable cmd and no args."""
    call_count = 0

    def my_cmd() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    func = Function("reset", call_cmd=my_cmd)
    result = func()
    assert result == "ok"
    assert call_count == 1


def test_function_with_callable_cmd_and_args() -> None:
    """Function with args validation, callable cmd."""

    def my_cmd(x: float) -> float:
        return x * 2

    func = Function("double", call_cmd=my_cmd, args=[Numbers(0, 100)])
    assert func(5) == 10


def test_function_validate_wrong_arg_count() -> None:
    """validate() raises TypeError when wrong number of args."""
    func = Function("noop", call_cmd=lambda x: x, args=[Numbers()])

    with pytest.raises(TypeError, match="called with 0 args but requires 1"):
        func.validate()

    with pytest.raises(TypeError, match="called with 2 args but requires 1"):
        func.validate(1, 2)


def test_function_validate_wrong_type() -> None:
    """validate() raises error when arg fails validation."""
    func = Function("typed", call_cmd=lambda x: x, args=[Numbers(0, 10)])

    with pytest.raises(Exception):
        func.validate(100)


def test_function_validate_passes() -> None:
    """validate() succeeds with valid args."""
    func = Function("typed", call_cmd=lambda x: x, args=[Numbers(0, 10)])
    func.validate(5)


def test_function_args_must_be_validators() -> None:
    """_set_args raises TypeError for non-Validator objects."""
    with pytest.raises(TypeError, match="all args must be Validator objects"):
        Function("bad", call_cmd=lambda x: x, args=["not_a_validator"])  # type: ignore[list-item]


def test_function_short_name() -> None:
    """short_name returns the function name."""
    func = Function("my_func", call_cmd=lambda: None)
    assert func.short_name == "my_func"


def test_function_name_parts_no_instrument() -> None:
    """name_parts returns [name] when no instrument."""
    func = Function("my_func", call_cmd=lambda: None)
    assert func.name_parts == ["my_func"]


def test_function_name_parts_with_instrument_like_object() -> None:
    """name_parts uses instrument.name_parts if available."""
    mock_instr = MagicMock()
    mock_instr.name_parts = ["instr", "sub"]
    mock_instr.write = MagicMock()
    mock_instr.ask = MagicMock()

    func = Function("my_func", instrument=mock_instr, call_cmd=lambda: None)
    assert func.name_parts == ["instr", "sub", "my_func"]


def test_function_name_parts_instrument_no_name_parts() -> None:
    """name_parts falls back to instrument.name when name_parts is empty."""
    mock_instr = MagicMock()
    mock_instr.name_parts = []
    mock_instr.name = "fallback_instr"
    mock_instr.write = MagicMock()

    func = Function("my_func", instrument=mock_instr, call_cmd=lambda: None)
    assert func.name_parts == ["fallback_instr", "my_func"]


def test_function_full_name() -> None:
    """full_name joins name_parts with underscore."""
    mock_instr = MagicMock()
    mock_instr.name_parts = ["dev", "ch1"]
    mock_instr.write = MagicMock()

    func = Function("read", instrument=mock_instr, call_cmd=lambda: None)
    assert func.full_name == "dev_ch1_read"


def test_function_get_attrs() -> None:
    """get_attrs returns the expected attribute list."""
    func = Function("my_func", call_cmd=lambda: None)
    assert func.get_attrs() == ["__doc__", "_args", "_arg_count"]


def test_function_call_method() -> None:
    """call() wraps __call__."""

    def my_cmd(x: int) -> int:
        return x + 1

    func = Function("inc", call_cmd=my_cmd, args=[Numbers()])
    assert func.call(5) == 6


def test_function_docstring() -> None:
    """Custom docstring is set on function."""
    func = Function("my_func", call_cmd=lambda: None, docstring="Custom doc")
    assert func.__doc__ == "Custom doc"


def test_function_with_arg_parser_and_return_parser() -> None:
    """Function with arg_parser and return_parser via callable cmd."""
    # When using a callable cmd, parsers are not applied by Function itself
    # (they go through Command). We use a string cmd to test parsers fully.
    mock_instr = MagicMock()
    mock_instr.ask = MagicMock(return_value="42")
    mock_instr.write = MagicMock()

    func = Function(
        "measure",
        instrument=mock_instr,
        call_cmd="MEAS {}",
        args=[Numbers()],
        arg_parser=int,
        return_parser=int,
    )
    result = func(3.14)
    assert result == 42
    mock_instr.ask.assert_called_once()


def test_function_multiple_args_validation() -> None:
    """Function with multiple args validates each."""

    def my_cmd(x: Any, y: Any) -> str:
        return f"{x},{y}"

    func = Function(
        "dual",
        call_cmd=my_cmd,
        args=[Numbers(0, 10), Strings()],
    )
    result = func(5, "hello")
    assert result == "5,hello"

    with pytest.raises(Exception):
        func(5, 123)


def test_function_instrument_property() -> None:
    """Instrument property returns the bound instrument."""
    func = Function("my_func", call_cmd=lambda: None)
    assert func.instrument is None

    mock_instr = MagicMock()
    mock_instr.write = MagicMock()
    func2 = Function("my_func2", instrument=mock_instr, call_cmd=lambda: None)
    assert func2.instrument is mock_instr
