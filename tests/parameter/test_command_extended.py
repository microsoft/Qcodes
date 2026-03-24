"""Extended tests for qcodes.parameters.command.Command covering all call_by_* methods."""

from __future__ import annotations

import pytest

from qcodes.parameters.command import Command, NoCommandError


def test_call_by_str_no_parsers() -> None:
    """String cmd + exec_str, no parsers -> call_by_str."""
    results: list[str] = []

    def exec_fn(cmd_str: str) -> str:
        results.append(cmd_str)
        return cmd_str

    cmd = Command(arg_count=1, cmd="SET {}", exec_str=exec_fn)
    result = cmd(42)
    assert result == "SET 42"
    assert results == ["SET 42"]


def test_call_by_str_zero_args() -> None:
    """String cmd with 0 args -> call_by_str with no formatting."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    cmd = Command(arg_count=0, cmd="*RST", exec_str=exec_fn)
    assert cmd() == "*RST"


def test_call_by_str_parsed_out() -> None:
    """String cmd + output_parser -> call_by_str_parsed_out."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    cmd = Command(
        arg_count=1,
        cmd="READ {}",
        exec_str=exec_fn,
        output_parser=lambda x: x.upper(),
    )
    result = cmd("ch1")
    assert result == "READ CH1"


def test_call_by_str_parsed_in() -> None:
    """String cmd + single input_parser -> call_by_str_parsed_in."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    cmd = Command(
        arg_count=1,
        cmd="SET {}",
        exec_str=exec_fn,
        input_parser=lambda x: x * 2,
    )
    result = cmd(5)
    assert result == "SET 10"


def test_call_by_str_parsed_in_out() -> None:
    """String cmd + input_parser + output_parser -> call_by_str_parsed_in_out."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    cmd = Command(
        arg_count=1,
        cmd="MEAS {}",
        exec_str=exec_fn,
        input_parser=lambda x: x + 1,
        output_parser=lambda x: f"result:{x}",
    )
    result = cmd(9)
    assert result == "result:MEAS 10"


def test_call_by_str_parsed_in2() -> None:
    """String cmd + multi-arg input_parser -> call_by_str_parsed_in2."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    def multi_parser(a: int, b: int) -> tuple[int, int]:
        return (a * 10, b * 10)

    cmd = Command(
        arg_count=2,
        cmd="SET {} {}",
        exec_str=exec_fn,
        input_parser=multi_parser,
    )
    result = cmd(3, 4)
    assert result == "SET 30 40"


def test_call_by_str_parsed_in2_out() -> None:
    """String cmd + multi-arg input_parser + output_parser."""

    def exec_fn(cmd_str: str) -> str:
        return cmd_str

    def multi_parser(a: int, b: int) -> tuple[int, int]:
        return (a + 1, b + 1)

    cmd = Command(
        arg_count=2,
        cmd="CMD {} {}",
        exec_str=exec_fn,
        input_parser=multi_parser,
        output_parser=lambda x: x.replace("CMD", "OUT"),
    )
    result = cmd(0, 1)
    assert result == "OUT 1 2"


def test_call_cmd_no_parsers() -> None:
    """Callable cmd, no parsers -> direct call."""

    def my_func(a: int) -> int:
        return a * 3

    cmd = Command(arg_count=1, cmd=my_func)
    assert cmd(7) == 21


def test_call_cmd_parsed_out() -> None:
    """Callable cmd + output_parser -> call_cmd_parsed_out."""

    def my_func(a: int) -> int:
        return a + 1

    cmd = Command(
        arg_count=1,
        cmd=my_func,
        output_parser=lambda x: x * 100,
    )
    assert cmd(5) == 600


def test_call_cmd_parsed_in() -> None:
    """Callable cmd + single input_parser -> call_cmd_parsed_in."""

    def my_func(a: int) -> int:
        return a

    cmd = Command(
        arg_count=1,
        cmd=my_func,
        input_parser=lambda x: x + 10,
    )
    assert cmd(5) == 15


def test_call_cmd_parsed_in_out() -> None:
    """Callable cmd + input_parser + output_parser -> call_cmd_parsed_in_out."""

    def my_func(a: int) -> int:
        return a * 2

    cmd = Command(
        arg_count=1,
        cmd=my_func,
        input_parser=lambda x: x + 1,
        output_parser=lambda x: x + 100,
    )
    # input_parser(3) = 4, my_func(4) = 8, output_parser(8) = 108
    assert cmd(3) == 108


def test_call_cmd_parsed_in2() -> None:
    """Callable cmd + multi-arg input_parser -> call_cmd_parsed_in2."""

    def my_func(a: int, b: int) -> int:
        return a + b

    def multi_parser(a: int, b: int) -> tuple[int, int]:
        return (a * 10, b * 10)

    cmd = Command(
        arg_count=2,
        cmd=my_func,
        input_parser=multi_parser,
    )
    assert cmd(3, 4) == 70


def test_call_cmd_parsed_in2_out() -> None:
    """Callable cmd + multi-arg input_parser + output_parser."""

    def my_func(a: int, b: int) -> int:
        return a + b

    def multi_parser(a: int, b: int) -> tuple[int, int]:
        return (a * 2, b * 3)

    cmd = Command(
        arg_count=2,
        cmd=my_func,
        input_parser=multi_parser,
        output_parser=lambda x: x * -1,
    )
    # multi_parser(5, 10) = (10, 30), my_func(10, 30) = 40, output_parser(40) = -40
    assert cmd(5, 10) == -40


def test_wrong_arg_count_raises_type_error() -> None:
    """Calling with wrong number of args raises TypeError."""

    def my_func(a: int) -> int:
        return a

    cmd = Command(arg_count=1, cmd=my_func)

    with pytest.raises(TypeError, match="command takes exactly 1 args"):
        cmd()

    with pytest.raises(TypeError, match="command takes exactly 1 args"):
        cmd(1, 2)


def test_no_command_error_when_no_cmd() -> None:
    """NoCommandError raised when no cmd and no no_cmd_function."""
    with pytest.raises(NoCommandError, match="no ``cmd`` provided"):
        Command(arg_count=0, cmd=None)


def test_no_cmd_with_no_cmd_function() -> None:
    """no_cmd_function is used as fallback when cmd is None."""

    def fallback() -> str:
        return "fallback_called"

    cmd = Command(arg_count=0, cmd=None, no_cmd_function=fallback)
    assert cmd() == "fallback_called"


def test_str_cmd_without_exec_str_raises() -> None:
    """String cmd with no exec_str raises TypeError."""
    with pytest.raises(TypeError, match="exec_str cannot be None"):
        Command(arg_count=0, cmd="*RST", exec_str=None)
