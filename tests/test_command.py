from typing import Any, NoReturn

import pytest

from qcodes.parameters.command import Command, NoCommandError


class CustomError(Exception):
    pass


def test_bad_calls() -> None:
    with pytest.raises(TypeError):
        Command()  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Command(cmd="")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Command(0, "", output_parser=lambda: 1)  # type: ignore[arg-type, misc]

    with pytest.raises(TypeError):
        Command(1, "", input_parser=lambda: 1)

    with pytest.raises(TypeError):
        Command(0, cmd="", exec_str="not a function")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        Command(
            0,
            cmd=lambda: 1,
            no_cmd_function="not a function",  # type: ignore[arg-type]
        )


def test_no_cmd() -> None:
    with pytest.raises(NoCommandError):
        Command(0)

    def no_cmd_function() -> NoReturn:
        raise CustomError("no command")

    no_cmd: Command[Any, Any] = Command(0, no_cmd_function=no_cmd_function)
    with pytest.raises(CustomError):
        no_cmd()


def test_cmd_str() -> None:
    def f_now(x):
        return x + " now"

    def upper(s):
        return s.upper()

    def reversestr(s):
        return s[::-1]

    def swap(a, b):
        return b, a

    # basic exec_str
    cmd: Command[Any, Any] = Command(0, "pickles", exec_str=f_now)
    assert cmd() == "pickles now"

    # with output parsing
    cmd = Command(0, "blue", exec_str=f_now, output_parser=upper)
    assert cmd() == "BLUE NOW"

    # parameter insertion
    cmd = Command(3, "{} is {:.2f}% better than {}", exec_str=f_now)
    assert cmd("ice cream", 56.2, "cake") == "ice cream is 56.20% better than cake now"
    with pytest.raises(ValueError):
        cmd("cake", "a whole lot", "pie")

    with pytest.raises(TypeError):
        cmd("donuts", 100, "bagels", "with cream cheese")

    # input parsing
    cmd = Command(1, "eat some {}", exec_str=f_now, input_parser=upper)
    assert cmd("ice cream") == "eat some ICE CREAM now"

    # input *and* output parsing
    cmd = Command(
        1, "eat some {}", exec_str=f_now, input_parser=upper, output_parser=reversestr
    )
    assert cmd("ice cream") == "won MAERC ECI emos tae"

    # multi-input parsing, no output parsing
    cmd = Command(2, "{} and {}", exec_str=f_now, input_parser=swap)
    assert cmd("I", "you") == "you and I now"

    # multi-input parsing *and* output parsing
    cmd = Command(
        2, "{} and {}", exec_str=f_now, input_parser=swap, output_parser=upper
    )
    assert cmd("I", "you") == "YOU AND I NOW"


def test_cmd_function_1() -> None:
    def myexp(a: float, b: float) -> float:
        return a**b

    cmd: Command[float, float] = Command(2, myexp)
    assert cmd(10, 3) == 1000

    with pytest.raises(TypeError):
        Command(3, myexp)

    # with output parsing
    cmd = Command(2, myexp, output_parser=lambda x: 5 * x)
    assert cmd(10, 3) == 5000


def test_cmd_function_2() -> None:
    def myexp(a: float, b: float) -> float:
        return a**b

    # input parsing
    # since the Command class is not generic
    # in the input type this does not understand
    # that this command should only be called with
    # float/int. We ignore this below
    cmd: Command[float, float] = Command(
        1,
        abs,  # pyright: ignore
        input_parser=lambda x: x + 1,
    )
    assert cmd(-10) == 9

    # input *and* output parsing
    cmd = Command(
        1,
        abs,  # pyright: ignore
        input_parser=lambda x: x + 2,
        output_parser=lambda y: 3 * y,  # pyright: ignore
    )
    assert cmd(-6) == 12

    # multi-input parsing, no output parsing
    cmd = Command(2, myexp, input_parser=lambda x, y: (y, x))
    assert cmd(3, 10) == 1000

    # multi-input parsing *and* output parsing
    cmd = Command(
        2, myexp, input_parser=lambda x, y: (y, x), output_parser=lambda x: 10 * x
    )
    assert cmd(8, 2) == 2560
