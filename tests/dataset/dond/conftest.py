from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from qcodes import config, validators
from qcodes.parameters import Parameter

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest._py.path import LocalPath


@pytest.fixture(autouse=True)
def set_tmp_output_dir(tmpdir: "LocalPath") -> "Generator[None, None, None]":
    old_config = config.user.mainfolder
    try:
        config.user.mainfolder = str(tmpdir)
        yield
    finally:
        config.user.mainfolder = old_config


@pytest.fixture()
def plot_close() -> "Generator[None, None, None]":
    yield
    plt.close("all")


@pytest.fixture()
def _param() -> Parameter[int]:
    p = Parameter("simple_parameter", set_cmd=None, get_cmd=lambda: 1)
    return p


@pytest.fixture()
def _param_2() -> Parameter[int]:
    p = Parameter("simple_parameter_2", set_cmd=None, get_cmd=lambda: 2)
    return p


@pytest.fixture()
def _param_complex() -> Parameter[complex]:
    p = Parameter(
        "simple_complex_parameter",
        set_cmd=None,
        get_cmd=lambda: 1 + 1j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_complex_2() -> Parameter[complex]:
    p = Parameter(
        "simple_complex_parameter_2",
        set_cmd=None,
        get_cmd=lambda: 2 + 2j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_set() -> Parameter[float | None]:
    p = Parameter("simple_setter_parameter", set_cmd=None, get_cmd=None)
    return p


@pytest.fixture()
def _param_set_2() -> Parameter[float | None]:
    p = Parameter("simple_setter_parameter_2", set_cmd=None, get_cmd=None)
    return p


def _param_func(_p: Parameter[int]) -> Parameter[int]:
    """
    A private utility function.
    """
    _new_param = Parameter(
        "modified_parameter", set_cmd=None, get_cmd=lambda: _p.get() * 2
    )
    return _new_param


@pytest.fixture()
def _param_callable(_param: Parameter[int]) -> Parameter[int]:
    return _param_func(_param)


def test_param_callable(_param_callable: Parameter[int]) -> None:
    _param_modified = _param_callable
    assert _param_modified.get() == 2


@pytest.fixture()
def _string_callable() -> str:
    return "Call"
