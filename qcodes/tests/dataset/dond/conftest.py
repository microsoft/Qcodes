import matplotlib.pyplot as plt
import pytest

from qcodes import config, validators
from qcodes.parameters import Parameter


@pytest.fixture(autouse=True)
def set_tmp_output_dir(tmpdir):
    old_config = config.user.mainfolder
    try:
        config.user.mainfolder = str(tmpdir)
        yield
    finally:
        config.user.mainfolder = old_config


@pytest.fixture()
def plot_close():
    yield
    plt.close("all")


@pytest.fixture()
def _param():
    p = Parameter("simple_parameter", set_cmd=None, get_cmd=lambda: 1)
    return p


@pytest.fixture()
def _param_2():
    p = Parameter("simple_parameter_2", set_cmd=None, get_cmd=lambda: 2)
    return p


@pytest.fixture()
def _param_complex():
    p = Parameter(
        "simple_complex_parameter",
        set_cmd=None,
        get_cmd=lambda: 1 + 1j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_complex_2():
    p = Parameter(
        "simple_complex_parameter_2",
        set_cmd=None,
        get_cmd=lambda: 2 + 2j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_set():
    p = Parameter("simple_setter_parameter", set_cmd=None, get_cmd=None)
    return p


@pytest.fixture()
def _param_set_2():
    p = Parameter("simple_setter_parameter_2", set_cmd=None, get_cmd=None)
    return p


def _param_func(_p):
    """
    A private utility function.
    """
    _new_param = Parameter(
        "modified_parameter", set_cmd=None, get_cmd=lambda: _p.get() * 2
    )
    return _new_param


@pytest.fixture()
def _param_callable(_param):
    return _param_func(_param)


def test_param_callable(_param_callable) -> None:
    _param_modified = _param_callable
    assert _param_modified.get() == 2


@pytest.fixture()
def _string_callable():
    return "Call"
