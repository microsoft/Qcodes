"""
Test suite for instrument.base.*
"""
import pytest
import inspect

from qcodes.utils import validators
from qcodes.instrument.base import (
    InstanceAttr,
    add_parameter,
    _ADD_PARAMETER_ATTR_NAME,
    _DECORATED_METHOD_PREFIX,
)
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.decorator_style import (
    ManualInstrument,
    InstrumentWithCmds,
    InstrumentWithInitValue,
    MyInstrumentDriver,
    SubMyInstrumentDriver,
)


# @pytest.fixture(name='testdummy', scope='function')
# def _dummy_dac():
#     instrument = DummyInstrument(
#         name='testdummy', gates=['dac1', 'dac2', 'dac3'])
#     try:
#         yield instrument
#     finally:
#         instrument.close()


def test_instanceattr():
    my_attr = InstanceAttr("_some_attribute")
    assert hasattr(my_attr, "attr_name")
    assert getattr(my_attr, "attr_name", "_some_attribute")

    assert repr(my_attr) == "InstanceAttr('_some_attribute')"


def test_add_parameter_decorator():

    def _parameter_time(
        self,
        parameter_class=ManualParameter,
        initial_value=3,
        unit="s",
        label="Time",
        vals=validators.Numbers(),
    ):
        """Docstring of time par."""

    decorated_method = add_parameter(_parameter_time)

    assert hasattr(decorated_method, _ADD_PARAMETER_ATTR_NAME)
    assert getattr(decorated_method, _ADD_PARAMETER_ATTR_NAME)
    assert decorated_method.__name__ == _DECORATED_METHOD_PREFIX + "time"
    assert inspect.signature(_parameter_time) == inspect.signature(decorated_method)
    some_object = None
    with pytest.raises(RuntimeError, match="Method not intended to be called."):
        decorated_method(some_object)

    def _bad_par_name_time(
        self,
        parameter_class=ManualParameter
    ):
        """A method that is not allowed to be decorated."""

    with pytest.raises(ValueError, match=_DECORATED_METHOD_PREFIX):
        add_parameter(_bad_par_name_time)

