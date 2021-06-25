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
from qcodes.instrument.base import Instrument
from qcodes.instrument_drivers.decorator_style import (
    ManualInstrument,
    MyInstrumentDriver,
    SubMyInstrumentDriver,
)

# pylint: disable=unused-import
from .test_instrument import _close_before_and_after

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

    with pytest.raises(RuntimeError, match="Method not intended to be called."):
        decorated_method(None)

    def _bad_par_name(
        self,
        parameter_class=ManualParameter
    ):
        """A method that is not allowed to be decorated."""

    with pytest.raises(ValueError, match=_DECORATED_METHOD_PREFIX):
        add_parameter(_bad_par_name)

    def _parameter_time_name(
        self,
        parameter_class=ManualParameter,
        name="time"
    ):
        pass

    with pytest.raises(
        RuntimeError,
        match="`name` kwarg was provided to '_parameter_time_name' method."
    ):
        add_parameter(_parameter_time_name)

    def _parameter_time_docstring(
        self,
        parameter_class=ManualParameter,
        docstring="time"
    ):
        pass

    with pytest.raises(
        RuntimeError,
        match="`docstring` kwarg was provided to '_parameter_time_docstring' method."
    ):
        add_parameter(_parameter_time_docstring)

def test_no_attr_on_intrument(close_before_and_after):
    m_instrument = ManualInstrument("m_instrument")
    assert not hasattr(m_instrument, "_call_add_params_from_decorated_methods")

def test_parameter(close_before_and_after):
    m_instrument = ManualInstrument("m_instrument")

    # Check parameter works as expected
    m_instrument.time(m_instrument.time())

    assert isinstance(m_instrument.time, ManualParameter)
    assert m_instrument.time.label == "Time"
    assert m_instrument.time.unit == "s"
    assert isinstance(m_instrument.time.vals, validators.Numbers)
    assert m_instrument.time.__doc__.startswith(
        ManualInstrument._parameter_time.__doc__
    )

def test_add_params_from_decorated_methods_called(mocker, close_before_and_after):

    spy = mocker.patch.object(
        ManualInstrument,
        "_add_params_from_decorated_methods",
    )

    spy_add_parameter = mocker.patch.object(
        ManualInstrument,
        "add_parameter",
    )

    ManualInstrument("m_instrument")
    assert spy.call_count == 1
    assert spy_add_parameter.call_count == 1


def test_instance_and_init_arg(close_before_and_after):

    m_instrument = MyInstrumentDriver(name="m_instrument", init_freq=20e3)
    assert not getattr(m_instrument, "_call_add_params_from_decorated_methods")
    assert m_instrument.freq() == 20e3  # calls `self._get_freq`
    assert m_instrument.freq() == m_instrument._freq

def test_class_inheritance_override_parameter(close_before_and_after):

    m_instrument = MyInstrumentDriver(name="m_instrument", init_freq=20e3)
    assert hasattr(m_instrument, "freq")
    assert hasattr(m_instrument, "time")


    sm_instrument = SubMyInstrumentDriver(name="sm_instrument", init_freq=10e3)
    assert hasattr(sm_instrument, "freq")
    assert hasattr(sm_instrument, "time")
    assert hasattr(sm_instrument, "amplitude")

    # overridden parameter with distinct initial value
    assert m_instrument.time() != sm_instrument.time()

def test_non_exiting_instance_attr():
    class BadInstrument(Instrument):
        @add_parameter
        def _parameter_amp(
            self,
            parameter_class=ManualParameter,
            initial_value=InstanceAttr("_non_existing_attr")
        ):
            pass

    with pytest.raises(AttributeError):
        BadInstrument(name="raises")
