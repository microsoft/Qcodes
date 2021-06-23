"""Driver examples using the :code:`@add_parameter` decorator style.

This module defines a few driver examples that use the
:func:`@add_parameter <qcodes.instrument.base.add_parameter>` decorator style
of adding parameters.

Author: Victor Negîrneac, vnegirneac@qblox.com
"""
# NB Usage examples are not included in the corresponding docstrings because their
# code is used throughout the docs with literal include, so we keep them minimal.

# Intentional module level import so that a literalinclude of the objects in this
# file is as complete as possible
import qcodes
from qcodes.instrument.parameter import ManualParameter, Parameter
from qcodes.utils import validators


class ManualInstrument(qcodes.instrument.base.Instrument):

    @qcodes.instrument.base.add_parameter
    def _parameter_time(  # NB the parameter name will be just `time`
        self,
        parameter_class=ManualParameter,
        initial_value=3,
        unit="s",
        label="Time",
        vals=validators.Numbers(),
    ) -> float:
        """Docstring of `time` parameter."""


class InstrumentWithCmds(qcodes.instrument.base.Instrument):
    """
    An instrument that needs to pass methods when instantiating its parameters
    and adding them.
    """

    @qcodes.instrument.base.add_parameter
    def _parameter_freq(  # NB the parameter name will be just `freq`
        self,
        parameter_class=Parameter,
        unit="Hz",
        label="Frequency",
        vals=validators.Numbers(),
        # `self._set_freq` will be passed to `Parameter`
        set_cmd=qcodes.instrument.base.InstanceAttr("_set_freq"),
        # `self._get_freq` will be passed to `Parameter`
        get_cmd=qcodes.instrument.base.InstanceAttr("_get_freq"),
    ) -> float:
        """Docstring of :code:`freq` parameter."""

    def _set_freq(self, value) -> None:
        # ...
        pass

    def _get_freq(self) -> float:  # pylint: disable=no-self-use
        # ...
        return 1e9 # dummy value

class InstrumentWithInitValue(qcodes.instrument.base.Instrument):

    def __init__(self, some_arg: float, *args, **kwargs):
        """
        An instrument that requires information from its :code:`__init__` in order to
        instantiate and add its parameters.
        """
        # manual control over when the parameters are added to this instance
        self._call_add_params_from_decorated_methods = False
        super().__init__(*args, **kwargs)
        self._some_arg = some_arg

        self._add_params_from_decorated_methods()

    @qcodes.instrument.base.add_parameter
    def _parameter_time(  # NB the parameter name will be just `time`
        self,
        parameter_class=ManualParameter,
        initial_value=qcodes.instrument.base.InstanceAttr("_some_arg"),
        unit="s",
        label="Time",
        vals=validators.Numbers(),
    ) -> float:
        """Docstring of `time` parameter"""


class MyInstrumentDriver(qcodes.instrument.base.Instrument):

    def __init__(self, init_freq, *args, **kwargs):
        """
        Args:
            init_freq: will define the initial value of `freq` parameter.
        """
        self._call_add_params_from_decorated_methods = False
        super().__init__(*args, **kwargs)
        self._freq = init_freq

        self._add_params_from_decorated_methods()

    @qcodes.instrument.base.add_parameter
    def _parameter_time(  # NB the parameter name will be just `time`
        self,
        parameter_class=ManualParameter,
        initial_value=3,
        unit="s",
        label="Time",
        vals=validators.Numbers(),
    ) -> float:
        """Docstring of `time` parameter"""

    @qcodes.instrument.base.add_parameter
    def _parameter_freq(  # NB the parameter name will be just `freq`
        self,
        parameter_class=Parameter,
        unit="Hz",
        label="Frequency",
        vals=validators.Numbers(),
        # `self._freq` will be passed to `Parameter`
        initial_value=qcodes.instrument.base.InstanceAttr("_freq"),
        # `self._set_freq` will be passed to `Parameter`
        set_cmd=qcodes.instrument.base.InstanceAttr("_set_freq"),
        # `self._get_freq` will be passed to `Parameter`
        get_cmd=qcodes.instrument.base.InstanceAttr("_get_freq"),
    ) -> float:
        """Docstring of :code:`freq` parameter"""

    def _set_freq(self, value) -> None:
        self._freq = value

    def _get_freq(self) -> float:
        return self._freq


class SubMyInstrumentDriver(MyInstrumentDriver):
    """
    Same as MyInstrumentDriver but overriding a parameter and adding a new one.
    """

    @qcodes.instrument.base.add_parameter
    def _parameter_time(  # NB the parameter name will be just `time`
        self,
        parameter_class=ManualParameter,
        initial_value=7,
        unit="s",
        label="Time long",
        vals=validators.Numbers(),
    ) -> float:
        """Docstring of `time` parameter in the subclass."""

    @qcodes.instrument.base.add_parameter
    def _parameter_amplitude(  # NB the parameter name will be just `amplitude`
        self,
        parameter_class=ManualParameter,
        initial_value=0.0,
        unit="V",
        label="Amplitude",
        vals=validators.Numbers(),
    ) -> float:
        """
        Docstring of `amplitude` parameter that is specific to the subclass.
        """
