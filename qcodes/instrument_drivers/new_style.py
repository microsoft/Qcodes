"""
This module defines a few driver examples that use the new style of adding parameters.

.. seealso::

    :func:`qcodes.intrument.base.add_parameter` decorator.
"""

import qcodes
from qcodes import ManualParameter, Parameter
from qcodes import validators

class MyInstrumentDriver(qcodes.instrument.base.Instrument):
    r"""
    MyInstrumentDriver docstring

    .. code-block:: python

        from qcodes.instrument_drivers.new_style import MyInstrumentDriver

        instr = MyInstrumentDriver(name="instr", special_par=8)
        instr.freq(10)
        instr.print_readable_snapshot()
        print("\ninstr.time.label: ", instr.time.label)

    """

    def __init__(self, special_par, *args, **kwargs):
        self._call_add_params_from_decorated_methods = False
        super().__init__(*args, **kwargs)
        self._freq = special_par

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
    r"""Same as MyInstrumentDriver but overriding a parameter

    .. code-block:: python

        from qcodes.instrument_drivers.new_style import SubMyInstrumentDriver

        sub_instr = SubMyInstrumentDriver(name="sub_instr", special_par=99)
        sub_instr.time(sub_instr.time() * 2)
        sub_instr.print_readable_snapshot()
        print("\nsub_instr.time.label: ", sub_instr.time.label)

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
        """Docstring of `time` parameter in the subclass"""


class ManualInstrument(qcodes.instrument.base.Instrument):
    """
    Docstring of ManualInstrument class.
    """

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
        """Docstring of :code:`freq` parameter"""

    def _set_freq(self, value) -> None:
        # ...
        pass

    def _get_freq(self) -> float:
        # ...
        return 1e9 # dummy value

class InstrumentWithInitValue(qcodes.instrument.base.Instrument):
    """
    An instrument that requires information from its :code:`__init__` in order to
    instantiate and add its parameters.
    """

    def __init__(self, some_arg: float, *args, **kwargs):
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
