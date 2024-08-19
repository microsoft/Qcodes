from __future__ import annotations

import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBase,
    InstrumentBaseKWArgs,
    InstrumentChannel,
)
from qcodes.parameters import (
    ArrayParameter,
    MultiParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
)
from qcodes.validators import Arrays, ComplexNumbers, Numbers, OnOff, Strings
from qcodes.validators import Sequence as ValidatorSequence

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from typing_extensions import Unpack

log = logging.getLogger(__name__)


class DummyBase(Instrument):
    def get_idn(self) -> dict[str, str | None]:
        return {
            "vendor": "QCoDeS",
            "model": str(self.__class__),
            "serial": "NA",
            "firmware": "NA",
        }


class MockParabola(DummyBase):
    """
    Holds dummy parameters which are get and set able as well as provides
    some basic functions that depends on these parameters for testing
    purposes.

    This instrument is intended to be simpler than the mock model in that it
    does not emulate communications.

    It has 3 main parameters (x, y, z) in order to allow for testing of 3D
    sweeps. The function (parabola with optional noise) is chosen to allow
    testing of numerical optimizations.
    """

    def __init__(self, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]):
        super().__init__(name, **kwargs)

        # Instrument parameters
        for parname in ["x", "y", "z"]:
            self.add_parameter(
                parname,
                unit="a.u.",
                parameter_class=Parameter,
                vals=Numbers(),
                initial_value=0,
                get_cmd=None,
                set_cmd=None,
            )

        self.add_parameter(
            "noise",
            unit="a.u.",
            label="white noise amplitude",
            parameter_class=Parameter,
            vals=Numbers(),
            initial_value=0,
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter("parabola", unit="a.u.", get_cmd=self._measure_parabola)
        self.add_parameter(
            "skewed_parabola", unit="a.u.", get_cmd=self._measure_skewed_parabola
        )

    def _measure_parabola(self) -> float:
        return (
            self.x.get() ** 2
            + self.y.get() ** 2
            + self.z.get() ** 2
            + self.noise.get() * np.random.rand(1)
        )

    def _measure_skewed_parabola(self) -> float:
        """
        Adds an -x term to add a corelation between the parameters.
        """
        return (self.x.get() ** 2 + self.y.get() ** 2 + self.z.get() ** 2) * (
            1 + abs(self.y.get() - self.x.get())
        ) + self.noise.get() * np.random.rand(1)


class MockMetaParabola(InstrumentBase):
    """
    Test for a meta instrument, has a tunable gain knob

    Unlike a MockParabola, a MockMetaParabola does not have a connection, or
    access to ask_raw/write_raw, i.e. it would not be connected to a real instrument.

    It is also not tracked in the global _all_instruments list, but is still
    snapshottable in a station.
    """

    def __init__(
        self,
        name: str,
        mock_parabola_inst: MockParabola,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """
        Create a new MockMetaParabola, connected to an existing MockParabola instance.
        """
        super().__init__(name, **kwargs)
        self.mock_parabola_inst = mock_parabola_inst

        # Instrument parameters
        for parname in ["x", "y", "z"]:
            self.parameters[parname] = getattr(mock_parabola_inst, parname)
        self.add_parameter(
            "gain",
            parameter_class=Parameter,
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter("parabola", unit="a.u.", get_cmd=self._get_parabola)
        self.add_parameter(
            "skewed_parabola", unit="a.u.", get_cmd=self._get_skew_parabola
        )

    def _get_parabola(self) -> float:
        val = self.mock_parabola_inst.parabola.get()
        return val * self.gain.get()

    def _get_skew_parabola(self) -> float:
        val = self.mock_parabola_inst.skewed_parabola.get()
        return val * self.gain.get()


class DummyInstrument(DummyBase):
    def __init__(
        self,
        name: str = "dummy",
        gates: Sequence[str] = ("dac1", "dac2", "dac3"),
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """
        Create a dummy instrument that can be used for testing

        Args:
            name: name for the instrument
            gates: list of names that is used to create parameters for
                            the instrument
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, **kwargs)

        # make gates
        for _, g in enumerate(gates):
            self.add_parameter(
                g,
                parameter_class=Parameter,
                initial_value=0,
                label=f"Gate {g}",
                unit="V",
                vals=Numbers(-800, 400),
                get_cmd=None,
                set_cmd=None,
            )

        self.fixed_parameter = self.add_parameter(
            "fixed_parameter", set_cmd=None, get_cmd=None, initial_value=5
        )


class DummyFailingInstrument(DummyBase):
    def __init__(
        self,
        name: str = "dummy",
        fail: bool = True,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """
        Create a dummy instrument that fails on initialization
        that can be used for testing

        Args:
            name: name for the instrument
            fail: if true, instrument will throw a runtime error on creation.
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, **kwargs)

        if fail:
            raise RuntimeError("Failed to create instrument")


class DummyAttrInstrument(DummyBase):
    def __init__(self, name: str = "dummy", **kwargs: Unpack[InstrumentBaseKWArgs]):
        """
        Create a dummy instrument that can be used for testing.
        This instrument has its parameters declared as attributes
        and does not use add_parameter.
        """
        super().__init__(name, **kwargs)

        self.ch1 = Parameter(
            "ch1",
            instrument=self,
            initial_value=0,
            label="Gate ch1",
            unit="V",
            vals=Numbers(-800, 400),
            get_cmd=None,
            set_cmd=None,
        )

        self.ch2 = Parameter(
            "ch2",
            instrument=self,
            initial_value=0,
            label="Gate ch2",
            unit="V",
            vals=Numbers(-800, 400),
            get_cmd=None,
            set_cmd=None,
        )


class DmmExponentialParameter(Parameter):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, **kwargs)
        self._ed = self._exponential_decay(5, 0.2)
        next(self._ed)

    def get_raw(self) -> ParamRawDataType:
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        assert isinstance(self.root_instrument, DummyInstrumentWithMeasurement)
        dac = self.root_instrument._setter_instr
        val = self._ed.send(dac.ch1.cache.get())
        if self.root_instrument is not None:
            mylogger = self.root_instrument.log
        else:
            mylogger = log
        mylogger.debug("Getting raw value of parameter: %s as %s", self.full_name, val)
        return val

    @staticmethod
    def _exponential_decay(a: float, b: float) -> Generator[float, float, None]:
        """
        Yields a*exp(-b*x) where x is put in
        """
        x = 0.0
        while True:
            x = yield a * np.exp(-b * x) + 0.02 * a * np.random.randn()


class DmmGaussParameter(Parameter):
    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, **kwargs)
        self.x0 = 0.1
        self.y0 = 0.2
        self.sigma = 0.25
        self.noise: float = 0.0005
        self._gauss = self._gauss_model()
        next(self._gauss)

    def get_raw(self) -> ParamRawDataType:
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        assert isinstance(self.root_instrument, DummyInstrumentWithMeasurement)
        dac = self.root_instrument._setter_instr
        val = self._gauss.send((dac.ch1.cache.get(), dac.ch2.cache.get()))
        if self.root_instrument is not None:
            mylogger = self.root_instrument.log
        else:
            mylogger = log
        mylogger.debug("Getting raw value of parameter: %s as %s", self.full_name, val)
        return val

    def _gauss_model(self) -> Generator[float, tuple[float, float], None]:
        """
        Returns a generator sampling a gaussian. The gaussian is
        normalised such that its maximal value is simply 1
        """

        def gauss_2d(x: float, y: float) -> np.floating:
            return np.exp(
                -((self.x0 - x) ** 2 + (self.y0 - y) ** 2) / 2 / self.sigma**2
            ) * np.exp(2 * self.sigma**2)

        x = 0.0
        y = 0.0
        while True:
            noise = np.random.randn() * self.noise
            (x, y) = yield float(gauss_2d(x, y) + noise)


class DummyInstrumentWithMeasurement(DummyBase):
    def __init__(
        self,
        name: str,
        setter_instr: DummyInstrument,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(name=name, **kwargs)
        self._setter_instr = setter_instr
        self.add_parameter(
            "v1",
            parameter_class=DmmExponentialParameter,
            initial_value=0,
            label="Gate v1",
            unit="V",
            vals=Numbers(-800, 400),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "v2",
            parameter_class=DmmGaussParameter,
            initial_value=0,
            label="Gate v2",
            unit="V",
            vals=Numbers(-800, 400),
            get_cmd=None,
            set_cmd=None,
        )


class DummyChannel(InstrumentChannel):
    """
    A single dummy channel implementation
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: str,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(parent, name, **kwargs)

        self._channel = channel

        # Add the various channel parameters
        self.add_parameter(
            "temperature",
            parameter_class=Parameter,
            initial_value=0,
            label=f"Temperature_{channel}",
            unit="K",
            vals=Numbers(0, 300),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            name="dummy_multi_parameter", parameter_class=MultiSetPointParam
        )

        self.add_parameter(
            name="dummy_scalar_multi_parameter", parameter_class=MultiScalarParam
        )

        self.add_parameter(
            name="dummy_2d_multi_parameter", parameter_class=Multi2DSetPointParam
        )

        self.add_parameter(
            name="dummy_2d_multi_parameter_2",
            parameter_class=Multi2DSetPointParam2Sizes,
        )

        self.add_parameter(
            name="dummy_array_parameter", parameter_class=ArraySetPointParam
        )

        self.add_parameter(
            name="dummy_complex_array_parameter",
            parameter_class=ComplexArraySetPointParam,
        )

        self.add_parameter(
            "dummy_start",
            initial_value=0,
            unit="some unit",
            label="f start",
            vals=Numbers(0, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_stop",
            initial_value=100,
            unit="some unit",
            label="f stop",
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_n_points",
            initial_value=101,
            unit="",
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_start_2",
            initial_value=0,
            unit="some unit",
            label="f start",
            vals=Numbers(0, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_stop_2",
            initial_value=100,
            unit="some unit",
            label="f stop",
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_n_points_2",
            initial_value=101,
            unit="",
            vals=Numbers(1, 1e3),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "dummy_sp_axis",
            unit="some unit",
            label="Dummy sp axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.dummy_start,
            stopparam=self.dummy_stop,
            numpointsparam=self.dummy_n_points,
            vals=Arrays(shape=(self.dummy_n_points,)),
        )

        self.add_parameter(
            "dummy_sp_axis_2",
            unit="some unit",
            label="Dummy sp axis",
            parameter_class=GeneratedSetPoints,
            startparam=self.dummy_start_2,
            stopparam=self.dummy_stop_2,
            numpointsparam=self.dummy_n_points_2,
            vals=Arrays(shape=(self.dummy_n_points_2,)),
        )

        self.add_parameter(
            name="dummy_parameter_with_setpoints",
            label="Dummy Parameter with Setpoints",
            unit="some other unit",
            setpoints=(self.dummy_sp_axis,),
            vals=Arrays(shape=(self.dummy_n_points,)),
            parameter_class=DummyParameterWithSetpoints1D,
        )

        self.add_parameter(
            name="dummy_parameter_with_setpoints_2d",
            label="Dummy Parameter with Setpoints",
            unit="some other unit",
            setpoints=(self.dummy_sp_axis, self.dummy_sp_axis_2),
            vals=Arrays(shape=(self.dummy_n_points, self.dummy_n_points_2)),
            parameter_class=DummyParameterWithSetpoints2D,
        )

        self.add_parameter(
            name="dummy_text",
            label="Dummy text",
            unit="text unit",
            initial_value="thisisastring",
            set_cmd=None,
            vals=Strings(),
        )

        self.add_parameter(
            name="dummy_complex",
            label="Dummy complex",
            unit="complex unit",
            initial_value=1 + 1j,
            set_cmd=None,
            vals=ComplexNumbers(),
        )

        self.add_parameter(
            name="dummy_parameter_with_setpoints_complex",
            label="Dummy Parameter with Setpoints complex",
            unit="some other unit",
            setpoints=(self.dummy_sp_axis,),
            vals=Arrays(
                shape=(self.dummy_n_points,), valid_types=(np.complexfloating,)
            ),
            parameter_class=DummyParameterWithSetpointsComplex,
        )

        self.add_function(name="log_my_name", call_cmd=partial(log.debug, f"{name}"))

    def turn_on(self) -> None:
        pass


class DummyChannelInstrument(DummyBase):
    """
    Dummy instrument with channels
    """

    def __init__(
        self,
        name: str,
        channel_names: Sequence[str] | None = None,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(name, **kwargs)

        channels = ChannelList(self, "TempSensors", DummyChannel, snapshotable=False)

        if channel_names is None:
            channel_ids: Sequence[str] = ("A", "B", "C", "D", "E", "F")
            channel_names = tuple(f"Chan{chan_name}" for chan_name in channel_ids)
        else:
            channel_ids = channel_names
        for chan_name, chan_id in zip(channel_names, channel_ids):
            channel = DummyChannel(self, chan_name, chan_id)
            channels.append(channel)
            self.add_submodule(chan_id, channel)
        self.add_submodule("channels", channels.to_channel_tuple())


class DummyChannelOnlyInstrument(DummyBase):
    """
    Dummy instrument with channels that have not been added as individual submodules.
    Also use module names with _ in them to check that we can handle that.
    """

    def __init__(self, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]):
        super().__init__(name, **kwargs)

        channels = ChannelList(self, "Temp_Sensors", DummyChannel)
        channel_ids: Sequence[str] = ("A_a", "B_b", "C_c", "D_d", "E_e", "F_f")
        channel_names = tuple(f"Chan{chan_name}" for chan_name in channel_ids)

        for chan_name, chan_id in zip(channel_names, channel_ids):
            channel = DummyChannel(self, chan_name, chan_id)
            channels.append(channel)
        self.add_submodule("channels", channels.to_channel_tuple())


class MultiGetter(MultiParameter):
    """
    Test parameters with complicated return values
    instantiate with kwargs::

        MultiGetter(name1=return_val1, name2=return_val2)

    to set the names and (constant) return values of the
    pieces returned. Each return_val can be any array-like
    object
    eg::

        MultiGetter(one=1, onetwo=(1, 2))

    """

    def __init__(self, **kwargs: Any):
        names = tuple(sorted(kwargs.keys()))
        self._return = tuple(kwargs[k] for k in names)
        shapes = tuple(np.shape(v) for v in self._return)
        super().__init__(name="multigetter", names=names, shapes=shapes)

    def get_raw(self) -> ParamRawDataType:
        return self._return


class MultiSetPointParam(MultiParameter):
    """
    Multiparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "multi_setpoint_param",
        **kwargs: Any,
    ):
        shapes = ((5,), (5,))
        names = ("multi_setpoint_param_this", "multi_setpoint_param_that")
        labels = ("this label", "that label")
        units = ("this unit", "that unit")
        sp_base = tuple(np.linspace(5, 9, 5))
        setpoints = ((sp_base,), (sp_base,))
        setpoint_names = (
            ("multi_setpoint_param_this_setpoint",),
            ("multi_setpoint_param_this_setpoint",),
        )
        setpoint_labels = (("this setpoint",), ("this setpoint",))
        setpoint_units = (("this setpointunit",), ("this setpointunit",))
        super().__init__(
            name,
            names,
            shapes,
            instrument=instrument,
            labels=labels,
            units=units,
            setpoints=setpoints,
            setpoint_labels=setpoint_labels,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        items = (np.zeros(5), np.ones(5))
        return items


class Multi2DSetPointParam(MultiParameter):
    """
    Multiparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "multi_2d_setpoint_param",
        **kwargs: Any,
    ):
        shapes = ((5, 3), (5, 3))
        names = ("this", "that")
        labels = ("this label", "that label")
        units = ("this unit", "that unit")
        sp_base_1 = tuple(np.linspace(5, 9, 5))
        sp_base_2 = tuple(np.linspace(9, 11, 3))
        array_setpoints = setpoint_generator(sp_base_1, sp_base_2)
        setpoints = (array_setpoints, array_setpoints)
        setpoint_names = (
            (
                "multi_2d_setpoint_param_this_setpoint",
                "multi_2d_setpoint_param_that_setpoint",
            ),
            (
                "multi_2d_setpoint_param_this_setpoint",
                "multi_2d_setpoint_param_that_setpoint",
            ),
        )
        setpoint_labels = (
            ("this setpoint", "that setpoint"),
            ("this setpoint", "that setpoint"),
        )
        setpoint_units = (
            ("this setpointunit", "that setpointunit"),
            ("this setpointunit", "that setpointunit"),
        )
        super().__init__(
            name,
            names,
            shapes,
            instrument=instrument,
            labels=labels,
            units=units,
            setpoints=setpoints,
            setpoint_labels=setpoint_labels,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        items = (np.zeros((5, 3)), np.ones((5, 3)))
        return items


class Multi2DSetPointParam2Sizes(MultiParameter):
    """
    Multiparameter for testing containing individual parameters with different
    shapes.
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "multi_2d_setpoint_param",
        **kwargs: Any,
    ):
        shapes = ((5, 3), (2, 7))
        names = ("this_5_3", "this_2_7")
        labels = ("this label", "that label")
        units = ("this unit", "that unit")
        sp_base_1_1 = tuple(np.linspace(5, 9, 5))
        sp_base_2_1 = tuple(np.linspace(9, 11, 3))
        array_setpoints_1 = setpoint_generator(sp_base_1_1, sp_base_2_1)
        sp_base_1_2 = tuple(np.linspace(5, 9, 2))
        sp_base_2_2 = tuple(np.linspace(9, 11, 7))
        array_setpoints_2 = setpoint_generator(sp_base_1_2, sp_base_2_2)
        setpoints = (array_setpoints_1, array_setpoints_2)
        setpoint_names = (
            (
                "multi_2d_setpoint_param_this_setpoint_1",
                "multi_2d_setpoint_param_that_setpoint_1",
            ),
            (
                "multi_2d_setpoint_param_this_setpoint_2",
                "multi_2d_setpoint_param_that_setpoint_2",
            ),
        )
        setpoint_labels = (
            ("this setpoint 1", "that setpoint 1"),
            ("this setpoint 2", "that setpoint 2"),
        )
        setpoint_units = (
            ("this setpointunit", "that setpointunit"),
            ("this setpointunit", "that setpointunit"),
        )
        super().__init__(
            name,
            names,
            shapes,
            instrument=instrument,
            labels=labels,
            units=units,
            setpoints=setpoints,
            setpoint_labels=setpoint_labels,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        items = (np.zeros((5, 3)), np.ones((2, 7)))
        return items


class MultiScalarParam(MultiParameter):
    """
    Multiparameter whos elements are scalars i.e. similar to
    Parameter with no setpoints etc.
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "multiscalarparameter",
        **kwargs: Any,
    ):
        shapes = ((), ())
        names = ("thisparam", "thatparam")
        labels = ("thisparam label", "thatparam label")
        units = ("thisparam unit", "thatparam unit")
        setpoints = ((), ())
        super().__init__(
            name,
            names,
            shapes,
            instrument=instrument,
            labels=labels,
            units=units,
            setpoints=setpoints,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        items = (0, 1)
        return items


class ArraySetPointParam(ArrayParameter):
    """
    Arrayparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "array_setpoint_param",
        **kwargs: Any,
    ):
        shape = (5,)
        label = "this label"
        unit = "this unit"
        sp_base = tuple(np.linspace(5, 9, 5))
        setpoints = (sp_base,)
        setpoint_names = ("array_setpoint_param_this_setpoint",)
        setpoint_labels = ("this setpoint",)
        setpoint_units = ("this setpointunit",)
        super().__init__(
            name,
            shape,
            instrument,
            label=label,
            unit=unit,
            setpoints=setpoints,
            setpoint_labels=setpoint_labels,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        item = np.ones(5) + 1
        return item


class ComplexArraySetPointParam(ArrayParameter):
    """
    Arrayparameter that returns complex numbers
    """

    def __init__(
        self,
        instrument: InstrumentBase | None = None,
        name: str = "testparameter",
        **kwargs: Any,
    ):
        shape = (5,)
        label = "this label"
        unit = "this unit"
        sp_base = tuple(np.linspace(5, 9, 5))
        setpoints = (sp_base,)
        setpoint_names = ("this_setpoint",)
        setpoint_labels = ("this setpoint",)
        setpoint_units = ("this setpointunit",)
        super().__init__(
            name,
            shape,
            instrument,
            label=label,
            unit=unit,
            setpoints=setpoints,
            setpoint_labels=setpoint_labels,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            **kwargs,
        )

    def get_raw(self) -> ParamRawDataType:
        item = np.arange(5) - 1j * np.arange(5)
        return item


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """

    def __init__(
        self,
        startparam: Parameter,
        stopparam: Parameter,
        numpointsparam: Parameter,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._startparam = startparam
        self._stopparam = stopparam
        self._numpointsparam = numpointsparam

    def get_raw(self) -> ParamRawDataType:
        return np.linspace(
            self._startparam(), self._stopparam(), self._numpointsparam()
        )


class DummyParameterWithSetpoints1D(ParameterWithSetpoints):
    """
    Dummy parameter that returns data with a shape based on the
    `dummy_n_points` parameter in the instrument.
    """

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, DummyChannel)
        npoints = self.instrument.dummy_n_points()
        return np.random.rand(npoints)


class DummyParameterWithSetpoints2D(ParameterWithSetpoints):
    """
    Dummy parameter that returns data with a shape based on the
    `dummy_n_points` and `dummy_n_points_2` parameters in the instrument.
    """

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, DummyChannel)
        npoints = self.instrument.dummy_n_points()
        npoints_2 = self.instrument.dummy_n_points_2()
        return np.random.rand(npoints, npoints_2)


class DummyParameterWithSetpointsComplex(ParameterWithSetpoints):
    """
    Dummy parameter that returns data with a shape based on the
    `dummy_n_points` parameter in the instrument. Returns Complex values
    """

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, DummyChannel)
        npoints = self.instrument.dummy_n_points()
        return np.random.rand(npoints) + 1j * np.random.rand(npoints)


def setpoint_generator(
    *sp_bases: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray | Sequence[float], ...]:
    """
    Helper function to generate setpoints in the format that ArrayParameter
    (and MultiParameter) expects

    Args:
        *sp_bases: 1D Sequence or Numpy array of setpoints

    Returns:

    """
    setpoints: list[np.ndarray | Sequence[float]] = []
    for i, sp_base in enumerate(sp_bases):
        if i == 0:
            setpoints.append(sp_base)
        else:
            repeats = [len(sp) for sp in sp_bases[:i]]
            repeats.append(1)
            setpoints.append(np.tile(sp_base, repeats))

    return tuple(setpoints)


class SnapShotTestInstrument(DummyBase):
    """
    A highly specialized dummy instrument for testing the snapshot. Used by
    test_snapshot.py

    Args:
        name: name for the instrument
        params: parameter names. The instrument will have these as parameters
        params_to_skip: parameters to skip updating in the snapshot. Must be
            a subset of params
    """

    def __init__(
        self,
        name: str,
        params: Sequence[str] = ("v1", "v2", "v3"),
        params_to_skip: Sequence[str] = ("v2"),
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(name, **kwargs)

        if not (set(params_to_skip).issubset(params)):
            raise ValueError("Invalid input; params_to_skip must be a subset of params")

        self._params_to_skip = params_to_skip
        self._params = params

        # dict to keep track of how many time 'get' has been called on each
        # parameter. Useful for testing params_to_skip_update in the snapshot
        self._get_calls = {p: 0 for p in params}

        for p_name in params:
            self.add_parameter(
                p_name,
                label=f"{name} Label",
                unit="V",
                set_cmd=None,
                get_cmd=partial(self._getter, p_name),
            )

    def _getter(self, name: str) -> ParamRawDataType:
        val = self.parameters[name].cache.get(get_if_invalid=False)
        self._get_calls[name] += 1
        return val

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )
        return snap


class MockField(DummyBase):
    def __init__(
        self,
        name: str,
        vals: Numbers = Numbers(min_value=-1.0, max_value=1.0),
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """Mock instrument for emulating a magnetic field axis

        Args:
            name: Instrument name
            vals: Soft limits. Defaults to Numbers(min_value=-1., max_value=1.).
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name=name, **kwargs)
        self._field = 0.0
        self.add_parameter(
            "field",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="T",
            vals=vals,
            get_cmd=self.get_field,
            set_cmd=self.set_field,
        )
        self.add_parameter(
            "ramp_rate",
            parameter_class=Parameter,
            initial_value=0.1,
            unit="T/min",
            get_cmd=None,
            set_cmd=None,
        )
        self._ramp_start_time: float | None = None
        self._wait_time: float | None = None
        self._fr = self._field_ramp()
        next(self._fr)

    def get_field(self) -> float:
        """
        This method is automatically wrapped to
        provide a ``get`` method on the parameter instance.
        """
        if self._ramp_start_time:
            _time_since_start = time.time() - self._ramp_start_time
            val = self._fr.send(_time_since_start)
            self._field = val
        return self._field

    def set_field(self, value: float, block: bool = True) -> None | float:
        if self._field == value:
            return value

        wait_time = 60.0 * np.abs(self._field - value) / self.ramp_rate()
        self._wait_time = wait_time
        self._sign = np.sign(value - self._field)
        self._start_field = self._field
        self._target_field = value
        self._ramp_start_time = time.time()

        if block:
            time.sleep(wait_time)
            self._field = value
            return value
        else:
            return None

    def _field_ramp_fcn(self, _time: float) -> float:
        if self._wait_time is None:
            return self._field
        elif _time <= 0.0:
            return self._start_field
        elif _time >= self._wait_time:
            return self._target_field
        dfield = self.ramp_rate() * _time / 60.0
        return self._start_field + self._sign * dfield

    def _field_ramp(self) -> Generator[float, float, None]:
        """
        Yields field for a given point in time
        """
        time = 0.0
        while True:
            time = yield float(self._field_ramp_fcn(time))


class MockLockin(DummyBase):
    def __init__(self, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]):
        super().__init__(name=name, **kwargs)
        self.add_parameter(
            "X",
            parameter_class=Parameter,
            initial_value=1e-3,
            unit="V",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "Y",
            parameter_class=Parameter,
            initial_value=1e-5,
            unit="V",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "frequency",
            parameter_class=Parameter,
            initial_value=125.0,
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "amplitude",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="V",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "phase",
            parameter_class=Parameter,
            initial_value=0.0,
            unit="deg",
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "time_constant",
            parameter_class=Parameter,
            initial_value=1.0e-3,
            unit="s",
            get_cmd=None,
            set_cmd=None,
        )


class MockDACChannel(InstrumentChannel):
    """
    A single dummy channel implementation
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        num: str,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(parent, name, **kwargs)

        self._num = num
        self.add_parameter(
            "voltage",
            parameter_class=Parameter,
            initial_value=0.0,
            label=f"Voltage_{name}",
            unit="V",
            vals=Numbers(-2.0, 2.0),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "dac_output",
            parameter_class=Parameter,
            initial_value="off",
            vals=OnOff(),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "smc",
            parameter_class=Parameter,
            initial_value="off",
            vals=OnOff(),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "bus",
            parameter_class=Parameter,
            initial_value="off",
            vals=OnOff(),
            get_cmd=None,
            set_cmd=None,
        )
        self.add_parameter(
            "gnd",
            parameter_class=Parameter,
            initial_value="off",
            vals=OnOff(),
            get_cmd=None,
            set_cmd=None,
        )

    def channel_number(self) -> str:
        return self._num


class MockDAC(DummyBase):
    def __init__(
        self,
        name: str = "mdac",
        num_channels: int = 10,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        """
        Create a dummy instrument that can be used for testing

        Args:
            name: name for the instrument
            gates: list of names that is used to create parameters for
                            the instrument
            num_channels: Number of channels to add to the mock instrument.
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, **kwargs)

        # make gates
        channels = ChannelList(self, "channels", MockDACChannel)
        for n in range(num_channels):
            num = str(n + 1).zfill(2)
            chan_name = f"ch{num}"
            channel = MockDACChannel(parent=self, name=chan_name, num=num)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        self.add_submodule("channels", channels.to_channel_tuple())


class MockCustomChannel(InstrumentChannel):
    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        channel: str | InstrumentChannel,
        current_valid_range: Sequence[float] | None = None,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ) -> None:
        """
        A custom instrument channel emulating an existing channel.

        It adds a parameter not found in the original channel, the
        current_valid_range.

        Args:
            parent: Instrument to which the original channel belongs to,
                usually a dac.
            name: Name of channel.
            channel: The original instrument channel.
            current_valid_range: Voltage range the channel is expected to show
                interesting features. It's just an example of an additional
                parameter a regular instrument channel does not have.
            **kwargs: Forwarded to base class.
        """
        if isinstance(channel, str):
            _, channel_name = channel.split(".")
            instr_channel = getattr(parent, channel_name)
            self._dac_channel = instr_channel
        elif isinstance(channel, InstrumentChannel):
            self._dac_channel = channel
        else:
            raise ValueError('Unknown input type for "channel".')

        super().__init__(parent, name, **kwargs)

        if current_valid_range is None:
            current_valid_range = []
        super().add_parameter(
            name="current_valid_range",
            label=f"{name} valid voltage range",
            initial_value=current_valid_range,
            vals=ValidatorSequence(Numbers(), length=2),
            get_cmd=None,
            set_cmd=None,
        )

        self.add_parameter(
            "voltage",
            parameter_class=Parameter,
            initial_value=0.0,
            label=f"Voltage_{name}",
            unit="V",
            vals=Numbers(-2.0, 2.0),
            get_cmd=None,
            set_cmd=None,
        )
