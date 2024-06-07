from typing import TYPE_CHECKING, Optional

from qcodes.instrument import Instrument, InstrumentBaseKWArgs
from qcodes.parameters import MultiParameter, Parameter, ParamRawDataType
from qcodes.validators import Bool, Enum

if TYPE_CHECKING:
    from typing_extensions import Unpack


class VoltageParameter(MultiParameter):
    """
    Amplified voltage measurement via an SR560 preamp and a measured voltage.

    To be used when you feed a voltage into an SR560, send the SR560's
    output voltage to a lockin or other voltage amplifier, and you have
    the voltage reading from that amplifier as a qcodes parameter.

    ``VoltageParameter.get()`` returns ``(voltage_raw, voltage)``

    Args:
        measured_param: a gettable parameter returning the
            voltage read from the SR560 output.
        v_amp_ins: an SR560 instance where you manually
            maintain the present settings of the real SR560 amp.

            Note: it should be possible to use other voltage preamps, if they
            define parameters ``gain`` (V_out / V_in) and ``invert``
            (bool, output is inverted)

        name: the name of the current output. Default 'volt'.
            Also used as the name of the whole parameter.
    """
    def __init__(self,
                 measured_param: Parameter,
                 v_amp_ins: "SR560",
                 name: str = 'volt',
                 snapshot_value: bool = True):
        p_name = measured_param.name

        super().__init__(name=name,
                         names=(p_name+'_raw', name),
                         shapes=((), ()),
                         instrument=v_amp_ins)

        self._measured_param = measured_param

        p_label = getattr(measured_param, "label", "")
        p_unit = getattr(measured_param, "unit", "")

        self.labels = (p_label, 'Voltage')
        self.units = (p_unit, 'V')

    def get_raw(self) -> tuple[ParamRawDataType, ParamRawDataType]:
        volt = self._measured_param.get()
        assert isinstance(self.instrument, SR560)
        volt_amp = volt / self.instrument.gain.get()

        if self.instrument.invert.get():
            volt_amp *= -1

        value = (volt, volt_amp)
        return value


class SR560(Instrument):
    """
    QCoDeS driver for the Stanford Research Systems SR560 Voltage-preamplifier.

    This is a virtual driver only and will not talk to your instrument.

    Note:
        - The ``cutoff_lo`` and ``cutoff_hi`` parameters will interact with
          each other on the instrument (hi cannot be <= lo) but this is not
          managed here, you must ensure yourself that both are correct whenever
          you change one of them.
        - ``gain`` has a vernier setting, which does not yield a well-defined
          output. We restrict this driver to only the predefined gain values.

    """

    def __init__(self, name: str, **kwargs: "Unpack[InstrumentBaseKWArgs]"):
        super().__init__(name, **kwargs)

        cutoffs = ['DC', 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000,
                   3000, 10000, 30000, 100000, 300000, 1000000]

        gains = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000]

        self.cutoff_lo: Parameter = self.add_parameter(
            "cutoff_lo",
            get_cmd=None,
            set_cmd=None,
            initial_value="DC",
            label="High pass",
            unit="Hz",
            vals=Enum(*cutoffs),
        )
        """Parameter cutoff_lo"""

        self.cutoff_hi: Parameter = self.add_parameter(
            "cutoff_hi",
            get_cmd=None,
            set_cmd=None,
            initial_value=1e6,
            label="Low pass",
            unit="Hz",
            vals=Enum(*cutoffs),
        )
        """Parameter cutoff_hi"""

        self.invert: Parameter = self.add_parameter(
            "invert",
            get_cmd=None,
            set_cmd=None,
            initial_value=True,
            label="Inverted output",
            vals=Bool(),
        )
        """Parameter invert"""

        self.gain: Parameter = self.add_parameter(
            "gain",
            get_cmd=None,
            set_cmd=None,
            initial_value=10,
            label="Gain",
            unit=None,
            vals=Enum(*gains),
        )
        """Parameter gain"""

    def get_idn(self) -> dict[str, Optional[str]]:
        vendor = 'Stanford Research Systems'
        model = 'SR560'
        serial = None
        firmware = None

        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
