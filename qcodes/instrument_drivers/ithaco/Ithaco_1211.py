from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import MultiParameter
from qcodes.utils.validators import Enum, Bool


class CurrentParameter(MultiParameter):
    """
    Current measurement via an Ithaco preamp and a measured voltage.

    To be used when you feed a current into the Ithaco, send the Ithaco's
    output voltage to a lockin or other voltage amplifier, and you have
    the voltage reading from that amplifier as a qcodes parameter.

    ``CurrentParameter.get()`` returns ``(voltage_raw, current)``

    Args:
        measured_param (Parameter): a gettable parameter returning the
            voltage read from the Ithaco output.

        c_amp_ins (Ithaco_1211): an Ithaco instance where you manually
            maintain the present settings of the real Ithaco amp.

            Note: it should be possible to use other current preamps, if they
            define parameters ``sens`` (sensitivity, in A/V), ``sens_factor``
            (an additional gain) and ``invert`` (bool, output is inverted)

        name (str): the name of the current output. Default 'curr'.
            Also used as the name of the whole parameter.
    """
    def __init__(self, measured_param, c_amp_ins, name='curr'):
        p_name = measured_param.name

        super().__init__(name=name, names=(p_name+'_raw', name), shapes=((), ()),
                         snapshot_value=True)

        self._measured_param = measured_param
        self._instrument = c_amp_ins

        p_label = getattr(measured_param, 'label', None)
        p_unit = getattr(measured_param, 'unit', None)

        self.labels = (p_label, 'Current')
        self.units = (p_unit, 'A')

    def get(self):
        volt = self._measured_param.get()
        current = (self._instrument.sens.get() *
                   self._instrument.sens_factor.get()) * volt

        if self._instrument.invert.get():
            current *= -1

        value = (volt, current)
        self._save_val(value)
        return value


class Ithaco_1211(Instrument):
    """
    This is the qcodes driver for the Ithaco 1211 Current-preamplifier.

    This is a virtual driver only and will not talk to your instrument.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('sens',
                           initial_value=1e-8,
                           label='Sensitivity',
                           unit='A/V',
                           get_cmd=None, set_cmd=None,
                           vals=Enum(1e-11, 1e-10, 1e-09, 1e-08, 1e-07,
                                     1e-06, 1e-05, 1e-4, 1e-3))

        self.add_parameter('invert',
                           initial_value=True,
                           label='Inverted output',
                           get_cmd=None, set_cmd=None,
                           vals=Bool())

        self.add_parameter('sens_factor',
                           initial_value=1,
                           label='Sensitivity factor',
                           unit=None,
                           get_cmd=None, set_cmd=None,
                           vals=Enum(0.1, 1, 10))

        self.add_parameter('suppression',
                           initial_value=1e-7,
                           label='Suppression',
                           unit='A',
                           get_cmd=None, set_cmd=None,
                           vals=Enum(1e-10, 1e-09, 1e-08, 1e-07, 1e-06,
                                     1e-05, 1e-4, 1e-3))

        self.add_parameter('risetime',
                           initial_value=0.3,
                           label='Rise Time',
                           unit='msec',
                           get_cmd=None, set_cmd=None,
                           vals=Enum(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,
                                     100, 300, 1000))

    def get_idn(self):
        vendor = 'Ithaco (DL Instruments)'
        model = '1211'
        serial = None
        firmware = None
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}
