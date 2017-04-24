from qcodes.instrument.base import Instrument
from qcodes import validators as validator

from functools import partial

try:
    import nidaqmx
except ImportError:
    raise ImportError('to use the National Instrument PXIe-4322 driver, please install the nidaqmx package'
                      '(https://github.com/ni/nidaqmx-python)')


class PXIe_4322(Instrument):
    """
    This is the QCoDeS driver for the National Instrument PXIe-4322 Analog Output Module.

    The current version of this driver only allows using the PXIe-4322 as a DC Voltage Output.

    This driver makes use of the Python API for interacting with the NI-DAQmx driver. Both the NI-DAQmx driver and
    the nidaqmx package need to be installed in order to use this QCoDeS driver.
    """

    def __init__(self, name, chassis, slot, **kwargs):
        super().__init__(name, **kwargs)

        self.chassis = chassis
        self.slot = slot

        self.channels = 8

        self.__voltage = [0] * self.channels

        for i in range(self.channels):
            self.add_parameter('voltage_channel_{}'.format(i),
                               label='voltage channel {}'.format(i),
                               unit='Hz',
                               set_cmd=partial(self.set_voltage, channel=i),
                               get_cmd=partial(self.get_voltage, channel=i),
                               docstring='The DC output voltage of channel {}'.format(i),
                               vals=validator.Numbers(-16, 16))

    def set_voltage(self, voltage, channel):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('PXI{}Slot{}/ao{}'.format(self.chassis, self.slot, channel))
            task.write(voltage, auto_start=True)
            self.__voltage[channel] = voltage

    def get_voltage(self, channel):
        return self.__voltage[channel]
