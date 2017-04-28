from qcodes.instrument.base import Instrument
from qcodes import validators as validator

from functools import partial
import warnings
import json

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

    def __init__(self, name, device_name, **kwargs):
        super().__init__(name, **kwargs)

        self.device_name = device_name

        self.channels = 8

        try:
            with open('NI_voltages_{}.json'.format(device_name)) as data_file:
                self.__voltage = json.load(data_file)
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            self.__voltage = [0] * self.channels

        print('Please read the following warning message:')

        warnings.warn('The last known output values are: {} Please check these values and make sure they correspond '
                      'to the actual output of the PXIe-4322 module. Any difference between stored value and actual '
                      'value WILL cause sudden jumps in output.'.format(self.__voltage), UserWarning)

        for i in range(self.channels):
            self.add_parameter('voltage_channel_{}'.format(i),
                               label='voltage channel {}'.format(i),
                               unit='V',
                               set_cmd=partial(self.set_voltage, channel=i),
                               get_cmd=partial(self.get_voltage, channel=i),
                               docstring='The DC output voltage of channel {}'.format(i),
                               vals=validator.Numbers(-16, 16))

    def set_voltage(self, voltage, channel, save_to_file=True):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(self.device_name, channel),
                                                 min_val=-16.0, max_val=16.0)
            task.write(voltage, auto_start=True)
            self.__voltage[channel] = voltage
            if save_to_file:
                with open('NI_voltages_{}.json'.format(self.device_name), 'w') as output_file:
                    json.dump(self.__voltage, output_file, ensure_ascii=False)

    def get_voltage(self, channel):
        return self.__voltage[channel]
