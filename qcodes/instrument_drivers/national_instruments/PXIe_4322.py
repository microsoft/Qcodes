from qcodes.instrument.base import Instrument
from qcodes import validators as validator
from functools import partial
import logging
import warnings
import json
import math
from timeit import default_timer as timer
from time import sleep
from os import mkdir
import threading

try:
    import nidaqmx
except ImportError:
    raise ImportError('to use the National Instrument PXIe-4322 driver, please install the nidaqmx package'
                      '(https://github.com/ni/nidaqmx-python)')

logger = logging.getLogger(__name__)

class PXIe_4322(Instrument):
    """
    This is the QCoDeS driver for the National Instrument PXIe-4322 Analog Output Module.

    The current version of this driver only allows using the PXIe-4322 as a DC Voltage Output.

    This driver makes use of the Python API for interacting with the NI-DAQmx driver. Both the NI-DAQmx driver and
    the nidaqmx package need to be installed in order to use this QCoDeS driver.
    """

    def __init__(self, name, device_name, file_path, file_update_period=10, step_size=0.01, step_rate=10, **kwargs):
        super().__init__(name, **kwargs)

        self.device_name = device_name

        self.channels = 8

        self.step_size = step_size
        self.step_rate = step_rate
        self.step_delay = 1/step_rate
        self.voltage_file = file_path + 'NI_voltages_{}.json'.format(device_name)
        self._voltages_changed = False
        try:
            os.mkdir(file_path)
        except:
            pass

        try:
            with open(self.voltage_file) as data_file:
                self.__voltage = json.load(data_file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            self.__voltage = [0] * self.channels

        # t = threading.Timer(file_update_period, self._write_voltages_to_file)
        # t.start()

        print('Please read the following warning message:')

        logger.warning('The last known output values are: {} Please check these values and make sure they correspond '
                      'to the actual output of the PXIe-4322 module. Any difference between stored value and actual '
                      'value WILL cause sudden jumps in output.'.format(self.__voltage))

        for i in range(self.channels):
            self.add_parameter('voltage_channel_{}'.format(i),
                               label='voltage channel {}'.format(i),
                               unit='V',
                               set_cmd=partial(self.set_voltage, channel=i),
                               get_cmd=partial(self.get_voltage, channel=i),
                               docstring='The DC output voltage of channel {}'.format(i),
                               vals=validator.Numbers(-16, 16))

        # Start writing voltages to disk
        self._start_updating_file(file_update_period)

    def _start_updating_file(self, update_period):
        if (self._voltages_changed):
            self._voltages_changed = False
            self._write_voltages_to_file()
        t = threading.Timer(update_period, partial(self._start_updating_file, update_period))
        t.start()

    def set_voltage(self, voltage, channel, verbose=False):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(self.device_name, channel),
                                                 min_val=-16.0, max_val=16.0)

            if abs(voltage - self.__voltage[channel]) > self.step_size and not \
                    math.isclose(abs(voltage - self.__voltage[channel]), self.step_size, rel_tol=1e-5):
                if (voltage - self.__voltage[channel]) < 0:
                    step = -self.step_size
                else:
                    step = self.step_size
                for voltage_step in frange(self.__voltage[channel], voltage, step):
                    t_start = timer()
                    task.write(voltage_step)
                    if verbose:
                        print('Current gate {} value: {:.2f}'.format(channel, voltage_step), end='\r', flush=True)
                    else:
                        logger.debug('Current gate {} value: {:.2f}'.format(channel, voltage_step))
                    t_stop = timer()
                    sleep(max(self.step_delay-(t_stop-t_start), 0.0))

            task.write(voltage)
            if verbose:
                print('Current gate {} value: {:.2f}'.format(channel, voltage), end='\r', flush=True)
            else:
                logger.debug('Current gate {} value: {:.2f}'.format(channel, voltage))
            self.__voltage[channel] = voltage
            self._voltages_changed = True

    def get_voltage(self, channel):
        return self.__voltage[channel]

    def _write_voltages_to_file(self):
        with open(self.voltage_file, 'w') as output_file:
            json.dump(self.__voltage, output_file, ensure_ascii=False)

    def set_gates_simultaneously(self, gate_values):
        assert len(gate_values) == self.channels, 'number of values in gate_values list ({}) must be same as number ' \
                                                  'of channels: {}'.format(len(gate_values), self.channels)

        diff = [gate_values[i] - self.__voltage[i] for i in range(len(self.__voltage))]
        step = [self.step_size if diff_i >= 0 else -self.step_size for diff_i in diff]
        volt_steps = [frange(self.__voltage[i], gate_values[i], step[i]) for i in range(len(self.__voltage))]
        number_of_steps = [len(volt_steps[i]) for i in range(len(volt_steps))]

        channel_mask = [True] * self.channels

        for i in range(max(number_of_steps)):
            t_start = timer()
            for chan in range(self.channels):
                if channel_mask[chan]:
                    try:
                        voltage = volt_steps[chan][i]
                        self.set_voltage(voltage, chan, verbose=False)
                    except IndexError:
                        channel_mask[chan] = False
                        pass
            t_stop = timer()
            voltage_str = ''
            for item in self.__voltage:
                voltage_str += '{:.2f}, '.format(item)
            voltage_str = voltage_str[:-2]
            print('Current gate values: {}'.format(voltage_str), end='\r', flush=True)
            sleep(max(self.step_delay - (t_stop - t_start), 0.0))

        for i in range(self.channels):
            self.set_voltage(gate_values[i], i, verbose=False)
        voltage_str = ''
        for item in self.__voltage:
            voltage_str += '{:.2f}, '.format(item)
        voltage_str = voltage_str[:-2]
        print('Current gate values: {}'.format(voltage_str), end='\r', flush=True)
        logger.debug('Current gate values: {}'.format(voltage_str))

    def ramp_all_to_zero(self):
        gate_values = [0.0]*self.channels
        self.set_gates_simultaneously(gate_values)


def frange(start, stop, step):
    if stop is None:
        stop, start = start, 0.
    else:
        start = float(start)

    count = int(math.ceil((stop - start) / step))
    return [start + n * step for n in range(count)]

