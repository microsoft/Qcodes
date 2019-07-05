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

    def __init__(self, name, device_name, file_path, file_update_period=5, **kwargs):
        super().__init__(name, **kwargs)

        self.device_name = device_name

        self.channels = 8

        self.voltage_file = file_path + 'NI_voltages_{}.json'.format(device_name)
        self._voltages_changed = False
        try:
            os.mkdir(file_path)
        except:
            pass

        try:
            with open(self.voltage_file) as data_file:
                latest_voltages = json.load(data_file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logger.warning('No latest voltages found')
            latest_voltages = [0] * self.channels

        logger.warning('The last known output values are: {} Please check these values and make sure they correspond '
                      'to the actual output of the PXIe-4322 module. Any difference between stored value and actual '
                      'value WILL cause sudden jumps in output.'.format(latest_voltages))

        for i, latest_voltage in enumerate(latest_voltages):
            self.add_parameter(f'voltage_channel_{i}',
                               label=f'voltage channel {i}',
                               unit='V',
                               initial_value=latest_voltage,
                               set_cmd=partial(self.set_voltage, channel=i),
                               get_cmd=None,
                               docstring=f'The DC output voltage of channel {i}',
                               vals=validator.Numbers(-16, 16))

        # Start writing voltages to disk
        self._start_updating_file(file_update_period)

    def _start_updating_file(self, update_period):
        if hasattr(self, '_voltages_changed') and self._voltages_changed:
            self._voltages_changed = False
            self._write_voltages_to_file()
        t = threading.Timer(update_period, partial(self._start_updating_file, update_period))
        t.start()

    def set_voltage(self, voltage, channel):
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/ao{channel}', min_val=-16.0, max_val=16.0)
            task.write(voltage)

            self._voltages_changed = True

    def _write_voltages_to_file(self):
        with open(self.voltage_file, 'w') as output_file:
            voltages = [self.parameters[f'voltage_channel_{i}'].raw_value for i in range(self.channels)]
            json.dump(voltages, output_file, ensure_ascii=False)
