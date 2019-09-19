import numpy as np
import logging
import json
import pyperclip
from time import time

from qcodes import VisaInstrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators as vals
from time import sleep


logger = logging.getLogger(__name__)
cmdbase = "TERM LF\nFLSH\nFLOQ\n"

class SIM928(Parameter):
    """
    This is the parameter class for the SIM928 rechargeable isolated voltage source module

    Args:
        channel (int): SIM900 channel for the SIM928 module

        name (Optional[str]): Module name (default 'channel_{channel}')

        max_voltage (Optional[float]): Maximum voltage (default 20)
    """
    def __init__(self, channel, name=None, max_voltage=20, step=0.001,
                 inter_delay=0.035, t_recheck_cycles=600, **kwargs):
        if not name:
            name = 'channel_{}'.format(channel)

        self.t_last_cycle_check = None
        self.t_recheck_cycles = t_recheck_cycles
        self._latest_charge_cycles = None

        self.send_cmd = cmdbase + "SNDT {:d} ,".format(channel)

        super().__init__(name=name,
                         unit='V',
                         get_cmd=self.get_voltage,
                         set_cmd=self.send_cmd + '"VOLT {:.4f}"',
                         step=step,
                         inter_delay=inter_delay,
                         vals=vals.Numbers(-max_voltage, max_voltage),
                         **kwargs)
        self.channel = channel

        self._meta_attrs.extend(['reset', 'charge_cycles'])

    @property
    def charge_cycles(self):
        if (self.t_last_cycle_check is not None
            and time() - self.t_last_cycle_check < self.t_recheck_cycles):

            self._instrument.write(self.send_cmd + '"BIDN? CYCLES"')
            sleep(0.08)
            return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))

            try:
                self._latest_charge_cycles = int(return_str.rstrip()[5:])
            except:
                logger.warning('Return string not understood: ' + return_str)
                self._latest_charge_cycles = -1
        return self._latest_charge_cycles

    def get_voltage(self):
        """
        Retrieves the DAC voltage.
        Note that there is a small delay, since two commands must be sent.

        Returns:
            Channel voltage
        """
        # Two commands must be sent to the instrument to retrieve the channel voltage
        self._instrument.write(self.send_cmd + '"VOLT?"')
        # A small wait is needed before the actual voltage can be retrieved
        sleep(0.1)
        return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
        for k in range(5):
            if return_str == '#3000\n':
                logger.warning('Received return string {}, '
                               'resetting SIM {}'.format(return_str, self.name))
                self._instrument.reset_slot(self.channel)
                sleep(1)
                self._instrument.write(self.send_cmd + '"VOLT?"')
                sleep(1)
                return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
            else:
                break
        return float(return_str[5:-3])


class SIM900(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research SIM900.
    It is currently only programmed for DAC voltage sources.

    Args:
        name (str): name of the instrument.
        address (str): The GPIB address of the instrument.
    """

    # Dictionary containing current module classes
    modules = {'SIM928': SIM928}

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        # The SIM900 has eight channels
        self.number_of_channels = 8

        # Dictionary with (channel, module) elements
        self._modules = {}

        # Start with empty list of channels. These are
        self.add_parameter('channels',
                           initial_value={},
                           set_cmd=None,
                           vals=vals.Anything(),
                           snapshot_value=False)

    def define_slot(self, channel, name=None, module='SIM928', **kwargs):
        """
        Define a module for a SIM900 slot.
        Args:
            channel (int): The SIM900 slot channel for the module
            name (Optional[str]): Module name (default 'channel_{channel}')
            module (Optional[str]): Module type (default 'SIM928)
            **kwargs: Module-specific kwargs, and StandardParameter kwargs

        Returns:
            None
        """
        assert isinstance(channel, int), "Channel {} must be an integer".format(channel)
        assert channel not in self.channels().keys(), "Channel {} already exists".format(channel)
        assert module in self.modules.keys(), "Module {} is not programmed".format(module)

        self.add_parameter(name=name,
                           channel=channel,
                           parameter_class=self.modules[module],
                           **kwargs)

        # Add
        channels = self.channels()
        channels[channel] = name
        self.channels(channels)


    def reset_slot(self, channel):
        self.write(cmdbase + 'SRST {}'.format(channel))


voltage_parameters = []

def get_voltages(copy=True):
    """ Get scaled parameter voltages as dict """
    voltage_dict = {param.name: param() for param in voltage_parameters}
    if copy:
        voltage_json = json.dumps(voltage_dict)
        pyperclip.copy(voltage_json)
    return voltage_dict


def ramp_voltages(target_voltage=None, gate_names=None, **kwargs):
    """
    Ramp multiple gates in multiple steps.

    Note that voltage_parameters must contain the parameters to be varied

    Usage:
        ramp_voltages(target_voltage)
            Ramp voltages of all gates to target_voltage
        ramp_voltages(target_voltage, channels)
            Ramp voltages of gates with names in channels to target_voltage
        ramp_voltages(gate1=val1, gate2=val2, ...)
            Ramp voltage of gate1 to val1, gate2 to val2, etc.

    Args:
        target_voltage (int): target voltage (can be omitted)
        gate_names (str list): Names of gates to be ramped (can be omitted)
        use_scaled: Use scaled SIM parameter (SIM900_scaled_parameters)
        **kwargs:

    Returns:
        None
    """
    parameters = {param.name: param for param in voltage_parameters}

    if target_voltage is not None:
        if isinstance(target_voltage, dict):
            # Accidentally passed kwargs dict without splat
            kwargs = target_voltage
        else:
            if gate_names is None:
                gate_names = parameters.keys()
            target_voltages = {gate_name: target_voltage
                               for gate_name in gate_names}
    elif kwargs:
        gate_names = kwargs.keys()
        target_voltages = {gate_name: val for gate_name, val in kwargs.items()}

    initial_voltages = {gate_name: parameters[gate_name]()
                        for gate_name in gate_names}

    for ratio in np.linspace(0, 1, 11):
        for gate_name in gate_names:
            voltage = (1 - ratio) * initial_voltages[gate_name] + \
                      ratio * target_voltages[gate_name]
            parameters[gate_name](voltage)