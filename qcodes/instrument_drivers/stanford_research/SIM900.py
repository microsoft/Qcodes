
from functools import partial
from qcodes import VisaInstrument
from qcodes.instrument.parameter import StandardParameter, ManualParameter
from qcodes.utils import validators as vals
from time import sleep


cmdbase = "TERM LF\nFLSH\nFLOQ\n"

class SIM928(StandardParameter):
    """
    This is the parameter class for the SIM928 rechargeable isolated voltage source module

    Args:
        channel (int): SIM900 channel for the SIM928 module

        name (Optional[str]): Module name (default 'channel_{channel}')

        max_voltage (Optional[float]): Maximum voltage (default 20)
    """
    def __init__(self, channel, name=None, max_voltage=20, **kwargs):
        if not name:
            name = 'channel_{}'.format(channel)

        self.send_cmd = cmdbase + "SNDT {:d} ,".format(channel)

        super().__init__(name=name,
                         units='V',
                         get_cmd=self.get_voltage,
                         set_cmd=self.send_cmd + '"VOLT {:.4f}"',
                         step=0.005,
                         delay=0.025,
                         vals=vals.Numbers(0, max_voltage),
                         **kwargs)
        self.channel = channel

        self._meta_attrs.extend(['reset'])

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
        sleep(0.05)
        return_str = self._instrument.ask('GETN?{:d},100'.format(self.channel))
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
                           parameter_class=ManualParameter,
                           initial_value={},
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

