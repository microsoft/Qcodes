
from functools import partial
from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers
from time import sleep


def cmdbase(i): return "TERM LF\nFLSH\nFLOQ\nSNDT {:d} ,".format(i)


class SIM900(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research SIM900.
    It is currently only programmed for DAC voltage sources.

    Args:
        name (str): name of the instrument.
        address (str): The GPIB address of the instrument.
        channels (int): Number of DAC channels residing in the instument.
        channel_label (str, Optional): Prefix for the channels.
            This is useful when multiple SIM900 instruments are used.
    """
    def __init__(self, name, address, channels=8, channel_label='', **kwargs):
        super().__init__(name, address, **kwargs)

        self._channels = channels
        self.add_parameter('channels',
                           get_cmd=lambda: self._channels)
        for i in range(1, channels + 1):
            self.add_parameter('chan{}{}'.format(channel_label, i),
                               label='Gate Channel {} (V)'.format(i),
                               get_cmd=partial(self.get_voltage, channel=i),
                               set_cmd=cmdbase(i) + '"VOLT {:.4f}"',
                               step=0.003,
                               delay=0.05,
                               vals=Numbers(0, 20))

    def get_voltage(self, channel):
        """
        Retrieves the DAC voltage from a channel.
        Note that there is a small delay, since two commands must be sent.
        Args:
            channel (int): DAC channel from which to retrieve the voltage

        Returns:
            Channel voltage
        """
        # Two commands must be sent to the instrument to retrieve the channel voltage
        self.write(cmdbase(channel) + '"VOLT?"')
        # A small wait is needed before the actual voltage can be retrieved
        sleep(0.05)
        return_str = self.ask('GETN?{:d},100'.format(channel))
        return float(return_str[5:-3])
