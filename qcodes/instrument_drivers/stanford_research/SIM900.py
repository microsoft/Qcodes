
from functools import partial
from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers
from time import sleep


def cmdbase(i): return "TERM LF\nFLSH\nFLOQ\nSNDT {:d} ,".format(i)


class SIM900(VisaInstrument):
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
                               step=0.01,
                               delay=0.2,
                               vals=Numbers(0, 20))

    def get_voltage(self, channel):
        self.write(cmdbase(channel) + '"VOLT?"')
        sleep(0.1)
        return_str = self.ask('GETN?{:d},100'.format(channel))
        return float(return_str[5:-3])
