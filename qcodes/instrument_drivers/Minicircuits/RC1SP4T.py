from qcodes import IPInstrument
import math
from qcodes.utils import validators as vals


class RC1SP4T(IPInstrument):
    """
    Mini-Circuits 1x 1P4T RF switch

    Args:
            name (str): the name of the channel
            address (str): ip address ie "10.0.0.1"
            port (int): port to connect to default Telnet:23
    """
    def __init__(self, name, address, port=23):
        super().__init__(name, address, port)
        self._recv()

        self.add_parameter('switch', 
                           label='switch',
                           set_cmd=self._set_switch,
                           get_cmd=self._get_switch,
                           vals=vals.Ints(0, 4)
                           )

        self.connect_message()
    
    def ask(self, cmd):
        ret = self.ask_raw(cmd)
        ret = ret.strip()
        return ret

    def _set_switch(self, switch):
        if switch == 0:
            val = 0
        else:
            val = 1 << (switch-1)
        self.write('SETP={}'.format(val))
    
    def _get_switch(self):
        val = int(self.ask('SWPORT?'))
        if val == 0:
            return 0
        ret = int(math.log2(val) + 1)
        return ret

    def get_idn(self):

        fw = self.ask('FIRMWARE?')
        MN = self.ask('MN?')
        SN = self.ask('SN?')

        id_dict = {'firmware': fw,
                   'model': MN[3:],
                   'serial': SN[3:],
                   'vendor': 'Mini-Circuits'}
        return id_dict
