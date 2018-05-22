from typing import Dict
from .lakeshore_base import LakeshoreBase
import qcodes.utils.validators as vals


class Output_372(BaseOutput):
    MODES = {'off': 0,
             'closed_loop': 1,
             'zone': 2,
             'open_loop': 3,
             'monitor_out': 4,
             'warm_up': 5}
    RANGES = {'off': 0,
              'low': 1,
              'medium': 2,
              'heigh': 3}

    def __init__(self, parent, output_name, output_index):
        super().__init__(parent, output_name, output_index)
        # TODO: PID only valid for channels 1 and 2
        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)


class Model_336_Channel(BaseSensorChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name, channel)
        self.add_parameter('setpoint',
                           get_cmd='setp? {}'.format(self._channel),
                           set_cmd='setp {},{{}}'.format(self._channel),
                           get_parser=float,
                           label = 'Temperature setpoint',
                           unit='K')

        self.add_parameter('range_id',
                           get_cmd='range? {}'.format(self._channel),
                           set_cmd='range {},{{}}'.format(self._channel),
                           get_parser=validators.Enum(1,2,3),
                           label = 'Range ID',
                           unit='K')

    def set_range_from_temperature(self, temperature):
        if temperature < self._parent.t_limit(1):
            range_id = 1
        elif temperature < self._parent.t_limit(2):
            range_id = 2
        else:
            range_id = 3
        self.range_id(range_id)

    def set_setpoint_and_range(self, temperature):
        self.set_range_from_temperature(temperature)
        self.setpoint(temperature)

class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    Controlled via sockets
    """
    channel_name_command: Dict[str,str] = {'A': 'A',
                                           'B': 'B',
                                           'C': 'C',
                                           'D': 'D'}
    CHANNEL_CLASS = Model_336_Channel
