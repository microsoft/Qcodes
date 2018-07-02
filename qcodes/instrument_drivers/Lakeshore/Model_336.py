from typing import ClassVar, Dict
import logging
from .lakeshore_base import LakeshoreBase, BaseOutput, BaseSensorChannel, VAL_MAP_TYPE
import qcodes.utils.validators as vals

log = logging.getLogger(__name__)

class Output_336(BaseOutput):

    MODES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'closed_loop': 1,
        'zone': 2,
        'open_loop': 3,
        'monitor_out': 4,
        'warm_up': 5}
    RANGES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'low': 1,
        'medium': 2,
        'heigh': 3}

    def __init__(self, parent, output_name, output_index):
        if output_name not in ['A','B']:
            self._has_pid = False
        super().__init__(parent, output_name, output_index)
        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)

        self.range_limits.vals = validators.Sequence(
            validators.Numbers(0,400),length=2, require_sorted=True)


class Model_336_Channel(BaseSensorChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name, channel)


class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    """
    channel_name_command: Dict[str,str] = {'A': 'A',
                                           'B': 'B',
                                           'C': 'C',
                                           'D': 'D'}
    CHANNEL_CLASS = Model_336_Channel
