from typing import Dict
from .lakeshore_base import LakeshoreBase


class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    Controlled via sockets
    """
    channel_name_command: Dict[str,str] = {'A': 'A',
                                           'B': 'B',
                                           'C': 'C',
                                           'D': 'D'}