from .lakeshore_base import LakeshoreBase, BaseSensorChannel

class Model_372_Channel(BaseSensorChannel):
    pass


class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver
    Controlled via sockets
    """
    CHANNEL_CLASS = Model_372_Channel
    channel_name_command: Dict[str,str] = {'ch{:02}'.format(i): str(i) for i in range(1,17)}
  