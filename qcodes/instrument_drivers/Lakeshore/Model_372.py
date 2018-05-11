from .lakeshore_base import LakeshoreBase


class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver
    Controlled via sockets
    """
    channel_name_command: Dict[str,str] = {'ch{:02}'.format(i): str(i) for i in range(1,17)}
  