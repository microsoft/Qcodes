import pytest
from qcodes.instrument_drivers.Keysight.Keysight_34980A import Keysight_34980A
import random


@pytest.fixture(scope="module")
def SwitchMatrix():
    SwitchMatrix = Keysight_34980A("inst", "TCPIP0::10.193.36.65::inst0::INSTR")
    return SwitchMatrix


def test_connections(SwitchMatrix):
    connections = [(random.randint(1, 8), random.randint(1, 128)) for i in range(8)]
    SwitchMatrix.connect_paths(connections)
    for r, c in connections:
        if SwitchMatrix.is_open(r, c):
            raise SystemError(f'r{r} and c{c} are not connected')
        assert SwitchMatrix.is_closed(r, c)


def test_error_handeling(SwitchMatrix):
    pass
