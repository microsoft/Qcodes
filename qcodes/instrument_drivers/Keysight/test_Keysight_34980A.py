import pytest
from qcodes.instrument_drivers.Keysight.Keysight_34980A import Keysight_34980A
import random


@pytest.fixture(scope="module")
def switch_matrix():
    switch_matrix = Keysight_34980A("inst", "TCPIP0::10.193.36.65::inst0::INSTR")
    return switch_matrix


def test_connections(switch_matrix):
    connections = [(random.randint(1, 8), random.randint(1, 128)) for i in range(8)]
    switch_matrix.connect_paths(connections)
    for r, c in connections:
        if switch_matrix.is_open(r, c):
            raise SystemError(f'r{r} and c{c} are not connected')
        assert switch_matrix.is_closed(r, c)


def test_connections_all(switch_matrix):
    for row in range(1, 9):
        for column in range(1, 129):
            switch_matrix.connect_path(row, column)
            if switch_matrix.is_open(row, column):
                raise SystemError(f'r{row} and c{column} are not connected')
            assert switch_matrix.is_closed(row, column)
            switch_matrix.disconnect_path(row, column)
            if switch_matrix.is_closed(row, column):
                raise SystemError(f'r{row} and c{column} are still connected')
            assert switch_matrix.is_open(row, column)


def test_error_handeling(switch_matrix):
    pass
