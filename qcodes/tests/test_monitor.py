"""
Test suite for monitor
"""
import json
import random

import pytest
import websockets

from qcodes.monitor import monitor
from qcodes.parameters import Parameter
from qcodes.tests.instrument_mocks import DummyChannelInstrument, DummyInstrument

monitor.WEBSOCKET_PORT = random.randint(50000, 60000)


@pytest.fixture(name="inst_and_monitor")
def _make_inst_and_monitor():
    instr = DummyInstrument("MonitorDummy")
    param = Parameter("DummyParam",
                      unit="V",
                      get_cmd=None,
                      set_cmd=None)
    param(1)
    monitor_parameters = tuple(instr.parameters.values())[1:]
    my_monitor = monitor.Monitor(*monitor_parameters, param, interval=0.1)
    try:
        yield instr, my_monitor, monitor_parameters, param
    finally:
        my_monitor.stop()
        instr.close()


@pytest.fixture(name="channel_instr")
def _make_channel_instr():
    instr = DummyChannelInstrument("MonitorDummy")
    try:
        yield instr
    finally:
        instr.close()


@pytest.fixture(name="channel_instr_monitor", params=[True, False])
def _make_channel_instr_monitor(channel_instr, request):

    m = monitor.Monitor(
        channel_instr.A.dummy_start,
        channel_instr.B.dummy_start,
        use_root_instrument=request.param,
    )
    try:
        yield m, request.param
    finally:
        m.stop()


def test_setup_teardown(request):
    """
    Check that monitor starts up and closes correctly
    """
    m = monitor.Monitor()
    request.addfinalizer(m.stop)
    assert m.is_alive()
    assert m.loop.is_running()
    assert monitor.Monitor.running == m
    m.stop()
    assert not m.loop.is_running()
    assert m.loop.is_closed()
    assert not m.is_alive()
    assert monitor.Monitor.running is None


def test_monitor_replace(request):
    """
    Check that monitors get correctly replaced
    """
    m = monitor.Monitor()
    request.addfinalizer(m.stop)
    assert monitor.Monitor.running == m
    m2 = monitor.Monitor()
    request.addfinalizer(m2.stop)
    assert monitor.Monitor.running == m2
    assert m.loop.is_closed()
    assert not m.is_alive()
    assert m2.is_alive()
    m2.stop()


def test_double_join(request):
    """
    Check that a double join doesn't cause a hang
    """
    m = monitor.Monitor()
    request.addfinalizer(m.stop)
    assert monitor.Monitor.running == m
    m.stop()
    m.stop()


@pytest.mark.usefixtures("inst_and_monitor")
@pytest.mark.asyncio
async def test_connection():
    """
    Test that we can connect to a monitor instance
    """
    async with websockets.connect(f"ws://localhost:{monitor.WEBSOCKET_PORT}"):
        pass


@pytest.mark.asyncio
async def test_instrument_update(inst_and_monitor):
    """
    Test instrument updates
    """
    instr, my_monitor, monitor_parameters, param = inst_and_monitor
    async with websockets.connect(
        f"ws://localhost:{monitor.WEBSOCKET_PORT}"
    ) as websocket:

        # Receive data from monitor
        data = await websocket.recv()
        data = json.loads(data)
        # Check fields
        assert "ts" in data
        assert "parameters" in data
        # Check one instrument and "No Instrument" is being sent
        assert len(data["parameters"]) == 2
        assert data["parameters"][1]["instrument"] == "Unbound Parameter"
        metadata = data["parameters"][0]
        assert metadata["instrument"] == str(instr)

        # Check parameter values
        old_timestamps = {}
        for local_param, mon in zip(monitor_parameters, metadata["parameters"]):
            assert isinstance(local_param, Parameter)
            assert str(local_param.get_latest()) == mon["value"]
            assert local_param.label == mon["name"]
            old_timestamps[local_param.label] = float(mon["ts"])
            local_param(random.random())

        # Check parameter updates
        data_str = await websocket.recv()
        data_str = await websocket.recv()
        data = json.loads(data_str)
        metadata = data["parameters"][0]
        for local_param, mon in zip(monitor_parameters, metadata["parameters"]):
            assert str(local_param.get_latest()) == mon["value"]
            assert isinstance(local_param, Parameter)
            assert local_param.label == mon["name"]
            assert float(mon["ts"]) > old_timestamps[local_param.label]

        # Check unbound parameter
        metadata = data["parameters"][1]["parameters"]
        assert len(metadata) == 1
        assert param.label == metadata[0]["name"]


@pytest.mark.asyncio
async def test_monitor_root_instr(channel_instr_monitor):
    _, use_root_instrument = channel_instr_monitor
    async with websockets.connect(
        f"ws://localhost:{monitor.WEBSOCKET_PORT}"
    ) as websocket:

        # Receive data from monitor
        data = await websocket.recv()
        data = json.loads(data)
        if use_root_instrument:
            assert len(data["parameters"]) == 1
        else:
            assert len(data["parameters"]) == 2
