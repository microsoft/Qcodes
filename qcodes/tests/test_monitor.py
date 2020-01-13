"""
Test suite for monitor
"""
from unittest import TestCase

import asyncio
import json
import random
import websockets

from qcodes.monitor import monitor
from qcodes.instrument.base import Parameter
from qcodes.tests.instrument_mocks import DummyInstrument

monitor.WEBSOCKET_PORT = random.randint(50000, 60000)

class TestMonitor(TestCase):
    """
    Test cases for the qcodes monitor
    """
    def test_setup_teardown(self):
        """
        Check that monitor starts up and closes correctly
        """
        m = monitor.Monitor()
        self.assertTrue(m.is_alive())
        self.assertTrue(m.loop.is_running())
        self.assertEqual(monitor.Monitor.running, m)
        m.stop()
        self.assertFalse(m.loop.is_running())
        self.assertTrue(m.loop.is_closed())
        self.assertFalse(m.is_alive())
        self.assertIsNone(monitor.Monitor.running)

    def test_monitor_replace(self):
        """
        Check that monitors get correctly replaced
        """
        m = monitor.Monitor()
        self.assertEqual(monitor.Monitor.running, m)
        m2 = monitor.Monitor()
        self.assertEqual(monitor.Monitor.running, m2)
        self.assertTrue(m.loop.is_closed())
        self.assertFalse(m.is_alive())
        self.assertTrue(m2.is_alive())
        m2.stop()

    def test_double_join(self):
        """
        Check that a double join doesn't cause a hang
        """
        m = monitor.Monitor()
        self.assertEqual(monitor.Monitor.running, m)
        m.stop()
        m.stop()


    def test_connection(self):
        """
        Test that we can connect to a monitor instance
        """
        m = monitor.Monitor()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def async_test_connection():
            websocket = await websockets.connect(f"ws://localhost:{monitor.WEBSOCKET_PORT}")
            await websocket.close()
        loop.run_until_complete(async_test_connection())

        m.stop()

class TestMonitorWithInstr(TestCase):
    """
    Test monitor values from instruments
    """
    def setUp(self):
        """
        Create a dummy instrument for use in monitor tests, and hook it into
        a monitor
        """
        self.instr = DummyInstrument("MonitorDummy")
        self.param = Parameter("DummyParam",
                               unit="V",
                               get_cmd=None,
                               set_cmd=None)
        self.param(1)
        self.monitor_parameters = tuple(self.instr.parameters.values())[1:]
        self.monitor = monitor.Monitor(*self.monitor_parameters, self.param, interval=0.1)

    def tearDown(self):
        """
        Close the dummy instrument
        """
        self.monitor.stop()
        self.instr.close()

    def test_parameter(self):
        """
        Test instrument updates
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def async_test_monitor():
            websocket = await websockets.connect(f"ws://localhost:{monitor.WEBSOCKET_PORT}")

            # Recieve data from monitor
            data = await websocket.recv()
            data = json.loads(data)
            # Check fields
            self.assertIn("ts", data)
            self.assertIn("parameters", data)
            # Check one instrument and "No Instrument" is being sent
            self.assertEqual(len(data["parameters"]), 2)
            self.assertEqual(data["parameters"][1]["instrument"], "Unbound Parameter")
            metadata = data["parameters"][0]
            self.assertEqual(metadata["instrument"], str(self.instr))

            # Check parameter values
            old_timestamps = {}
            for local, mon in zip(self.monitor_parameters, metadata["parameters"]):
                self.assertEqual(str(local.get_latest()), mon["value"])
                self.assertEqual(local.label, mon["name"])
                old_timestamps[local.label] = float(mon["ts"])
                local(random.random())

            # Check parameter updates
            data = await websocket.recv()
            data = await websocket.recv()
            data = json.loads(data)
            metadata = data["parameters"][0]
            for local, mon in zip(self.monitor_parameters, metadata["parameters"]):
                self.assertEqual(str(local.get_latest()), mon["value"])
                self.assertEqual(local.label, mon["name"])
                self.assertGreater(float(mon["ts"]), old_timestamps[local.label])

            # Check unbound parameter
            metadata = data["parameters"][1]["parameters"]
            self.assertEqual(len(metadata), 1)
            self.assertEqual(self.param.label, metadata[0]["name"])

        loop.run_until_complete(async_test_monitor())
