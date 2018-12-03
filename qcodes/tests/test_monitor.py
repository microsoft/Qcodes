"""
Test suite for monitor
"""
from unittest import TestCase

import time
import asyncio
import websockets
import json
import random

import qcodes as qc
from qcodes.instrument.base import Parameter
from qcodes.tests.instrument_mocks import DummyInstrument

class TestMonitor(TestCase):
    """
    Test cases for the qcodes monitor
    """
    def test_setup_teardown(self):
        """
        Check that monitor starts up and closes correctly
        """
        m = qc.Monitor()
        self.assertTrue(m.is_alive())
        # Wait for loop to start
        while m.loop is None:
            time.sleep(0.01)
        self.assertTrue(m.loop.is_running())
        self.assertEqual(qc.Monitor.running, m)
        m.stop()
        self.assertFalse(m.loop.is_running())
        self.assertTrue(m.loop.is_closed())
        self.assertFalse(m.is_alive())
        self.assertIsNone(qc.Monitor.running)

    def test_connection(self):
        """
        Test that we can connect to a monitor instance
        """
        m = qc.Monitor()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        @asyncio.coroutine
        def async_test_connection():
            websocket = yield from websockets.connect(f"ws://localhost:{qc.monitor.monitor.WEBSOCKET_PORT}")
            yield from websocket.close()
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
        self.monitor = qc.Monitor(*self.monitor_parameters, self.param)

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

        @asyncio.coroutine
        def async_test_monitor():
            websocket = yield from websockets.connect(f"ws://localhost:{qc.monitor.monitor.WEBSOCKET_PORT}")

            # Recieve data from monitor
            data = yield from websocket.recv()
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
            for local, monitor in zip(self.monitor_parameters, metadata["parameters"]):
                self.assertEqual(str(local.get_latest()), monitor["value"])
                self.assertEqual(local.label, monitor["name"])
                old_timestamps[local.label] = float(monitor["ts"])
                local(random.random())

            # Check parameter updates
            data = yield from websocket.recv()
            data = yield from websocket.recv()
            data = json.loads(data)
            metadata = data["parameters"][0]
            for local, monitor in zip(self.monitor_parameters, metadata["parameters"]):
                self.assertEqual(str(local.get_latest()), monitor["value"])
                self.assertEqual(local.label, monitor["name"])
                self.assertGreater(float(monitor["ts"]), old_timestamps[local.label])

            # Check unbound parameter
            metadata = data["parameters"][1]["parameters"]
            self.assertEqual(len(metadata), 1)
            self.assertEqual(self.param.label, metadata[0]["name"])

        loop.run_until_complete(async_test_monitor())
