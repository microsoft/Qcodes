from unittest import TestCase

from qcodes.loops import Loop
from qcodes.data.io import DiskIO
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers
from .instrument_mocks import AMockModel, MockGates, MockSource, MockMeter


class TestMockInstLoop(TestCase):
    def setUp(self):
        self.model = AMockModel()

        self.gates = MockGates(self.model)
        self.source = MockSource(self.model)
        self.meter = MockMeter(self.model)
        self.location = '_loop_test_'
        self.io = DiskIO('.')

    def tearDown(self):
        self.io.remove_all(self.location)

    def test_instruments_in_loop(self):
        # make sure that an unpicklable instrument can indeed run in a loop
        self.assertFalse(self.io.list(self.location))
        c1 = self.gates.chan1
        loop = Loop(c1[1:5:1], 0.001).each(c1)

        # TODO: if we don't save the dataset (location=False) then we can't
        # sync it when we're done. Should fix that - for now that just means
        # you can only do foreground in-memory loops
        data = loop.run(location=self.location, quiet=True)

        # wait for process to finish (ensures that this was run in the bg,
        # because otherwise there *is* no loop.process)
        loop.process.join()

        data.sync()

        # TODO: data.chan1_ is a weird name. Weird use case (set and measure
        # the same parameter), but lets get a better name
        self.assertEqual(list(data.chan1_0), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(list(data.chan1_), [1.0, 2.0, 3.0, 4.0])

        self.assertTrue(self.io.list(self.location))
