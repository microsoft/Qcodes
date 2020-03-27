import gc

from unittest import TestCase

from qcodes.loops import Loop
from qcodes.actions import UnsafeThreadingException
from qcodes.tests.instrument_mocks import DummyInstrument


class TestUnsafeThreading(TestCase):

    def setUp(self):
        self.inst1 = DummyInstrument(name='inst1',
                                     gates=['v1', 'v2'])
        self.inst2 = DummyInstrument(name='inst2',
                                     gates=['v1', 'v2'])

    def tearDown(self):
        self.inst1.close()
        self.inst2.close()

        del self.inst1
        del self.inst2

        gc.collect()

    def test_unsafe_exception(self):
        to_meas = (self.inst1.v1, self.inst1.v2)
        loop = Loop(self.inst2.v1.sweep(0, 1, num=10)).each(*to_meas)

        with self.assertRaises(UnsafeThreadingException):
            loop.run(use_threads=True)
