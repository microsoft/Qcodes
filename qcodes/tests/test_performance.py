from unittest import TestCase
import time

from qcodes.tests.instrument_mocks import DummyInstrument


class TestInstrumentGet(TestCase):

    def test_get_overhead(self, verbose=0):
        # test overhead of local and remote parameter
        local = DummyInstrument(name='local', server_name=None)
        remote = DummyInstrument(name='remote')

        gettime = 1e-3
        t0 = time.time()
        for ii in range(1000):
            time.sleep(gettime)
        dt = time.time() - t0

        t0 = time.time()
        for ii in range(1000):
            local.dac1.get()
            time.sleep(gettime)
        dtlocal = time.time() - t0

        t0 = time.time()
        for ii in range(1000):
            remote.dac1.get()
            time.sleep(gettime)
        dtremote = time.time() - t0
        overheadlocal = dtlocal / dt
        overheadremote = dtremote / dt

        if verbose:
            print('test_get_overhead: local %f, remote %f: overhead %.2f %.2f' %
                  (dtlocal, dtremote, overheadlocal, overheadremote))

        self.assertLess(overheadlocal, 1.2)
                        # at most 20% overhead for local measurement
        self.assertLess(overheadremote, 1.4)
                        # at most 40% overhead for remote measurement
