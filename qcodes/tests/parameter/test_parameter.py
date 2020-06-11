"""
Test suite for parameter
"""
from unittest import TestCase

from qcodes.instrument.parameter import Parameter
from qcodes.tests.instrument_mocks import DummyInstrument


class TestStandardParam(TestCase):
    def set_p(self, val):
        self._p = val

    def set_p_prefixed(self, val):
        self._p = 'PVAL: {:d}'.format(val)

    def strip_prefix(self, val):
        return int(val[6:])

    def get_p(self):
        return self._p

    def parse_set_p(self, val):
        return '{:d}'.format(val)

    def test_param_cmd_with_parsing(self):
        p = Parameter('p_int', get_cmd=self.get_p, get_parser=int,
                      set_cmd=self.set_p, set_parser=self.parse_set_p)

        p(5)
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)
        self.assertEqual(p.get_latest(), 5)

    def test_settable(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=False)

        p(10)
        self.assertEqual(self._p, 10)
        with self.assertRaises(NotImplementedError):
            p()

        self.assertTrue(hasattr(p, 'set'))
        self.assertTrue(p.settable)
        self.assertFalse(hasattr(p, 'get'))
        self.assertFalse(p.gettable)

        # For settable-only parameters, using ``cache.set`` may not make
        # sense, nevertheless, it works
        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)

    def test_gettable(self):
        p = Parameter('p', get_cmd=self.get_p)
        self._p = 21

        self.assertEqual(p(), 21)
        self.assertEqual(p.get(), 21)

        with self.assertRaises(NotImplementedError):
            p(10)

        self.assertTrue(hasattr(p, 'get'))
        self.assertTrue(p.gettable)
        self.assertFalse(hasattr(p, 'set'))
        self.assertFalse(p.settable)

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 21)
        self.assertEqual(p(), 21)
        self.assertEqual(p.get_latest(), 21)


class TestManualParameterValMapping(TestCase):
    def setUp(self):
        self.instrument = DummyInstrument('dummy_holder')

    def tearDown(self):
        self.instrument.close()
        del self.instrument

    def test_val_mapping(self):
        self.instrument.add_parameter('myparameter', set_cmd=None, get_cmd=None, val_mapping={'A': 0, 'B': 1})
        self.instrument.myparameter('A')
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter.raw_value == 0



