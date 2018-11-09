import sys
import unittest
from unittest.mock import patch, MagicMock, call

sys.modules['spirack'] = MagicMock(name='spirack')
from spirack import M2j_module, SPI_rack
from qcodes.instrument_drivers.QuTech.M2j import M2j


class TestZIHDAWG8(unittest.TestCase):

    def test_gain(self):
        spi_rack = MagicMock()
        m2j = M2j('test', spi_rack, 42)

        with self.assertRaises(ValueError):
            m2j.gain.set(32)
        with self.assertRaises(ValueError):
            m2j.gain.set(56)

        for gain in range(33, 56):
            m2j.gain.set(gain)

        ref_scales = [4095, 3740, 3325, 3030, 2801, 2615, 2457, 2320, 2199,
                      2091, 1994, 1904, 1822, 1747, 1676, 1610, 1548, 1489,
                      1434, 1381, 1331, 1284, 1238]
        calls = [call.set_gain(ref_scale) for ref_scale in ref_scales]
        m2j.m2j.assert_has_calls(calls)

    def test_rf_level(self):
        spi_rack = MagicMock()
        m2j = M2j('test_rf', spi_rack, 42)

        m2j.RF_level.get()
        m2j.m2j.assert_has_calls([call.get_level()])