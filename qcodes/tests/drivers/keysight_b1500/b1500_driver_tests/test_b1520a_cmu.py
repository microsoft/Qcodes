from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    B1520A


class TestB1520A:
    def setup_method(self, method):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.mainframe = MagicMock()
        self.slot_nr = 3
        self.cmu = B1520A(parent=self.mainframe, name='B1520A',
                          slot_nr=self.slot_nr)

    def test_force_dc_voltage(self):
        self.cmu.voltage_dc(10)

        self.mainframe.write.assert_called_once_with('DCV 3,10')

    def test_force_ac_voltage(self):
        self.cmu.voltage_ac(0.1)

        self.mainframe.write.assert_called_once_with('ACV 3,0.1')

    def test_set_ac_frequency(self):
        self.cmu.frequency(100e3)

        self.mainframe.write.assert_called_once_with('FC 3,100000.0')

    def test_get_capacitance(self):
        self.mainframe.ask.return_value = "NCC-1.45713E-06,NCY-3.05845E-03"

        result = self.cmu.capacitance()

        assert pytest.approx((-1.45713E-06, -3.05845E-03)) == result

    def test_raise_error_on_unsupported_result_format(self):
        self.mainframe.ask.return_value = "NCR-1.1234E-03,NCX-4.5677E-03"

        with pytest.raises(ValueError):
            self.cmu.capacitance()