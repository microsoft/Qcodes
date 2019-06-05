from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    B1517A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    VOutputRange, CompliancePolarityMode, IOutputRange


class TestB1517A:
    def setup_method(self, method):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.mainframe = MagicMock()
        self.slot_nr = 1
        self.smu = B1517A(parent=self.mainframe, name='B1517A',
                          slot_nr=self.slot_nr)

    def test_force_voltage_with_autorange(self):
        self.smu.source_config(output_range=VOutputRange.AUTO)
        self.smu.voltage(10)

        self.mainframe.write.assert_called_once_with('DV 1,0,10')

    def test_force_voltage_autorange_and_compliance(self):
        self.smu.source_config(output_range=VOutputRange.AUTO,
                               compliance=1e-6,
                               compl_polarity=CompliancePolarityMode.AUTO,
                               min_compliance_range=IOutputRange.MIN_10uA)
        self.smu.voltage(20)

        self.mainframe.write.assert_called_once_with('DV 1,0,20,1e-06,0,15')

    def test_new_source_config_should_invalidate_old_source_config(self):
        self.smu.source_config(output_range=VOutputRange.AUTO,
                               compliance=1e-6,
                               compl_polarity=CompliancePolarityMode.AUTO,
                               min_compliance_range=IOutputRange.MIN_10uA)

        self.smu.source_config(output_range=VOutputRange.AUTO)
        self.smu.voltage(20)

        self.mainframe.write.assert_called_once_with('DV 1,0,20')

    def test_unconfigured_source_defaults_to_autorange_v(self):
        self.smu.voltage(10)

        self.mainframe.write.assert_called_once_with('DV 1,0,10')

    def test_unconfigured_source_defaults_to_autorange_i(self):
        self.smu.current(0.2)

        self.mainframe.write.assert_called_once_with('DI 1,0,0.2')

    def test_force_current_with_autorange(self):
        self.smu.source_config(output_range=IOutputRange.AUTO)
        self.smu.current(0.1)

        self.mainframe.write.assert_called_once_with('DI 1,0,0.1')

    def test_raise_warning_output_range_mismatches_output_command(self):
        self.smu.source_config(output_range=VOutputRange.AUTO)
        with pytest.raises(TypeError):
            self.smu.current(0.1)

        self.smu.source_config(output_range=IOutputRange.AUTO)
        with pytest.raises(TypeError):
            self.smu.voltage(0.1)

    def test_measure_current(self):
        self.mainframe.ask.return_value = "NAI+000.005E-06\r"
        assert pytest.approx(0.005e-6) == self.smu.current()

    def test_measure_voltage(self):
        self.mainframe.ask.return_value = "NAV+000.123E-06\r"
        assert pytest.approx(0.123e-6) == self.smu.voltage()