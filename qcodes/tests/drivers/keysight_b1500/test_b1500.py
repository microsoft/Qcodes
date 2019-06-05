from unittest.mock import MagicMock

import pytest
from pyvisa.errors import VisaIOError

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    KeysightB1500, B1530A
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    parse_module_query_response, B1500Module, B1517A, B1520A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import ChNr, \
    SlotNr, VOutputRange, CompliancePolarityMode, IOutputRange


@pytest.fixture
def b1500():
    try:
        resource_name = 'insert_Keysight_B2200_VISA_resource_name_here'
        instance = KeysightB1500('SPA',
                                 address=resource_name)
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real
        # instrument found at the specified address => use simulated instrument
        import qcodes.instrument.sims as sims
        path_to_yaml = sims.__file__.replace('__init__.py',
                                             'keysight_b1500.yaml')

        instance = KeysightB1500('SPA',
                                 address='GPIB::1::INSTR',
                                 visalib=path_to_yaml + '@sim'
                                 )

    instance.get_status()
    instance.reset()

    yield instance

    instance.close()


class TestB1500:
    def test_init(self, b1500):
        assert hasattr(b1500, 'smu1')
        assert hasattr(b1500, 'smu2')
        assert hasattr(b1500, 'cmu1')

    def test_submodule_access_by_class(self, b1500):
        assert b1500.smu1 in b1500.by_class['SMU']
        assert b1500.smu2 in b1500.by_class['SMU']
        assert b1500.cmu1 in b1500.by_class['CMU']

    def test_submodule_access_by_slot(self, b1500):
        assert b1500.smu1 is b1500.by_slot[SlotNr.SLOT01]
        assert b1500.smu2 is b1500.by_slot[SlotNr.SLOT02]
        assert b1500.cmu1 is b1500.by_slot[3]

    def test_submodule_access_by_channel(self, b1500):
        assert b1500.smu1 is b1500.by_channel[ChNr.SLOT_01_CH1]
        assert b1500.smu2 is b1500.by_channel[ChNr.SLOT_02_CH1]
        assert b1500.cmu1 is b1500.by_channel[ChNr.SLOT_03_CH1]
        assert b1500.aux1 is b1500.by_channel[ChNr.SLOT_06_CH1]
        assert b1500.aux1 is b1500.by_channel[ChNr.SLOT_06_CH2]

    def test_enable_multiple_channels(self, b1500):
        mock_write = MagicMock()
        b1500.write = mock_write

        b1500.enable_channels({1, 2, 3})

        mock_write.assert_called_once_with("CN 1,2,3")

    def test_disable_multiple_channels(self, b1500):
        mock_write = MagicMock()
        b1500.write = mock_write

        b1500.disable_channels({1, 2, 3})

        mock_write.assert_called_once_with("CL 1,2,3")


def test_parse_module_query_response():
    response = 'B1517A,0;B1517A,0;B1520A,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0'
    expected = {SlotNr.SLOT01: 'B1517A',
                SlotNr.SLOT02: 'B1517A',
                SlotNr.SLOT03: 'B1520A'}

    actual = parse_module_query_response(response)

    assert actual == expected


class TestB1500Module:
    def test_make_module(self):
        mainframe = MagicMock()

        with pytest.raises(NotImplementedError):
            B1500Module.from_model_name(model='unsupported_module', slot_nr=0,
                                        parent=mainframe, name='dummy')

        smu = B1500Module.from_model_name(model='B1517A', slot_nr=1,
                                          parent=mainframe, name='dummy')

        assert isinstance(smu, B1517A)

        cmu = B1500Module.from_model_name(model='B1520A', slot_nr=2,
                                          parent=mainframe)

        assert isinstance(cmu, B1520A)

        aux = B1500Module.from_model_name(model='B1530A', slot_nr=3,
                                          parent=mainframe)

        assert isinstance(aux, B1530A)

    def test_is_enabled(self):
        mainframe = MagicMock()

        smu = B1517A(parent=mainframe, name='B1517A',
                     slot_nr=1)  # Uses concrete
        # subclass because B1500Module does not assign channels

        mainframe.ask.return_value = 'CN 1,2,4,8'
        assert smu.is_enabled()
        mainframe.ask.assert_called_once_with('*LRN? 0')

        mainframe.reset_mock(return_value=True)
        mainframe.ask.return_value = 'CN 2,4,8'
        assert not smu.is_enabled()
        mainframe.ask.assert_called_once_with('*LRN? 0')

    def test_enable_output(self):
        mainframe = MagicMock()
        slot_nr = 1
        smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)  # Uses
        # concrete subclass because B1500Module does not assign channels

        smu.enable_outputs()
        mainframe.write.assert_called_once_with(f'CN {slot_nr}')

    def test_disable_output(self):
        mainframe = MagicMock()
        slot_nr = 1
        smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)  # Uses
        # concrete subclass because B1500Module does not assign channels

        smu.disable_outputs()
        mainframe.write.assert_called_once_with(f'CL {slot_nr}')


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

# class TestMFCMUResultParameter:
#     def test_init(self):
#         m = MFCMUResult()
#
