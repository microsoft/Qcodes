from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    B1500Module, B1517A, B1520A, B1530A


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