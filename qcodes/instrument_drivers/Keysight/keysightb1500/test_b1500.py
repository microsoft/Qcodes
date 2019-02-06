import pytest
from unittest.mock import MagicMock

from pyvisa.errors import VisaIOError

from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import parse_module_query_response, B1500Module, \
    B1517A, B1520A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import ChNr, SlotNr


@pytest.fixture
def b1500():
    try:
        resource_name = 'insert_Keysight_B2200_VISA_resource_name_here'
        instance = KeysightB1500('SPA',
                                 address=resource_name)
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real
        # instrument found at the specified address => use simulated instrument
        import qcodes.instrument_drivers.Keysight.keysightb1500 as module
        path_to_yaml = module.__file__.replace('__init__.py',
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


def test_parse_module_query_response():
    from qcodes.instrument_drivers.Keysight.keysightb1500.constants import SlotNr
    response = 'B1517A,0;B1517A,0;B1520A,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0'
    expected = {SlotNr.SLOT01: 'B1517A',
                SlotNr.SLOT02: 'B1517A',
                SlotNr.SLOT03: 'B1520A'}

    actual = parse_module_query_response(response)

    assert actual == expected


class TestB1500Module:
    def test_make_module(self):
        mainframe = MagicMock()

        smu = B1500Module.from_model_name(model='B1517A', slot_nr=1, parent=mainframe, name='dummy')

        assert type(smu) == B1517A

        cmu = B1500Module.from_model_name(model='B1520A', slot_nr=2, parent=mainframe)

        assert type(cmu) == B1520A
