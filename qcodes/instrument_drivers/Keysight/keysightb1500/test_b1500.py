import pytest
from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500
from pyvisa.errors import VisaIOError


@pytest.fixture
def uut():
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
    def test_all(self, uut):
        uut.mb.message
