import pytest
from qcodes.instrument_drivers.test import DriverTestCase

@pytest.fixture
def uut():
    from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X
    import qcodes.instrument.sims as sims
    path_to_yaml = sims.__file__.replace('__init__.py', 'keysight_b220x.yaml')

    uut = KeysightB220X('switch_matrix',
                        address='GPIB::1::INSTR',
                        visalib=path_to_yaml+'@sim'
                        )

    yield uut

    uut.close()


class TestSimulatedKeysightB220X:
    def test_idn_command(self, uut):
        assert "AGILENT" in uut.IDN()['vendor']
        assert 0 == int(uut.ask('*ESR?'))

    def test_connection_rule(self, uut):
        uut.connection_rule('single')
        assert 0 == int(uut.ask('*ESR?'))
        assert 'single' == uut.connection_rule()
        assert 0 == int(uut.ask('*ESR?'))


    def test_bias_disable_all(self, uut):
        uut.bias_disable_all()
        assert 0 == int(uut.ask('*ESR?'))

    def test_bias_disable_channel(self, uut):
        uut.bias_disable_channel(1)
        assert 0 == int(uut.ask('*ESR?'))

    def test_bias_enable_all(self, uut):
        uut.bias_enable_all()
        assert 0 == int(uut.ask('*ESR?'))

    def test_bias_enable_channel(self, uut):
        uut.bias_enable_channel(1)
        assert 0 == int(uut.ask('*ESR?'))
