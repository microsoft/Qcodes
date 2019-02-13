import pytest

from qcodes.instrument_drivers.tektronix.Keithley_2600_channels import \
    Keithley_2600

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_2600.yaml@sim')


@pytest.fixture(scope='module')
def driver():
    driver = Keithley_2600('Keithley_2600',
                           address='GPIB::1::INSTR',
                           visalib=visalib)

    yield driver
    driver.close()


def test_idn(driver):
    assert {'firmware': '3.0.0',
            'model': '2601B',
            'serial': '1398687',
            'vendor': 'Keithley Instruments Inc.'} == driver.IDN()


def test_smu_channels_and_their_parameters(driver):
    assert {'smua', 'smub'} == set(list(driver.submodules.keys()))

    for smu_name in {'smua', 'smub'}:
        smu = getattr(driver, smu_name)

        smu.volt(1.0)
        assert 1.0 == smu.volt()

        smu.curr(1.0)
        assert 1.0 == smu.curr()

        assert 0.0 == smu.res()

        assert 'current' == smu.mode()
        smu.mode('voltage')

        assert 'off' == smu.output()
        smu.output('on')

        assert 0.0 == smu.nplc()
        smu.nplc(2.3)

        assert 0.0 == smu.sourcerange_v()
        some_valid_sourcerange_v = driver._vranges[smu.model][2]
        smu.sourcerange_v(some_valid_sourcerange_v)

        assert 0.0 == smu.sourcerange_i()
        some_valid_sourcerange_i = driver._iranges[smu.model][2]
        smu.sourcerange_i(some_valid_sourcerange_i)

        assert 0.0 == smu.measurerange_i()
        some_valid_measurerange_i = driver._iranges[smu.model][2]
        smu.measurerange_i(some_valid_measurerange_i)

        assert 0.0 == smu.limitv()
        smu.limitv(2.3)

        assert 0.0 == smu.limiti()
        smu.limiti(2.3)
