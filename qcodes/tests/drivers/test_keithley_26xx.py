import pytest
import numpy as np
from collections import Counter

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

        assert 0 == smu.output()
        smu.output(1)

        assert 0.0 == smu.nplc()
        smu.nplc(2.3)

        assert 0.0 == smu.sourcerange_v()
        some_valid_sourcerange_v = driver._vranges[smu.model][2]
        smu.sourcerange_v(some_valid_sourcerange_v)

        assert 0.0 == smu.source_autorange_v()
        smu.source_autorange_v(1)

        assert 0.0 == smu.measurerange_v()
        some_valid_measurerange_v = driver._vranges[smu.model][2]
        smu.measurerange_v(some_valid_measurerange_v)

        assert 1.0 == smu.measure_autorange_v()
        smu.measure_autorange_v(0)

        assert 0.0 == smu.sourcerange_i()
        some_valid_sourcerange_i = driver._iranges[smu.model][2]
        smu.sourcerange_i(some_valid_sourcerange_i)

        assert 0.0 == smu.source_autorange_i()
        smu.source_autorange_i(1)

        assert 0.0 == smu.measurerange_i()
        some_valid_measurerange_i = driver._iranges[smu.model][2]
        smu.measurerange_i(some_valid_measurerange_i)

        assert 1.0 == smu.measure_autorange_i()
        smu.measure_autorange_i(0)

        assert 0.0 == smu.limitv()
        smu.limitv(2.3)

        assert 0.0 == smu.limiti()
        smu.limiti(2.3)

        assert None == smu.timetrace_mode()
        smu.timetrace_mode('v')

        assert 500 == smu.npts()
        smu.npts(600)

        assert 0.001 == smu.dt()
        smu.dt(0.002)

        dt = smu.dt()
        npts = smu.npts()
        expected_time_axis = np.linspace(0, dt*npts, npts, endpoint=False)
        assert len(expected_time_axis) == len(smu.time_axis())
        assert Counter(expected_time_axis) == Counter(smu.time_axis())
        assert set(expected_time_axis) == set(smu.time_axis())
