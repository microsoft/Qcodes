import pytest

from qcodes.instrument_drivers.Keysight.N9030B import (N9030B,
                                                       SpectrumAnalyzerMode,
                                                       PhaseNoiseMode)
import qcodes.instrument.sims as sims

VISALIB = sims.__file__.replace('__init__.py', 'Keysight_N9030B.yaml@sim')


@pytest.fixture(name="driver")
def _make_driver():
    driver = N9030B('n9030B_sim', address="GPIB::1::INSTR", visalib=VISALIB)
    yield driver
    driver.close()


@pytest.fixture(name="swept_sa")
def _activate_swept_sa_measurement(driver):
    yield driver.sa.swept_sa

@pytest.fixture(name="pn")
def _activate_log_plot_measurement(driver):
    yield driver.pn.log_plot


def test_idn(driver):
    assert {'firmware': '0.1',
            'model': 'N9030B',
            'serial': '1000',
            'vendor': 'Keysight Technologies'} == driver.IDN()


def test_swept_sa_setup(swept_sa):
    assert isinstance(swept_sa, SpectrumAnalyzerMode)

    swept_sa.setup_swept_sa_sweep(1e3, 1e6, 501)
    assert swept_sa.start() == 1e3
    assert swept_sa.stop() == 1e6
    assert swept_sa.npts() == 501


def test_log_plot_setup(log_plot):
    assert isinstance(log_plot, PhaseNoiseMode)

    log_plot.setup_log_plot_sweep(100, 1000, 10001)
    assert log_plot.start_offset() == 100
    assert log_plot.stop_offset() == 1000
    assert log_plot.npts() == 10001
