import pytest
import numpy as np
from qcodes.instrument.mockers.simulated_ats_api import SimulatedATS9360API
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionInterface
from qcodes.instrument_drivers.AlazarTech.ATS9360 import (
    AlazarTech_ATS9360 as ATS9360)


def ones_generator(data):
    data[:] = np.ones(data.shape)


@pytest.fixture(scope='function')
def simulated_alazar():
    driver = ATS9360(
        'Alazar',
        api=SimulatedATS9360API(
            dll_path='simulated',
            buffer_generator=ones_generator)
    )
    with driver.syncing():
        driver.sample_rate(1_000_000_000)
    try:
        yield driver
    finally:
        if driver:
            driver.close()


@pytest.fixture(scope='function')
def alazar_ctrl():
    class TestAcquisitionController(AcquisitionInterface):

        def __init__(self):
            self.buffers = []

        def handle_buffer(self, buffer, buffer_number=None):
            self.buffers.append(np.copy(buffer))

        def post_acquire(self):
            return self.buffers

    yield TestAcquisitionController()


@pytest.mark.win32
def test_simulated_alazar(simulated_alazar, alazar_ctrl):
    alazar = simulated_alazar
    buffers_per_acquisition = 10
    data = alazar.acquire(
        mode='NPT',
        samples_per_record=int(1e4*128),
        records_per_buffer=1,
        buffers_per_acquisition=buffers_per_acquisition,
        channel_selection='A',
        enable_record_headers=None,
        alloc_buffers=None,
        fifo_only_streaming=None,
        interleave_samples=None,
        get_processed_data=None,
        allocated_buffers=4,
        buffer_timeout=None,
        acquisition_controller=alazar_ctrl)
    assert len(data) == buffers_per_acquisition
    for d in data:
        assert np.allclose(d, np.ones(d.shape))
