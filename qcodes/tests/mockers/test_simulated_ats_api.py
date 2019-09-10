import pytest
import numpy as np
from qcodes.instrument.mockers.simulated_ats_api import SimulatedAlazarATSAPI
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionInterface
from qcodes.instrument_drivers.AlazarTech.ATS9360 import (
    AlazarTech_ATS9360 as ATS9360)



@pytest.fixture(scope='function')
def simulated_alazar():
    driver = ATS9360(
        'Alazar',
        api=SimulatedAlazarATSAPI(dll_path='simulated'))
    with driver.syncing():
        driver.sample_rate(1_000_000_000)
        driver.decimation(1)
        driver.trigger_delay(0)
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


def test_simulated_alazar(simulated_alazar, alazar_ctrl):
    alazar = simulated_alazar
    alazar.aux_io_mode('AUX_IN_TRIGGER_ENABLE')
    alazar.aux_io_param('TRIG_SLOPE_POSITIVE')
    alazar.sync_settings_to_card()

    alazar.acquire(
        mode='NPT',
        samples_per_record=int(1e4*128),
        records_per_buffer=1,
        buffers_per_acquisition=1,
        channel_selection='A',
        enable_record_headers=None,
        alloc_buffers=None,
        fifo_only_streaming=None,
        interleave_samples=None,
        get_processed_data=None,
        allocated_buffers=4,
        buffer_timeout=None,
        acquisition_controller=alazar_ctrl)


