import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from qcodes.instrument.mockers.simulated_ats_api import SimulatedATS9360API
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionInterface
from qcodes.instrument_drivers.AlazarTech.ATS9360 import AlazarTechATS9360 as ATS9360

if TYPE_CHECKING:
    from collections.abc import Generator


def ones_generator(data: npt.NDArray) -> None:
    data[:] = np.ones(data.shape)


@pytest.fixture(scope="function")
def simulated_alazar() -> "Generator[ATS9360, None, None]":
    driver = ATS9360(
        "Alazar",
        api=SimulatedATS9360API(dll_path="simulated", buffer_generator=ones_generator),
    )
    with driver.syncing():
        driver.sample_rate(1_000_000_000)
    try:
        yield driver
    finally:
        if driver:
            driver.close()


@pytest.fixture(scope="function")
def alazar_ctrl() -> "Generator[AcquisitionInterface[list[npt.NDArray]], None, None]":
    class TestAcquisitionController(AcquisitionInterface[list[npt.NDArray]]):
        def __init__(self) -> None:
            self.buffers: list[npt.NDArray] = []

        def handle_buffer(
            self, buffer: npt.NDArray, buffer_number: int | None = None
        ) -> None:
            self.buffers.append(np.copy(buffer))

        def post_acquire(self) -> list[npt.NDArray]:
            return self.buffers

    yield TestAcquisitionController()


@pytest.mark.win32
def test_simulated_alazar(
    simulated_alazar: ATS9360,
    alazar_ctrl: AcquisitionInterface[list[npt.NDArray]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    alazar = simulated_alazar
    buffers_per_acquisition = 10
    with caplog.at_level(logging.DEBUG):
        data = alazar.acquire(
            mode="NPT",
            samples_per_record=int(1e4 * 128),
            records_per_buffer=1,
            buffers_per_acquisition=buffers_per_acquisition,
            channel_selection="A",
            enable_record_headers=None,
            alloc_buffers=None,
            fifo_only_streaming=None,
            interleave_samples=None,
            get_processed_data=None,
            allocated_buffers=4,
            buffer_timeout=None,
            # TODO: AlazarTechATS.acquire annotates acquisition_controller as
            # AcquisitionController[OutputType] but only uses methods from
            # AcquisitionInterface, so any AcquisitionInterface subclass works
            # at runtime. AcquisitionController also subclasses
            # AcquisitionInterface[Any], so OutputType is never bound to the
            # return type. Widen the annotation to
            # AcquisitionInterface[OutputType] and remove this ignore.
            acquisition_controller=alazar_ctrl,  # pyright: ignore[reportArgumentType]
        )
    assert "tot acquire time" in caplog.records[-1].msg
    assert len(data) == buffers_per_acquisition

    for d in data:
        assert np.allclose(d, np.ones(d.shape))
