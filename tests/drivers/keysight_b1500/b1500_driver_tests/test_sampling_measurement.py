from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500 import constants
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_module import (
    MeasurementNotTaken,
)
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1511B import (
    KeysightB1511B,
)

if TYPE_CHECKING:
    from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_base import (
        KeysightB1500,
    )
    from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1511B import (
        KeysightB1511B,
    )


@pytest.fixture
def smu(b1500: "KeysightB1500") -> "KeysightB1511B":
    return b1500.smu1


@pytest.fixture
def smu_output() -> tuple[int, npt.NDArray[np.float64]]:
    n_samples = 7
    np.random.seed(1)
    data_to_return = np.random.rand(n_samples)
    return n_samples, data_to_return


@pytest.fixture
def smu_sampling_measurement(
    smu: "KeysightB1511B", smu_output: tuple[int, npt.NDArray[np.float64]]
) -> tuple["KeysightB1511B", str, str, str]:
    _, data_to_return = smu_output
    status = "N"
    channel = "A"
    type_ = "I"
    prefix = f"{status}{channel}{type_}"
    visa_data_response = ",".join(prefix + f"{d:+012.3E}" for d in data_to_return)
    smu_sm = smu
    original_ask = smu_sm.root_instrument.ask

    def return_predefined_data_on_xe(cmd: str) -> str:
        if cmd == "XE":
            return visa_data_response
        else:
            return original_ask(cmd)

    smu_sm.root_instrument.ask = Mock(spec_set=smu.root_instrument.ask)  # type: ignore
    smu_sm.root_instrument.ask.side_effect = return_predefined_data_on_xe
    return smu_sm, status, channel, type_


def test_timing_parameters_is_none_at_init(smu: "KeysightB1511B") -> None:
    assert smu._timing_parameters["interval"] is None
    assert smu._timing_parameters["number"] is None
    assert smu._timing_parameters["h_bias"] is None
    assert smu._timing_parameters["h_base"] is None


def test_measurement_requires_timing_parameters_to_be_set(
    smu: "KeysightB1511B",
) -> None:
    with pytest.raises(Exception, match="set timing parameters first"):
        smu.sampling_measurement_trace.get()


def test_sampling_measurement(
    smu_sampling_measurement: tuple["KeysightB1511B", str, str, str],
    smu_output: tuple[int, npt.NDArray[np.float64]],
) -> None:
    smu, _, _, _ = smu_sampling_measurement
    n_samples, data_to_return = smu_output
    smu.timing_parameters(h_bias=0, interval=0.1, number=n_samples)
    actual_data = smu.sampling_measurement_trace.get()

    np.testing.assert_allclose(actual_data, data_to_return, atol=1e-3)
    smu.root_instrument.ask.assert_called_with("XE")


def test_compliance_needs_data_from_sampling_measurement(smu: "KeysightB1511B") -> None:
    with pytest.raises(
        MeasurementNotTaken,
        match="First run sampling_measurement method to generate the data",
    ):
        smu.sampling_measurement_trace.compliance()


def test_compliance(
    smu_sampling_measurement: tuple["KeysightB1511B", str, str, str],
    smu_output: tuple[int, npt.NDArray[np.float64]],
) -> None:
    n_samples, _ = smu_output
    smu, status, _, _ = smu_sampling_measurement
    smu.timing_parameters(h_bias=0, interval=0.1, number=n_samples)
    smu.sampling_measurement_trace.get()
    compliance_list_string = [status] * n_samples
    compliance_list = [
        constants.MeasurementError[i[0]].value for i in compliance_list_string
    ]
    smu_compliance = smu.sampling_measurement_trace.compliance()
    assert isinstance(smu_compliance, list)
    np.testing.assert_array_equal(smu_compliance, compliance_list)


def test_output_data_type_and_data_channel(
    smu_sampling_measurement: tuple["KeysightB1511B", str, str, str],
    smu_output: tuple[int, npt.NDArray[np.float64]],
) -> None:
    n_samples, _ = smu_output
    smu, _, channel, type_ = smu_sampling_measurement
    smu.timing_parameters(h_bias=0, interval=0.1, number=n_samples)
    smu.sampling_measurement_trace.get()

    expected_channel_output = [channel] * n_samples
    expected_channel_output = [
        constants.ChannelName[i].value for i in expected_channel_output
    ]
    expected_type_output = [type_] * n_samples

    data_type = smu.sampling_measurement_trace.data.type
    data_channel = smu.sampling_measurement_trace.data.channel
    np.testing.assert_array_equal(data_type, expected_type_output)
    np.testing.assert_array_equal(data_channel, expected_channel_output)
