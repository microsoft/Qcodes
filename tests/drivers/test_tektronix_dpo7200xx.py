from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from qcodes.instrument_drivers.tektronix.DPO7200xx import TektronixDPO7000xx

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def tektronix_dpo() -> "Generator[TektronixDPO7000xx, None, None]":
    """
    A six channel-per-relay instrument
    """
    driver = TektronixDPO7000xx(
        "dpo",
        address="TCPIP0::0.0.0.0::inst0::INSTR",
        pyvisa_sim_file="Tektronix_DPO7200xx.yaml",
    )

    yield driver
    driver.close()


def test_adjust_timer(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    """
    After adjusting the type of the measurement or the source of the
    measurement, we need wait at least 0.1 seconds
    ('minimum_adjustment_time') before a measurement value can be
    retrieved. Test this.
    """
    measurement = tektronix_dpo.measurement[0]
    min_time = measurement._minimum_adjustment_time

    mock_time = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")

    # Simulate: source was set at t=1.0, measurement read at t=1.05
    # (only 0.05s elapsed, less than the 0.1s minimum)
    mock_time.perf_counter.return_value = 1.05
    measurement._adjustment_time = 1.0
    measurement.wait_adjustment_time()
    mock_time.sleep.assert_called_once_with(pytest.approx(min_time - 0.05))

    # Simulate: enough time has passed (0.2s > 0.1s minimum)
    mock_time.reset_mock()
    mock_time.perf_counter.return_value = 1.2
    measurement._adjustment_time = 1.0
    measurement.wait_adjustment_time()
    mock_time.sleep.assert_not_called()

    # Verify _set_source records the adjustment time
    mock_time.perf_counter.return_value = 5.0
    measurement._set_source(1, "CH1")
    assert measurement._adjustment_time == 5.0

    # Verify _set_measurement_type records the adjustment time
    mock_time.perf_counter.return_value = 6.0
    measurement._set_measurement_type("AMPlitude")
    assert measurement._adjustment_time == 6.0


def test_measurements_return_float(tektronix_dpo: TektronixDPO7000xx) -> None:
    amplitude = tektronix_dpo.measurement[0].amplitude()
    assert isinstance(amplitude, float)

    mean_amplitude = tektronix_dpo.measurement[0].amplitude.mean()
    assert isinstance(mean_amplitude, float)


def test_measurement_sets_state(tektronix_dpo: TektronixDPO7000xx) -> None:
    tektronix_dpo.measurement[1].frequency()
    assert tektronix_dpo.measurement[1].state() is True


def test_record_length(tektronix_dpo: TektronixDPO7000xx) -> None:
    record_length = tektronix_dpo.horizontal.record_length()
    assert record_length == 1_000_000
    assert isinstance(record_length, int)

    tektronix_dpo.horizontal.record_length(1000)
    assert tektronix_dpo.horizontal.record_length() == 1000


def test_single_waits_until_acquiring_and_ready(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    write = mocker.patch.object(tektronix_dpo, "write")
    ask = mocker.patch.object(
        tektronix_dpo,
        "ask",
        side_effect=["0\n", "1\n", "1\n", "0\n", "1\n", "1\n"],
    )
    clock = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")
    clock.time.return_value = 0.0

    tektronix_dpo.single()

    assert write.call_args_list == [
        mocker.call("ACQuire:STOPAfter SEQUENCE"),
        mocker.call("ACQuire:STATE RUN"),
    ]
    assert (
        ask.call_args_list
        == [
            mocker.call("ACQuire:STATE?"),
            mocker.call("TRIGger:A:READY?"),
        ]
        * 3
    )
    assert clock.sleep.call_args_list == [mocker.call(0.1)] * 2


@pytest.mark.parametrize("timeout", [0.5, None])
def test_single_timeout(
    tektronix_dpo: TektronixDPO7000xx,
    mocker: "MockerFixture",
    timeout: float | None,
) -> None:
    mocker.patch.object(tektronix_dpo, "timeout", return_value=timeout)
    mocker.patch.object(tektronix_dpo, "write")
    ask = mocker.patch.object(tektronix_dpo, "ask", return_value="0\n")
    clock = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")
    timeout_seconds = timeout if timeout is not None else 10.0
    clock.time.side_effect = [0.0, 0.0, timeout_seconds + 1.0]

    tektronix_dpo.single()

    assert ask.call_count == 4
    clock.sleep.assert_called_once_with(0.1)


@pytest.mark.parametrize("enabled_channels", [(), (0, 2), (0, 1, 2, 3)])
def test_download_waveforms(
    tektronix_dpo: TektronixDPO7000xx,
    mocker: "MockerFixture",
    enabled_channels: tuple[int, ...],
) -> None:
    ask = mocker.patch.object(
        tektronix_dpo,
        "ask",
        side_effect=[
            "1\n",
            "0\n",
            *("1\n" if i in enabled_channels else "0\n" for i in range(4)),
        ],
    )
    record_length = mocker.patch.object(
        tektronix_dpo.horizontal, "record_length", return_value=3
    )
    traces = [np.arange(3, dtype=float) + i for i in range(4)]
    set_lengths = [
        mocker.patch.object(channel, "set_trace_length")
        for channel in tektronix_dpo.channel
    ]
    get_traces = [
        mocker.patch.object(channel.waveform, "trace", return_value=trace)
        for channel, trace in zip(tektronix_dpo.channel, traces)
    ]
    clock = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")
    clock.time.return_value = 0.0

    waveforms = tektronix_dpo.download_waveforms()

    assert isinstance(waveforms, tuple)
    assert len(waveforms) == len(enabled_channels)
    for waveform, index in zip(waveforms, enabled_channels):
        np.testing.assert_array_equal(waveform, traces[index])
    for i, (set_length, get_trace) in enumerate(zip(set_lengths, get_traces)):
        if i in enabled_channels:
            set_length.assert_called_once_with(3)
            get_trace.assert_called_once_with()
        else:
            set_length.assert_not_called()
            get_trace.assert_not_called()
    assert record_length.call_count == len(enabled_channels)
    assert ask.call_args_list == [
        mocker.call("ACQuire:STATE?"),
        mocker.call("ACQuire:STATE?"),
        *(mocker.call(f"SELect:CH{i}?") for i in range(1, 5)),
    ]
    clock.sleep.assert_called_once_with(0.1)


@pytest.mark.parametrize("timeout", [0.5, None])
def test_download_waveforms_timeout(
    tektronix_dpo: TektronixDPO7000xx,
    mocker: "MockerFixture",
    timeout: float | None,
) -> None:
    mocker.patch.object(tektronix_dpo, "timeout", return_value=timeout)
    ask = mocker.patch.object(
        tektronix_dpo, "ask", side_effect=["1\n", "1\n", "0", "0", "0", "0"]
    )
    clock = mocker.patch("qcodes.instrument_drivers.tektronix.DPO7200xx.time")
    timeout_seconds = timeout if timeout is not None else 60.0
    clock.time.side_effect = [0.0, 0.0, timeout_seconds + 1.0]

    assert tektronix_dpo.download_waveforms() == ()

    assert ask.call_count == 6
    clock.sleep.assert_called_once_with(0.1)


def test_get_timebase(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    expected_axis = np.array([0.0, 1e-9, 2e-9])
    trace_axis = mocker.patch.object(
        tektronix_dpo.channel[0].waveform, "trace_axis", return_value=expected_axis
    )
    trace_axis.unit = "s"

    unit, axis = tektronix_dpo.get_timebase()

    assert unit == "s"
    np.testing.assert_array_equal(axis, expected_axis)
    trace_axis.assert_called_once_with()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ON", True), ("off", False), (1, True), (0, False)],
)
def test_measurement_state_accepts_on_off_strings_and_ints(
    tektronix_dpo: TektronixDPO7000xx, value: str | int, expected: bool
) -> None:
    # while it does not type check, on/off strings and ints are accepted
    tektronix_dpo.measurement[1].state(value)  # pyright: ignore[reportArgumentType]
    assert tektronix_dpo.measurement[1].state() is expected


# raw samples in digitizing levels (valid for all binary formats) and the same
# samples scaled with the simulated preamble: (raw - YOFF) * YMULT + YZERO
# with YOFF = -50, YMULT = 4 mV and YZERO = -100 mV
RAW_SAMPLES = [0, 50, 75]
SCALED_SAMPLES = [0.1, 0.3, 0.4]


def test_waveform_preamble(tektronix_dpo: TektronixDPO7000xx) -> None:
    waveform = tektronix_dpo.channel[0].waveform
    raw_data_offset = waveform.raw_data_offset()
    assert raw_data_offset == -50
    assert isinstance(raw_data_offset, float)
    assert waveform.scale() == 4e-3
    assert waveform.offset() == -0.1


def test_trace_ascii(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    tektronix_dpo.waveform.is_binary(False)
    query = mocker.patch.object(
        tektronix_dpo.visa_handle,
        "query_ascii_values",
        return_value=np.array(RAW_SAMPLES, dtype=float),
    )

    trace = tektronix_dpo.channel[0].waveform.trace()

    query.assert_called_once_with("CURVE?", container=np.ndarray)
    assert trace.dtype == np.float64
    np.testing.assert_allclose(trace, SCALED_SAMPLES)


@pytest.mark.parametrize(
    ("data_format", "bytes_per_sample", "is_big_endian", "datatype"),
    [
        ("signed_integer", 1, False, "b"),
        ("unsigned_integer", 1, False, "B"),
        ("signed_integer", 2, True, "h"),
        ("unsigned_integer", 2, False, "H"),
        ("floating_point", 4, False, "f"),
        ("floating_point", 8, True, "d"),
    ],
)
def test_trace_binary(
    tektronix_dpo: TektronixDPO7000xx,
    mocker: "MockerFixture",
    data_format: Literal["signed_integer", "unsigned_integer", "floating_point"],
    bytes_per_sample: Literal[1, 2, 4, 8],
    is_big_endian: bool,
    datatype: str,
) -> None:
    waveform_format = tektronix_dpo.waveform
    waveform_format.is_binary(True)
    waveform_format.data_format(data_format)
    waveform_format.bytes_per_sample(bytes_per_sample)
    waveform_format.is_big_endian(is_big_endian)
    # integer samples are returned as an integer array, which must still be
    # scaled to floats
    query = mocker.patch.object(
        tektronix_dpo.visa_handle,
        "query_binary_values",
        return_value=np.array(RAW_SAMPLES, dtype=np.dtype(datatype)),
    )

    trace = tektronix_dpo.channel[0].waveform.trace()

    query.assert_called_once_with(
        "CURVE?",
        datatype=datatype,
        is_big_endian=is_big_endian,
        container=np.ndarray,
    )
    assert trace.dtype == np.float64
    np.testing.assert_allclose(trace, SCALED_SAMPLES)


def test_trace_binary_unsupported_format(
    tektronix_dpo: TektronixDPO7000xx, mocker: "MockerFixture"
) -> None:
    waveform_format = tektronix_dpo.waveform
    waveform_format.is_binary(True)
    waveform_format.data_format("floating_point")
    waveform_format.bytes_per_sample(1)
    query = mocker.patch.object(tektronix_dpo.visa_handle, "query_binary_values")

    with pytest.raises(ValueError, match="Unsupported combination"):
        tektronix_dpo.channel[0].waveform.trace()
    query.assert_not_called()
