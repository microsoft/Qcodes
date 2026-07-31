from __future__ import annotations

import struct
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.instrument_drivers.tektronix import (
    TektronixAWG5014,
    TektronixAWG5014Channel,
    TektronixAWG5014Marker,
)
from qcodes.instrument_drivers.tektronix.AWG5014 import parsestr
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function")
def awg() -> Generator[TektronixAWG5014, None, None]:
    awg_sim = TektronixAWG5014(
        "awg_sim",
        address="GPIB0::1::INSTR",
        timeout=1,
        terminator="\n",
        pyvisa_sim_file="Tektronix_AWG5014C.yaml",
    )
    yield awg_sim

    awg_sim.close()


# ── Initialisation and structure ──────────────────────────────────────


def test_init_awg(awg: TektronixAWG5014) -> None:
    idn_dict = awg.IDN()

    assert idn_dict["vendor"] == "QCoDeS"


def test_channel_count(awg: TektronixAWG5014) -> None:
    """The instrument should have exactly 4 output channels."""
    assert len(awg.channels) == 4


def test_channel_type(awg: TektronixAWG5014) -> None:
    """Each channel should be a TektronixAWG5014Channel."""
    for ch in awg.channels:
        assert isinstance(ch, TektronixAWG5014Channel)


def test_marker_subchannels(awg: TektronixAWG5014) -> None:
    """Each channel should have two marker submodules."""
    for ch in awg.channels:
        assert isinstance(ch.m1, TektronixAWG5014Marker)
        assert isinstance(ch.m2, TektronixAWG5014Marker)


def test_channel_submodule_names(awg: TektronixAWG5014) -> None:
    """Channels should be accessible as submodules ch1..ch4."""
    for i in range(1, 5):
        name = f"ch{i}"
        assert name in awg.submodules
        assert awg.submodules[name].channel == i


def test_marker_channel_and_marker_numbers(awg: TektronixAWG5014) -> None:
    """Markers should record their parent channel and marker number."""
    for i, ch in enumerate(awg.channels, start=1):
        assert ch.m1.channel == i
        assert ch.m1.marker == 1
        assert ch.m2.channel == i
        assert ch.m2.marker == 2


# ── Helper functions ──────────────────────────────────────────────────


def test_strips_quotes_and_whitespace() -> None:
    assert parsestr('  "hello"  ') == "hello"


def test_no_quotes() -> None:
    assert parsestr("plain") == "plain"


def test_empty_quoted_string() -> None:
    assert parsestr('""') == ""


def test_strips_only_outer_quotes() -> None:
    assert parsestr('"a"b"') == 'a"b'


def test_strips_trailing_newline(awg: TektronixAWG5014) -> None:
    assert awg.newlinestripper("hello\n") == "hello"


def test_no_trailing_newline(awg: TektronixAWG5014) -> None:
    assert awg.newlinestripper("hello") == "hello"


def test_empty_string(awg: TektronixAWG5014) -> None:
    assert awg.newlinestripper("") == ""


def test_only_newline(awg: TektronixAWG5014) -> None:
    assert awg.newlinestripper("\n") == ""


def test_normal_value(awg: TektronixAWG5014) -> None:
    assert awg._tek_outofrange_get_parser("1.5") == 1.5


def test_out_of_range_value(awg: TektronixAWG5014) -> None:
    assert awg._tek_outofrange_get_parser("9.9e37") == float("INF")


def test_zero(awg: TektronixAWG5014) -> None:
    assert awg._tek_outofrange_get_parser("0") == 0.0


# ── Instrument-level parameters (via sim) ─────────────────────────────


def test_clock_freq_get(awg: TektronixAWG5014) -> None:
    val = awg.clock_freq.get()
    assert val == 1e9


def test_trigger_impedance_get(awg: TektronixAWG5014) -> None:
    val = awg.trigger_impedance.get()
    assert val == 50.0


def test_clock_source_get(awg: TektronixAWG5014) -> None:
    assert awg.clock_source.get() == "INT"


def test_ref_source_get(awg: TektronixAWG5014) -> None:
    assert awg.ref_source.get() == "INT"


def test_trigger_source_get(awg: TektronixAWG5014) -> None:
    assert awg.trigger_source.get() == "INT"


def test_trigger_level_get(awg: TektronixAWG5014) -> None:
    assert awg.trigger_level.get() == 0.0


def test_event_impedance_get(awg: TektronixAWG5014) -> None:
    assert awg.event_impedance.get() == 50.0


def test_event_level_get(awg: TektronixAWG5014) -> None:
    assert awg.event_level.get() == 0.0


def test_run_mode_get(awg: TektronixAWG5014) -> None:
    assert awg.run_mode.get() == "CONT"


def test_trigger_slope_get(awg: TektronixAWG5014) -> None:
    assert awg.trigger_slope.get() == "POS"


def test_event_polarity_get(awg: TektronixAWG5014) -> None:
    assert awg.event_polarity.get() == "POS"


def test_event_jump_timing_get(awg: TektronixAWG5014) -> None:
    assert awg.event_jump_timing.get() == "SYNC"


def test_DC_output_get(awg: TektronixAWG5014) -> None:
    assert awg.DC_output.get() == 0


def test_sequence_length_get(awg: TektronixAWG5014) -> None:
    assert awg.sequence_length.get() == 0


def test_get_state(awg: TektronixAWG5014) -> None:
    assert awg.get_state() == "Idle"


# ── Channel parameters (via sim) ──────────────────────────────────────


def test_channel_state_get(awg: TektronixAWG5014) -> None:
    assert awg.channels[0].state.get() == 0


def test_channel_amp_get(awg: TektronixAWG5014) -> None:
    assert awg.channels[0].amp.get() == 0.5


def test_channel_offset_get(awg: TektronixAWG5014) -> None:
    assert awg.channels[0].offset.get() == 0.0


def test_channel_dc_out_get(awg: TektronixAWG5014) -> None:
    assert awg.channels[0].DC_out.get() == 0.0


def test_channel_filter_get(awg: TektronixAWG5014) -> None:
    # sim returns 9.9e37 → parsed to inf
    assert awg.channels[0].filter.get() == float("inf")


def test_all_channels_have_consistent_defaults(awg: TektronixAWG5014) -> None:
    """All four channels should return the same default values."""
    for ch in awg.channels:
        assert ch.state.get() == 0
        assert ch.amp.get() == 0.5
        assert ch.offset.get() == 0.0


# ── Marker parameters (via sim) ───────────────────────────────────────


def test_marker_delay_get(awg: TektronixAWG5014) -> None:
    ch1 = awg.channels[0]
    assert ch1.m1.delay.get() == 0.0


def test_marker_high_get(awg: TektronixAWG5014) -> None:
    ch1 = awg.channels[0]
    assert ch1.m1.high.get() == 1.0


def test_marker_low_get(awg: TektronixAWG5014) -> None:
    ch1 = awg.channels[0]
    assert ch1.m1.low.get() == 0.0


def test_all_markers_have_consistent_defaults(awg: TektronixAWG5014) -> None:
    """All 8 markers should share the same defaults."""
    for ch in awg.channels:
        for mrk in (ch.m1, ch.m2):
            assert mrk.delay.get() == 0.0
            assert mrk.high.get() == 1.0
            assert mrk.low.get() == 0.0


# ── all_channels_on / all_channels_off ────────────────────────────────


def test_all_channels_off(awg: TektronixAWG5014) -> None:
    awg.all_channels_off()
    for ch in awg.channels:
        assert ch.state.get() == 0


def test_all_channels_on(awg: TektronixAWG5014) -> None:
    awg.all_channels_on()
    for ch in awg.channels:
        assert ch.state.get() == 1


def test_all_channels_on_then_off(awg: TektronixAWG5014) -> None:
    awg.all_channels_on()
    awg.all_channels_off()
    for ch in awg.channels:
        assert ch.state.get() == 0


# ── _pack_waveform ────────────────────────────────────────────────────


def test_basic_pack(awg: TektronixAWG5014) -> None:
    N = 25
    rng = np.random.default_rng(42)
    wf = rng.random(N) * 2 - 1  # values in [-1, 1]
    m1 = rng.integers(0, 2, N)
    m2 = rng.integers(0, 2, N)

    result = awg._pack_waveform(wf, m1, m2)

    assert result is not None
    assert len(result) == N
    assert result.dtype == np.uint16


def test_known_values(awg: TektronixAWG5014) -> None:
    """Verify encoding of specific known inputs."""
    wf = np.array([0.0])
    m1 = np.array([0])
    m2 = np.array([0])
    result = awg._pack_waveform(wf, m1, m2)
    # wf=0 → 0*8191+8191.5 = 8191.5, trunc → 8191
    assert result[0] == 8191


def test_markers_set_correct_bits(awg: TektronixAWG5014) -> None:
    """Marker bits should be at positions 14 and 15."""
    wf = np.array([0.0])
    # m1 only
    r_m1 = awg._pack_waveform(wf, np.array([1]), np.array([0]))
    assert r_m1[0] & 0x4000  # bit 14
    # m2 only
    r_m2 = awg._pack_waveform(wf, np.array([0]), np.array([1]))
    assert r_m2[0] & 0x8000  # bit 15
    # both
    r_both = awg._pack_waveform(wf, np.array([1]), np.array([1]))
    assert r_both[0] & 0xC000 == 0xC000


def test_mismatched_lengths_raises(awg: TektronixAWG5014) -> None:
    with pytest.raises(Exception, match=r"sizes.*do not match"):
        awg._pack_waveform(np.array([0.0, 0.0]), np.array([0]), np.array([0]))


def test_waveform_out_of_bounds_raises(awg: TektronixAWG5014) -> None:
    with pytest.raises(TypeError, match="Waveform values out of bo"):
        awg._pack_waveform(np.array([1.5]), np.array([0]), np.array([0]))


def test_waveform_below_bounds_raises(awg: TektronixAWG5014) -> None:
    with pytest.raises(TypeError, match="Waveform values out of bo"):
        awg._pack_waveform(np.array([-1.5]), np.array([0]), np.array([0]))


def test_invalid_marker1_raises(awg: TektronixAWG5014) -> None:
    with pytest.raises(TypeError, match="Marker 1"):
        awg._pack_waveform(np.array([0.0]), np.array([2]), np.array([0]))


def test_invalid_marker2_raises(awg: TektronixAWG5014) -> None:
    with pytest.raises(TypeError, match="Marker 2"):
        awg._pack_waveform(np.array([0.0]), np.array([0]), np.array([3]))


# ── _pack_record ──────────────────────────────────────────────────────


def test_pack_short(awg: TektronixAWG5014) -> None:
    """Pack a 16-bit integer record."""
    result = awg._pack_record("MAGIC", 5000, "h")
    # name = "MAGIC\0" (6 bytes), data = 2 bytes (short)
    name_size, data_size = struct.unpack_from("<II", result, 0)
    assert name_size == 6
    assert data_size == 2
    # actual data value
    data_val = struct.unpack_from("<h", result, 8 + name_size)[0]
    assert data_val == 5000


def test_pack_double(awg: TektronixAWG5014) -> None:
    """Pack a 64-bit float record."""
    result = awg._pack_record("SAMPLING_RATE", 1e9, "d")
    name_size, data_size = struct.unpack_from("<II", result, 0)
    assert data_size == 8
    data_val = struct.unpack_from("<d", result, 8 + name_size)[0]
    assert data_val == 1e9


def test_pack_string(awg: TektronixAWG5014) -> None:
    """Pack a string record."""
    result = awg._pack_record("TEST_NAME", "hello\x00", "6s")
    name_size, data_size = struct.unpack_from("<II", result, 0)
    assert data_size == 6
    data_str = result[8 + name_size : 8 + name_size + data_size]
    assert data_str == b"hello\x00"


def test_pack_array_of_unsigned_shorts(awg: TektronixAWG5014) -> None:
    """Pack a numpy array as unsigned 16-bit integers."""
    data = np.array([1, 2, 3], dtype=np.uint16)
    result = awg._pack_record("WF_DATA", data, "3H")
    _name_size, data_size = struct.unpack_from("<II", result, 0)
    assert data_size == 6  # 3 * 2 bytes


# ── _file_dict ────────────────────────────────────────────────────────


def test_file_dict(awg: TektronixAWG5014) -> None:
    wf = np.array([0.0, 0.5, -0.5])
    m1 = np.array([0, 1, 0])
    m2 = np.array([1, 0, 1])
    clock = 1e9

    result = awg._file_dict(wf, m1, m2, clock)

    assert result["w"] is not None
    assert result["m1"] is not None
    assert result["m2"] is not None
    assert np.array_equal(result["w"], wf)
    assert np.array_equal(result["m1"], m1)
    assert np.array_equal(result["m2"], m2)
    assert result["clock_freq"] == clock
    assert result["numpoints"] == 3


def test_file_dict_none_clock(awg: TektronixAWG5014) -> None:
    wf = np.array([0.0])
    m1 = np.array([0])
    m2 = np.array([0])

    result = awg._file_dict(wf, m1, m2, None)
    assert result["clock_freq"] is None


# ── parse_marker_channel_name ─────────────────────────────────────────


def test_valid_1m1() -> None:
    result = TektronixAWG5014.parse_marker_channel_name("1M1")
    assert result.channel == 1
    assert result.marker == 1


def test_valid_3m2() -> None:
    result = TektronixAWG5014.parse_marker_channel_name("3M2")
    assert result.channel == 3
    assert result.marker == 2


def test_valid_4m1() -> None:
    result = TektronixAWG5014.parse_marker_channel_name("4M1")
    assert result.channel == 4
    assert result.marker == 1


def test_invalid_raises() -> None:
    with pytest.raises(AssertionError):
        TektronixAWG5014.parse_marker_channel_name("bad")


# ── make_awg_file ─────────────────────────────────────────────────────


def test_basic_awg_file(awg: TektronixAWG5014) -> None:
    N = 25
    rng = np.random.default_rng(42)
    waveforms = [[rng.random(N) * 2 - 1]]
    m1s = [[rng.integers(0, 2, N)]]
    m2s = [[rng.integers(0, 2, N)]]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        [1],
        [0],
        [0],
        [0],
        preservechannelsettings=False,
    )
    assert len(awgfile) > 0
    assert isinstance(awgfile, bytes)


def test_multi_segment_awg_file(awg: TektronixAWG5014) -> None:
    """File with two sequence elements on one channel."""
    N = 10
    rng = np.random.default_rng(123)
    waveforms = [[rng.random(N) * 2 - 1, rng.random(N) * 2 - 1]]
    m1s = [[rng.integers(0, 2, N), rng.integers(0, 2, N)]]
    m2s = [[rng.integers(0, 2, N), rng.integers(0, 2, N)]]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        preservechannelsettings=False,
    )
    assert len(awgfile) > 0


def test_multi_channel_awg_file(awg: TektronixAWG5014) -> None:
    """File with one element each on two channels."""
    N = 10
    rng = np.random.default_rng(456)
    waveforms = [[rng.random(N) * 2 - 1], [rng.random(N) * 2 - 1]]
    m1s = [[rng.integers(0, 2, N)], [rng.integers(0, 2, N)]]
    m2s = [[rng.integers(0, 2, N)], [rng.integers(0, 2, N)]]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        [1],
        [0],
        [0],
        [0],
        preservechannelsettings=False,
    )
    assert len(awgfile) > 0


def test_specific_channels(awg: TektronixAWG5014) -> None:
    """Using the channels parameter to target channels 2 and 4."""
    N = 10
    rng = np.random.default_rng(789)
    waveforms = [[rng.random(N) * 2 - 1], [rng.random(N) * 2 - 1]]
    m1s = [[rng.integers(0, 2, N)], [rng.integers(0, 2, N)]]
    m2s = [[rng.integers(0, 2, N)], [rng.integers(0, 2, N)]]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        [1],
        [0],
        [0],
        [0],
        channels=[2, 4],
        preservechannelsettings=False,
    )
    assert len(awgfile) > 0


def test_flat_input_format(awg: TektronixAWG5014) -> None:
    """make_awg_file also accepts flat (non-nested) waveform lists."""
    N = 10
    rng = np.random.default_rng(111)
    waveforms = [rng.random(N) * 2 - 1]
    m1s = [rng.integers(0, 2, N)]
    m2s = [rng.integers(0, 2, N)]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        [1],
        [0],
        [0],
        [0],
        preservechannelsettings=False,
    )
    assert len(awgfile) > 0


# ── generate_channel_cfg ──────────────────────────────────────────────


def test_returns_dict(awg: TektronixAWG5014) -> None:
    cfg = awg.generate_channel_cfg()
    assert isinstance(cfg, dict)


def test_contains_settings_after_get(awg: TektronixAWG5014) -> None:
    """After getting channel params, they should appear in the config."""
    ch1 = awg.channels[0]
    ch1.amp.get()
    ch1.offset.get()
    ch1.m1.high.get()
    ch1.m1.low.get()

    cfg = awg.generate_channel_cfg()
    assert "ANALOG_AMPLITUDE_1" in cfg
    assert "ANALOG_OFFSET_1" in cfg
    assert "MARKER1_HIGH_1" in cfg
    assert "MARKER1_LOW_1" in cfg


# ── generate_sequence_cfg ─────────────────────────────────────────────


def test_generate_sequence_cfg(awg: TektronixAWG5014) -> None:
    cfg = awg.generate_sequence_cfg()
    assert isinstance(cfg, dict)
    assert cfg["SAMPLING_RATE"] == 1e9
    assert cfg["CLOCK_SOURCE"] == 1  # INT
    assert cfg["REFERENCE_SOURCE"] == 1  # INT
    assert cfg["RUN_MODE"] == 4  # Sequence


# ── Legacy attribute backward compatibility ───────────────────────────


# Tests that the old flat ch{i}_* attribute names still work
# but emit a QCoDeSDeprecationWarning.

CHANNEL_PARAMS = (
    "state",
    "amp",
    "offset",
    "waveform",
    "direct_output",
    "add_input",
    "filter",
    "DC_out",
)
MARKER_PARAMS = (("del", "delay"), ("high", "high"), ("low", "low"))


def test_legacy_channel_param_exists(awg: TektronixAWG5014) -> None:
    """All old ch{i}_{param} names resolve to the correct parameter."""
    for i in range(1, 5):
        for param in CHANNEL_PARAMS:
            old_name = f"ch{i}_{param}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", QCoDeSDeprecationWarning)
                old_attr = getattr(awg, old_name)
            new_attr = getattr(awg.submodules[f"ch{i}"], param)
            assert old_attr is new_attr, f"{old_name} did not resolve to ch{i}.{param}"


def test_legacy_marker_param_exists(awg: TektronixAWG5014) -> None:
    """All old ch{i}_m{j}_{param} names resolve to the correct parameter."""
    for i in range(1, 5):
        for j in (1, 2):
            for old_suffix, new_name in MARKER_PARAMS:
                old_name = f"ch{i}_m{j}_{old_suffix}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", QCoDeSDeprecationWarning)
                    old_attr = getattr(awg, old_name)
                new_attr = getattr(
                    awg.submodules[f"ch{i}"].submodules[f"m{j}"], new_name
                )
                assert old_attr is new_attr, (
                    f"{old_name} did not resolve to ch{i}.m{j}.{new_name}"
                )


def test_legacy_channel_param_warns(awg: TektronixAWG5014) -> None:
    """Accessing an old channel param name emits QCoDeSDeprecationWarning."""
    with pytest.warns(QCoDeSDeprecationWarning, match="ch1_amp.*ch1.amp"):
        _ = awg.ch1_amp


def test_legacy_marker_param_warns(awg: TektronixAWG5014) -> None:
    """Accessing an old marker param name emits QCoDeSDeprecationWarning."""
    with pytest.warns(QCoDeSDeprecationWarning, match="ch2_m1_high.*ch2.m1.high"):
        _ = awg.ch2_m1_high


def test_legacy_marker_del_warns(awg: TektronixAWG5014) -> None:
    """The renamed 'del' -> 'delay' param emits a correct warning."""
    with pytest.warns(QCoDeSDeprecationWarning, match="ch3_m2_del.*ch3.m2.delay"):
        _ = awg.ch3_m2_del


def test_nonexistent_attr_raises(awg: TektronixAWG5014) -> None:
    """An attribute that doesn't match any legacy name still raises."""
    with pytest.raises(AttributeError, match="no_such_attr"):
        _ = awg.no_such_attr


def test_nonexistent_legacy_style_raises(awg: TektronixAWG5014) -> None:
    """A ch{i}_* name that doesn't map to a real param still raises."""
    with pytest.raises(AttributeError):
        _ = awg.ch1_bogus_param
