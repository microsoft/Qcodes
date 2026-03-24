"""
Extended tests for channel.py to improve coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import (
    ChannelList,
    ChannelTuple,
    InstrumentChannel,
    InstrumentModule,
)
from qcodes.instrument.channel import ChannelTupleValidator
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannel,
    DummyChannelInstrument,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="ch_instr", scope="function")
def _ch_instr() -> Iterator[DummyChannelInstrument]:
    inst = DummyChannelInstrument(name="ch_ext")
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture(name="chan_tuple", scope="function")
def _chan_tuple(ch_instr: DummyChannelInstrument) -> ChannelTuple:
    return ch_instr.channels


@pytest.fixture(name="mutable_list", scope="function")
def _mutable_list(ch_instr: DummyChannelInstrument) -> ChannelList:
    """Create an unlocked ChannelList with some channels."""
    cl = ChannelList(ch_instr, "TestList", DummyChannel)
    # Append two channels
    chan_x = DummyChannel(ch_instr, "ChanX", "X")
    chan_y = DummyChannel(ch_instr, "ChanY", "Y")
    cl.append(chan_x)
    cl.append(chan_y)
    return cl


# ---------------------------------------------------------------------------
# InstrumentModule — __repr__, write/ask proxy, parent/root/name_parts
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_instrument_module_repr(ch_instr: DummyChannelInstrument) -> None:
    chan_a = ch_instr.submodules["A"]
    r = repr(chan_a)
    assert "DummyChannel" in r
    assert "ch_ext" in r
    assert "ChanA" in r


@pytest.mark.serial
def test_instrument_module_write_proxy(ch_instr: DummyChannelInstrument) -> None:
    """write() on a module should proxy to parent, which raises for DummyBase."""
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    # DummyBase.get_idn exists but write_raw is not implemented on Instrument
    # The chain: module.write -> parent.write -> parent.write_raw -> NotImplementedError
    with pytest.raises(NotImplementedError):
        chan_a.write("test_cmd")


@pytest.mark.serial
def test_instrument_module_ask_proxy(ch_instr: DummyChannelInstrument) -> None:
    """ask() on a module should proxy to parent."""
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    with pytest.raises(NotImplementedError):
        chan_a.ask("test_cmd")


@pytest.mark.serial
def test_instrument_module_write_raw_proxy(
    ch_instr: DummyChannelInstrument,
) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    with pytest.raises(NotImplementedError):
        chan_a.write_raw("raw_cmd")


@pytest.mark.serial
def test_instrument_module_ask_raw_proxy(
    ch_instr: DummyChannelInstrument,
) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    with pytest.raises(NotImplementedError):
        chan_a.ask_raw("raw_cmd")


@pytest.mark.serial
def test_instrument_module_parent(ch_instr: DummyChannelInstrument) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.parent is ch_instr


@pytest.mark.serial
def test_instrument_module_root_instrument(
    ch_instr: DummyChannelInstrument,
) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.root_instrument is ch_instr


@pytest.mark.serial
def test_instrument_module_name_parts(ch_instr: DummyChannelInstrument) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.name_parts == ["ch_ext", "ChanA"]


# ---------------------------------------------------------------------------
# ChannelTuple — __reversed__, __contains__, __add__, index, count
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_reversed(chan_tuple: ChannelTuple) -> None:
    channels = list(chan_tuple)
    reversed_channels = list(reversed(chan_tuple))
    assert reversed_channels == list(reversed(channels))


@pytest.mark.serial
def test_channel_tuple_contains(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    chan_a = ch_instr.submodules["A"]
    assert chan_a in chan_tuple
    # Create a channel not in the tuple
    chan_new = DummyChannel(ch_instr, "ChanNew", "N")
    assert chan_new not in chan_tuple


@pytest.mark.serial
def test_channel_tuple_add(ch_instr: DummyChannelInstrument) -> None:
    """Adding two ChannelTuples should produce a combined tuple."""
    channels = ch_instr.channels
    # Split into two halves
    first = channels[0:3]
    second = channels[3:6]
    combined = first + second
    assert len(combined) == 6


@pytest.mark.serial
def test_channel_tuple_add_type_mismatch(
    ch_instr: DummyChannelInstrument,
) -> None:
    """Adding ChannelTuples of different types should raise."""

    class OtherChannel(InstrumentChannel):
        pass

    ct1 = ch_instr.channels
    ct2 = ChannelTuple(ch_instr, "other", OtherChannel)
    with pytest.raises(TypeError, match="same type"):
        ct1 + ct2


@pytest.mark.serial
def test_channel_tuple_add_different_parent() -> None:
    """Adding ChannelTuples with different parents should raise."""
    instr1 = DummyChannelInstrument(name="parent1")
    instr2 = DummyChannelInstrument(name="parent2")
    try:
        ct1 = instr1.channels[0:1]
        ct2 = instr2.channels[0:1]
        with pytest.raises(ValueError, match="same parent"):
            ct1 + ct2
    finally:
        instr1.close()
        instr2.close()


@pytest.mark.serial
def test_channel_tuple_index(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    chan_a = ch_instr.submodules["A"]
    idx = chan_tuple.index(chan_a)  # type: ignore[arg-type]
    assert idx == 0


@pytest.mark.serial
def test_channel_tuple_count(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    chan_a = ch_instr.submodules["A"]
    c = chan_tuple.count(chan_a)  # type: ignore[arg-type]
    assert c == 1


# ---------------------------------------------------------------------------
# ChannelTuple — get_channels_by_name, get_channel_by_name
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_get_channels_by_name(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    subset = chan_tuple.get_channels_by_name("ChanA", "ChanB")
    assert len(subset) == 2


@pytest.mark.serial
def test_get_channels_by_name_empty_raises(chan_tuple: ChannelTuple) -> None:
    with pytest.raises(TypeError, match="one or more names"):
        chan_tuple.get_channels_by_name()


@pytest.mark.serial
def test_get_channel_by_name(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    chan = chan_tuple.get_channel_by_name("ChanA")
    assert chan is ch_instr.submodules["A"]


# ---------------------------------------------------------------------------
# ChannelTuple — get_validator
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_get_validator(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    validator = chan_tuple.get_validator()
    assert isinstance(validator, ChannelTupleValidator)
    # Validate a channel that is in the tuple
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentChannel)
    validator.validate(chan_a)


@pytest.mark.serial
def test_channel_tuple_validator_rejects(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    validator = chan_tuple.get_validator()
    chan_new = DummyChannel(ch_instr, "ChanNew", "N")
    with pytest.raises(ValueError, match="is not part of the expected channel list"):
        validator.validate(chan_new)


@pytest.mark.serial
def test_channel_tuple_validator_requires_channel_tuple() -> None:
    with pytest.raises(ValueError, match="must be a ChannelTuple"):
        ChannelTupleValidator("not a channel tuple")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChannelTuple — repr, snapshot, name_parts, full_name, short_name
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_repr(chan_tuple: ChannelTuple) -> None:
    r = repr(chan_tuple)
    assert "ChannelTuple" in r
    assert "DummyChannel" in r


@pytest.mark.serial
def test_channel_tuple_name_parts(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    assert chan_tuple.short_name == "TempSensors"
    assert chan_tuple.name_parts == ["ch_ext", "TempSensors"]
    assert chan_tuple.full_name == "ch_ext_TempSensors"


@pytest.mark.serial
def test_channel_tuple_snapshot(chan_tuple: ChannelTuple) -> None:
    # The DummyChannelInstrument creates channels with snapshotable=False
    snap = chan_tuple.snapshot_base(update=False)
    assert "snapshotable" in snap
    assert "__class__" in snap


@pytest.mark.serial
def test_channel_tuple_snapshotable() -> None:
    """ChannelTuple with snapshotable=True should include channels in snapshot."""
    instr = DummyChannelInstrument(name="snap_ch")
    try:
        cl = ChannelList(instr, "SnapList", DummyChannel, snapshotable=True)
        chan = DummyChannel(instr, "ChanSnap", "S")
        cl.append(chan)
        ct = cl.to_channel_tuple()
        snap = ct.snapshot_base(update=False)
        assert "channels" in snap
    finally:
        instr.close()


# ---------------------------------------------------------------------------
# ChannelTuple — print_readable_snapshot, invalidate_cache
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_print_readable_snapshot_not_snapshotable(
    ch_instr: DummyChannelInstrument,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """print_readable_snapshot on non-snapshotable tuple should not print channels."""
    ch_instr.channels.print_readable_snapshot(update=False)
    captured = capsys.readouterr()
    # snapshotable=False means nothing should be printed
    assert captured.out == ""


@pytest.mark.serial
def test_channel_tuple_print_readable_snapshot_snapshotable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    instr = DummyChannelInstrument(name="prs_ch")
    try:
        cl = ChannelList(instr, "SnapList", DummyChannel, snapshotable=True)
        chan = DummyChannel(instr, "ChanPRS", "P")
        cl.append(chan)
        ct = cl.to_channel_tuple()
        ct.print_readable_snapshot(update=False)
        captured = capsys.readouterr()
        assert "ChanPRS" in captured.out
    finally:
        instr.close()


@pytest.mark.serial
def test_channel_tuple_invalidate_cache(
    ch_instr: DummyChannelInstrument,
) -> None:
    chan_a = ch_instr.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    chan_a.parameters["temperature"].set(100)
    chan_a.parameters["temperature"].get()
    assert chan_a.parameters["temperature"].cache.valid
    ch_instr.channels.invalidate_cache()
    assert not chan_a.parameters["temperature"].cache.valid


# ---------------------------------------------------------------------------
# ChannelTuple — __getitem__ with slice and tuple index
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_getitem_slice(chan_tuple: ChannelTuple) -> None:
    sliced = chan_tuple[1:3]
    assert isinstance(sliced, ChannelTuple)
    assert len(sliced) == 2


@pytest.mark.serial
def test_channel_tuple_getitem_tuple_index(chan_tuple: ChannelTuple) -> None:
    selected = chan_tuple[(0, 2, 4)]
    assert isinstance(selected, ChannelTuple)
    assert len(selected) == 3


@pytest.mark.serial
def test_channel_tuple_getitem_int(chan_tuple: ChannelTuple) -> None:
    single = chan_tuple[0]
    assert isinstance(single, InstrumentModule)


# ---------------------------------------------------------------------------
# ChannelList — mutation operations
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_list_append(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    chan_z = DummyChannel(ch_instr, "ChanZ", "Z")
    mutable_list.append(chan_z)
    assert len(mutable_list) == 3
    assert chan_z in mutable_list


@pytest.mark.serial
def test_channel_list_append_wrong_type(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    mod = InstrumentModule(ch_instr, "notchan")
    with pytest.raises(TypeError, match="same type"):
        mutable_list.append(mod)  # type: ignore[arg-type]


@pytest.mark.serial
def test_channel_list_extend(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    new_chans = [
        DummyChannel(ch_instr, "ChanE1", "E1"),
        DummyChannel(ch_instr, "ChanE2", "E2"),
    ]
    mutable_list.extend(new_chans)
    assert len(mutable_list) == 4


@pytest.mark.serial
def test_channel_list_extend_wrong_type(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    with pytest.raises(TypeError, match="same type"):
        mutable_list.extend([InstrumentModule(ch_instr, "bad")])  # type: ignore[list-item]


@pytest.mark.serial
def test_channel_list_insert(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    chan_ins = DummyChannel(ch_instr, "ChanIns", "I")
    mutable_list.insert(0, chan_ins)
    assert mutable_list[0] is chan_ins
    assert len(mutable_list) == 3


@pytest.mark.serial
def test_channel_list_insert_wrong_type(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    with pytest.raises(TypeError, match="same type"):
        mutable_list.insert(0, InstrumentModule(ch_instr, "bad"))  # type: ignore[arg-type]


@pytest.mark.serial
def test_channel_list_remove(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    first = mutable_list[0]
    mutable_list.remove(first)
    assert len(mutable_list) == 1
    assert first not in mutable_list


@pytest.mark.serial
def test_channel_list_clear(mutable_list: ChannelList) -> None:
    mutable_list.clear()
    assert len(mutable_list) == 0


@pytest.mark.serial
def test_channel_list_delitem(mutable_list: ChannelList) -> None:
    original_len = len(mutable_list)
    del mutable_list[0]
    assert len(mutable_list) == original_len - 1


# ---------------------------------------------------------------------------
# ChannelList — locked operations raise AttributeError
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_locked_list_append(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "Locked", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanL", "L")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl.append(DummyChannel(ch_instr, "ChanL2", "L2"))


@pytest.mark.serial
def test_locked_list_extend(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedE", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLE", "LE")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl.extend([DummyChannel(ch_instr, "ChanLE2", "LE2")])


@pytest.mark.serial
def test_locked_list_insert(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedI", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLI", "LI")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl.insert(0, DummyChannel(ch_instr, "ChanLI2", "LI2"))


@pytest.mark.serial
def test_locked_list_remove(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedR", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLR", "LR")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl.remove(chan)


@pytest.mark.serial
def test_locked_list_clear(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedC", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLC", "LC")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl.clear()


@pytest.mark.serial
def test_locked_list_delitem(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedD", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLD", "LD")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        del cl[0]


@pytest.mark.serial
def test_locked_list_setitem(ch_instr: DummyChannelInstrument) -> None:
    cl = ChannelList(ch_instr, "LockedS", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanLS", "LS")
    cl.append(chan)
    cl.lock()
    with pytest.raises(AttributeError, match="locked"):
        cl[0] = DummyChannel(ch_instr, "ChanLS2", "LS2")


# ---------------------------------------------------------------------------
# ChannelList — get_validator on unlocked list raises
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_unlocked_list_get_validator(mutable_list: ChannelList) -> None:
    with pytest.raises(AttributeError, match="Cannot create a validator"):
        mutable_list.get_validator()


# ---------------------------------------------------------------------------
# ChannelList — lock / to_channel_tuple
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_list_lock(mutable_list: ChannelList) -> None:
    mutable_list.lock()
    with pytest.raises(AttributeError, match="locked"):
        mutable_list.append(
            DummyChannel(
                mutable_list._parent,  # type: ignore[arg-type]
                "ChanLk",
                "Lk",
            )
        )


@pytest.mark.serial
def test_channel_list_lock_idempotent(mutable_list: ChannelList) -> None:
    """Locking an already-locked list should be a no-op."""
    mutable_list.lock()
    mutable_list.lock()  # should not raise


@pytest.mark.serial
def test_channel_list_to_channel_tuple(mutable_list: ChannelList) -> None:
    ct = mutable_list.to_channel_tuple()
    assert isinstance(ct, ChannelTuple)
    assert not isinstance(ct, ChannelList)
    assert len(ct) == len(mutable_list)


# ---------------------------------------------------------------------------
# ChannelList — repr
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_list_repr(mutable_list: ChannelList) -> None:
    r = repr(mutable_list)
    assert "ChannelList" in r
    assert "DummyChannel" in r


# ---------------------------------------------------------------------------
# ChannelList — __setitem__
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_list_setitem(
    ch_instr: DummyChannelInstrument, mutable_list: ChannelList
) -> None:
    new_chan = DummyChannel(ch_instr, "ChanSet", "S")
    mutable_list[0] = new_chan
    assert mutable_list[0] is new_chan


# ---------------------------------------------------------------------------
# ChannelList — constructed with existing channels (auto-locked)
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_list_init_with_channels(
    ch_instr: DummyChannelInstrument,
) -> None:
    """ChannelList created with a non-empty chan_list should auto-lock."""
    channels = [DummyChannel(ch_instr, f"Ch{i}", str(i)) for i in range(3)]
    cl = ChannelList(ch_instr, "AutoLocked", DummyChannel, chan_list=channels)
    assert cl._locked is True
    with pytest.raises(AttributeError, match="locked"):
        cl.append(DummyChannel(ch_instr, "ChNew", "N"))


# ---------------------------------------------------------------------------
# ChannelTupleValidator with unlocked ChannelList
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_validator_unlocked_list(
    ch_instr: DummyChannelInstrument,
) -> None:
    cl = ChannelList(ch_instr, "UnlockedV", DummyChannel)
    chan = DummyChannel(ch_instr, "ChanUV", "UV")
    cl.append(chan)
    with pytest.raises(AttributeError, match="must be locked"):
        ChannelTupleValidator(cl)


# ---------------------------------------------------------------------------
# ChannelTuple — multi_parameter / multi_function / __getattr__
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_multi_parameter(chan_tuple: ChannelTuple) -> None:
    """multi_parameter should return a MultiChannelInstrumentParameter."""
    mp = chan_tuple.multi_parameter("temperature")
    assert mp is not None


@pytest.mark.serial
def test_channel_tuple_multi_parameter_nonexistent(
    chan_tuple: ChannelTuple,
) -> None:
    with pytest.raises(AttributeError, match="no parameter"):
        chan_tuple.multi_parameter("nonexistent_param")


@pytest.mark.serial
def test_channel_tuple_multi_function(chan_tuple: ChannelTuple) -> None:
    """multi_function should return a callable for functions on channels."""
    mf = chan_tuple.multi_function("log_my_name")
    assert callable(mf)


@pytest.mark.serial
def test_channel_tuple_multi_function_callable(chan_tuple: ChannelTuple) -> None:
    """multi_function should detect callables (methods) on channels."""
    mf = chan_tuple.multi_function("turn_on")
    assert callable(mf)
    mf()  # should not raise


@pytest.mark.serial
def test_channel_tuple_multi_function_nonexistent(
    chan_tuple: ChannelTuple,
) -> None:
    with pytest.raises(AttributeError, match="no callable or function"):
        chan_tuple.multi_function("nonexistent_func")


@pytest.mark.serial
def test_channel_tuple_multi_function_empty() -> None:
    """multi_function on empty tuple raises AttributeError."""
    instr = DummyChannelInstrument(name="empty_mf")
    try:
        empty_ct = ChannelTuple(instr, "empty", DummyChannel)
        with pytest.raises(AttributeError, match="no callable or function"):
            empty_ct.multi_function("anything")
    finally:
        instr.close()


@pytest.mark.serial
def test_channel_tuple_getattr_parameter(chan_tuple: ChannelTuple) -> None:
    """__getattr__ should return a multi-parameter for known parameters."""
    temp = chan_tuple.temperature  # type: ignore[attr-defined]
    assert temp is not None


@pytest.mark.serial
def test_channel_tuple_getattr_channel_by_name(
    ch_instr: DummyChannelInstrument, chan_tuple: ChannelTuple
) -> None:
    """__getattr__ should return a channel by short_name."""
    chan = chan_tuple.ChanA  # type: ignore[attr-defined]
    assert chan is ch_instr.submodules["A"]


@pytest.mark.serial
def test_channel_tuple_getattr_nonexistent(chan_tuple: ChannelTuple) -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        chan_tuple.nonexistent_attr_xyz  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ChannelTuple — __dir__
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_channel_tuple_dir(chan_tuple: ChannelTuple) -> None:
    d = dir(chan_tuple)
    assert "temperature" in d
    assert "ChanA" in d
