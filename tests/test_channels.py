import logging
from collections.abc import Generator, Sequence

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_array_equal
from pytest import LogCaptureFixture

from qcodes.instrument import ChannelList, ChannelTuple, Instrument, InstrumentChannel
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannel,
    DummyChannelInstrument,
)


@pytest.fixture(scope="function", name="dci")
def _make_dci() -> Generator[DummyChannelInstrument, None, None]:
    dci = DummyChannelInstrument(name="dci")
    try:
        yield dci
    finally:
        dci.close()


@pytest.fixture(scope="function", name="dci_with_list")
def _make_dci_with_list() -> Generator[Instrument, None, None]:
    for i in range(10):
        pass

    dci = Instrument(name="dciwl")
    channels = ChannelList(dci, "ListElem", DummyChannel, snapshotable=False)
    for chan_name in ("A", "B", "C", "D", "E", "F"):
        channel = DummyChannel(dci, f"Chan{chan_name}", chan_name)
        channels.append(channel)
        dci.add_submodule(chan_name, channel)
    dci.add_submodule("channels", channels)

    try:
        yield dci
    finally:
        dci.close()


@pytest.fixture(scope="function", name="empty_instrument")
def _make_empty_instrument() -> Generator[Instrument, None, None]:
    instr = Instrument(name="dci")

    try:
        yield instr
    finally:
        instr.close()


class EmptyChannel(InstrumentChannel):
    pass


def test_instrument_channel_label(request) -> None:
    dci = DummyChannelInstrument(name="dci_with_labels", label="Instrument Label")
    request.addfinalizer(dci.close)
    channel = DummyChannel(dci, "A_with_label", "A_wl", label="A with f@ncy label")
    dci.add_submodule("A_with_label", channel)
    channel_2 = EmptyChannel(dci, "B_with_label", label="B with f@ncy label")
    dci.add_submodule("B_with_label", channel_2)
    assert dci.label == "Instrument Label"
    assert dci.A_with_label.label == "A with f@ncy label"
    assert dci.B_with_label.label == "B with f@ncy label"


def test_channels_call_function(dci, caplog: LogCaptureFixture) -> None:
    """
    Test that dci.channels.some_function() calls
    some_function on each of the channels
    """
    with caplog.at_level(
        logging.DEBUG, logger="qcodes.instrument_drivers.mock_instruments"
    ):
        caplog.clear()
        dci.channels.log_my_name()
        mssgs = [rec.message for rec in caplog.records]
        names = [ch.name.replace("dci_", "") for ch in dci.channels]
        assert mssgs == names


def test_channels_get(dci) -> None:
    temperatures = dci.channels.temperature.get()
    assert len(temperatures) == 6


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(value=hst.floats(0, 300), channel=hst.integers(0, 3))
def test_channel_access_is_identical(dci, value, channel) -> None:
    channel_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    label = channel_to_label[channel]
    channel_via_label = getattr(dci, label)
    channel_via_name = dci.channels.get_channel_by_name(f"Chan{label}")
    # set via labeled channel
    channel_via_label.temperature(value)
    assert channel_via_label.temperature() == value
    assert channel_via_name.temperature() == value
    assert dci.channels[channel].temperature() == value
    assert dci.channels.temperature()[channel] == value
    # reset via channel name
    channel_via_name.temperature(0)
    assert channel_via_label.temperature() == 0
    assert channel_via_name.temperature() == 0
    assert dci.channels[channel].temperature() == 0
    assert dci.channels.temperature()[channel] == 0
    # set via index into list
    dci.channels[channel].temperature(value)
    assert channel_via_label.temperature() == value
    assert channel_via_name.temperature() == value
    assert dci.channels[channel].temperature() == value
    assert dci.channels.temperature()[channel] == value
    # it's not possible to set via dci.channels.temperature
    # as this is a multi parameter that currently does not support set.


def test_invalid_channel_type_raises(empty_instrument: Instrument) -> None:
    with pytest.raises(
        ValueError,
        match="ChannelTuple can only hold instances of type InstrumentChannel",
    ):
        ChannelList(
            parent=empty_instrument,
            name="empty",
            chan_type=int,  # type: ignore[type-var]
        )


def test_invalid_multichan_type_raises(empty_instrument: Instrument) -> None:
    with pytest.raises(ValueError, match="multichan_paramclass must be a"):
        ChannelList(
            parent=empty_instrument,
            name="empty",
            chan_type=DummyChannel,
            multichan_paramclass=int,  # type: ignore[arg-type]
        )


def test_wrong_chan_type_raises(empty_instrument) -> None:
    with pytest.raises(TypeError, match="All items in this ChannelTuple must be of"):
        ChannelList(
            parent=empty_instrument,
            name="empty",
            chan_type=DummyChannel,
            chan_list=[EmptyChannel(parent=empty_instrument, name="empty_channel")],
        )


def test_append_channel(dci_with_list) -> None:
    n_channels_pre = len(dci_with_list.channels)
    n_channels_post = n_channels_pre + 1
    chan_num = "11"
    name = f"Chan{chan_num}"

    channel = DummyChannel(dci_with_list, name, chan_num)
    dci_with_list.channels.append(channel)
    dci_with_list.add_submodule(name, channel)

    assert len(dci_with_list.channels) == n_channels_post

    dci_with_list.channels.lock()
    # after locking the channels it's not possible to add any more channels
    with pytest.raises(AttributeError):
        name = "bar"
        channel = DummyChannel(dci_with_list, "Chan" + name, name)
        dci_with_list.channels.append(channel)
    assert len(dci_with_list.channels) == n_channels_post


def test_append_channel_wrong_type_raises(dci_with_list) -> None:
    n_channels = len(dci_with_list.channels)

    channel = EmptyChannel(dci_with_list, "foo")
    with pytest.raises(TypeError, match="All items in a channel list must"):
        dci_with_list.channels.append(channel)

    assert len(dci_with_list.channels) == n_channels


def test_extend_channels_from_generator(dci_with_list) -> None:
    n_channels = len(dci_with_list.channels)
    names = ("foo", "bar", "foobar")
    channels = (DummyChannel(dci_with_list, "Chan" + name, name) for name in names)
    dci_with_list.channels.extend(channels)

    assert len(dci_with_list.channels) == n_channels + len(names)


def test_extend_channels_from_tuple(dci_with_list) -> None:
    n_channels = len(dci_with_list.channels)
    names = ("foo", "bar", "foobar")
    channels = tuple(DummyChannel(dci_with_list, "Chan" + name, name) for name in names)
    dci_with_list.channels.extend(channels)

    assert len(dci_with_list.channels) == n_channels + len(names)


def test_extend_wrong_type_raises(dci_with_list) -> None:
    names = ("foo", "bar", "foobar")
    channels = tuple(EmptyChannel(dci_with_list, "Chan" + name) for name in names)
    with pytest.raises(
        TypeError, match="All items in a channel list must be of the same type."
    ):
        dci_with_list.channels.extend(channels)


def test_extend_locked_list_raises(dci_with_list) -> None:
    dci_with_list.channels.lock()
    names = ("foo", "bar", "foobar")
    channels = tuple(EmptyChannel(dci_with_list, "Chan" + name) for name in names)
    with pytest.raises(AttributeError, match="Cannot extend a locked channel list"):
        dci_with_list.channels.extend(channels)


def test_extend_then_remove(dci_with_list) -> None:
    n_channels = len(dci_with_list.channels)
    names = ("foo", "bar", "foobar")
    channels = [DummyChannel(dci_with_list, "Chan" + name, name) for name in names]
    dci_with_list.channels.extend(channels)

    assert len(dci_with_list.channels) == n_channels + len(names)
    last_channel = dci_with_list.channels[-1]
    dci_with_list.channels.remove(last_channel)
    assert last_channel not in dci_with_list.channels
    assert len(dci_with_list.channels) == n_channels + len(names) - 1


def test_insert_channel(dci_with_list) -> None:
    n_channels_pre = len(dci_with_list.channels)
    name = "foo"
    channel = DummyChannel(dci_with_list, "Chan" + name, name)
    dci_with_list.channels.insert(1, channel)
    dci_with_list.add_submodule(name, channel)

    n_channels_post = n_channels_pre + 1

    assert dci_with_list.channels.get_channel_by_name(f"Chan{name}") is channel
    assert len(dci_with_list.channels) == n_channels_post
    assert dci_with_list.channels[1] is channel
    dci_with_list.channels.lock()
    # after locking the channels it's not possible to add any more channels
    with pytest.raises(AttributeError):
        name = "bar"
        channel = DummyChannel(dci_with_list, "Chan" + name, name)
        dci_with_list.channels.insert(2, channel)
    assert len(dci_with_list.channels) == n_channels_post
    assert len(dci_with_list.channels._channel_mapping) == n_channels_post


def test_insert_channel_wrong_type_raises(dci_with_list) -> None:
    with pytest.raises(TypeError, match="All items in a channel list"):
        dci_with_list.channels.insert(1, EmptyChannel(parent=dci_with_list, name="foo"))


def test_add_none_channel_tuple_to_channel_tuple_raises(dci) -> None:
    with pytest.raises(TypeError, match="Can't add objects of type"):
        _ = dci.channels + [1]


def test_add_channel_tuples_of_different_types_raises(dci) -> None:
    extra_channels = [EmptyChannel(dci, f"chan{i}") for i in range(10)]
    extra_channel_list = ChannelList(
        parent=dci,
        name="extra_channels",
        chan_type=EmptyChannel,
        chan_list=extra_channels,
    )
    dci.add_submodule("extra_channels", extra_channel_list)

    with pytest.raises(TypeError, match="Both l and r arguments to add must contain"):
        _ = dci.channels + extra_channel_list


def test_add_channel_tuples_from_different_parents(dci, dci_with_list) -> None:
    with pytest.raises(ValueError, match="Can only add channels from the same"):
        _ = dci.channels + dci_with_list.channels


def test_chan_tuple_repr(dci) -> None:
    dci_repr = repr(dci.channels)
    assert dci_repr.startswith("ChannelTuple")


def test_chan_list_repr(dci_with_list) -> None:
    dci_repr = repr(dci_with_list.channels)
    assert dci_repr.startswith("ChannelList")


def test_channel_tuple_get_validator(dci) -> None:
    validator = dci.channels.get_validator()
    for chan in dci.channels:
        validator.validate(chan)


def test_channel_list_get_validator(dci_with_list) -> None:
    dci_with_list.channels.lock()
    validator = dci_with_list.channels.get_validator()
    for chan in dci_with_list.channels:
        validator.validate(chan)


def test_channel_list_get_validator_not_locked_raised(dci_with_list) -> None:
    with pytest.raises(AttributeError, match="Cannot create a validator"):
        dci_with_list.channels.get_validator()


def test_channel_tuple_index(dci) -> None:
    for i, chan in enumerate(dci.channels):
        assert dci.channels.index(chan) == i


def test_channel_tuple_snapshot(dci) -> None:
    snapshot = dci.channels.snapshot()
    assert snapshot["snapshotable"] is False
    assert len(snapshot.keys()) == 2


def test_channel_tuple_snapshot_enabled(empty_instrument) -> None:
    channels = ChannelList(
        empty_instrument, "ListElem", DummyChannel, snapshotable=True
    )
    for chan_name in ("A", "B", "C", "D", "E", "F"):
        channel = DummyChannel(empty_instrument, f"Chan{chan_name}", chan_name)
        channels.append(channel)
    empty_instrument.add_submodule("channels", channels)

    snapshot = empty_instrument.channels.snapshot()
    assert snapshot["snapshotable"] is True
    assert len(snapshot.keys()) == 3
    assert "channels" in snapshot.keys()


def test_channel_tuple_dir(dci) -> None:
    dir_list = dir(dci.channels)

    for chan in dci.channels:
        assert chan.short_name in dir_list

    for param in dci.channels[0].parameters.values():
        assert param.short_name in dir_list


def test_clear_channels(dci_with_list) -> None:
    channels = dci_with_list.channels
    channels.clear()
    assert len(channels) == 0


def test_clear_locked_channels(dci_with_list) -> None:
    channels = dci_with_list.channels
    original_length = len(channels)
    channels.lock()
    with pytest.raises(AttributeError):
        channels.clear()
    assert len(channels) == original_length


def test_remove_channel(dci_with_list) -> None:
    channels = dci_with_list.channels
    chan_a = dci_with_list.A
    original_length = len(channels.temperature())
    channels.remove(chan_a)
    with pytest.raises(AttributeError):
        getattr(channels, chan_a.short_name)
    assert len(channels) == original_length - 1
    assert len(channels.temperature()) == original_length - 1


def test_remove_locked_channel(dci_with_list) -> None:
    channels = dci_with_list.channels
    chan_a = dci_with_list.A
    channels.lock()
    with pytest.raises(AttributeError):
        channels.remove(chan_a)


def test_channel_list_lock_twice(dci_with_list) -> None:
    channels = dci_with_list.channels
    channels.lock()
    # locking twice should be a no op
    channels.lock()


def test_remove_tupled_channel(dci_with_list) -> None:
    channel_tuple = tuple(
        DummyChannel(dci_with_list, f"Chan{C}", C)
        for C in ("A", "B", "C", "D", "E", "F")
    )
    channels = ChannelList(
        dci_with_list,
        "TempSensorsTuple",
        DummyChannel,
        channel_tuple,
        snapshotable=False,
    )
    chan_a = channels.ChanA
    assert isinstance(chan_a, DummyChannel)
    with pytest.raises(AttributeError):
        channels.remove(chan_a)


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(setpoints=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
def test_combine_channels(dci, setpoints) -> None:
    assert len(dci.channels) == 6

    mychannels = dci.channels[0:2] + dci.channels[4:]

    assert len(mychannels) == 4
    assert mychannels[0] is dci.A
    assert mychannels[1] is dci.B
    assert mychannels[2] is dci.E
    assert mychannels[3] is dci.F

    for i, chan in enumerate(mychannels):
        chan.temperature(setpoints[i])

    expected = tuple(setpoints[0:2] + [0, 0] + setpoints[2:])
    assert dci.channels.temperature() == expected


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(start=hst.integers(-8, 7), stop=hst.integers(-8, 7), step=hst.integers(1, 7))
def test_access_channels_by_slice(dci, start, stop, step) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, "Chan" + name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)
    if stop < start:
        step = -step
    myslice = slice(start, stop, step)
    mychans = chlist[myslice]
    expected_channels = names[myslice]
    for chan, exp_chan in zip(mychans, expected_channels):
        assert chan.name == f"dci_Chan{exp_chan}"


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=1000)
@given(myindexs=hst.lists(elements=hst.integers(-8, 7), min_size=1))
def test_access_channels_by_tuple(dci, myindexs) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    mytuple = tuple(myindexs)
    channels = tuple(DummyChannel(dci, "Chan" + name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)

    mychans = chlist[mytuple]
    for chan, chanindex in zip(mychans, mytuple):
        assert chan.name == f"dci_Chan{names[chanindex]}"


def test_access_channels_by_name_empty_raises(dci) -> None:
    # todo this should raise a less generic error type
    with pytest.raises(Exception, match="one or more names must be given"):
        dci.channels.get_channel_by_name()


def test_delete_from_channel_list(dci_with_list) -> None:
    n_channels = len(dci_with_list.channels)
    chan0 = dci_with_list.channels[0]
    del dci_with_list.channels[0]
    assert chan0 not in dci_with_list.channels
    assert len(dci_with_list.channels) == n_channels - 1

    with pytest.raises(KeyError):
        dci_with_list.channels.get_channel_by_name(chan0.short_name)

    end_channels = dci_with_list.channels[-2:]
    del dci_with_list.channels[-2:]
    assert len(dci_with_list.channels) == n_channels - 3
    assert all(chan not in dci_with_list.channels for chan in end_channels)

    for chan in end_channels:
        with pytest.raises(KeyError):
            dci_with_list.channels.get_channel_by_name(chan.short_name)

    dci_with_list.channels.lock()
    with pytest.raises(
        AttributeError, match="Cannot delete from a locked channel list"
    ):
        del dci_with_list.channels[0]
    assert len(dci_with_list.channels) == n_channels - 3


def test_set_element_by_int(dci_with_list) -> None:
    dci_with_list.channels[0] = dci_with_list.channels[1]
    assert dci_with_list.channels[0] is dci_with_list.channels[1]


def test_set_element_by_slice(dci_with_list) -> None:
    foo = DummyChannel(dci_with_list, name="foo", channel="foo")
    bar = DummyChannel(dci_with_list, name="bar", channel="bar")
    dci_with_list.channels[0:2] = [foo, bar]
    assert dci_with_list.channels[0] is foo
    assert dci_with_list.channels[1] is bar

    assert (
        dci_with_list.channels.get_channel_by_name("foo") == dci_with_list.channels[0]
    )
    assert (
        dci_with_list.channels.get_channel_by_name("bar") == dci_with_list.channels[1]
    )


def test_set_element_locked_raises(dci_with_list) -> None:
    dci_with_list.channels.lock()

    with pytest.raises(
        AttributeError, match="Cannot set item in a locked channel list"
    ):
        dci_with_list.channels[0] = dci_with_list.channels[1]
    assert dci_with_list.channels[0] is not dci_with_list.channels[1]


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=1000)
@given(myindexs=hst.lists(elements=hst.integers(0, 7), min_size=2))
def test_access_channels_by_name(dci, myindexs) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, "Chan" + name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)

    channel_names = (f"Chan{names[i]}" for i in myindexs)
    mychans = chlist.get_channel_by_name(*channel_names)
    for chan, chanindex in zip(mychans, myindexs):
        assert chan.name == f"dci_Chan{names[chanindex]}"


def test_channels_contain(dci) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, "Chan" + name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)
    for chan in channels:
        assert chan in chlist


def test_channels_reverse(dci) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)
    reverse_names = reversed(names)
    for name, chan in zip(reverse_names, reversed(chlist)):
        assert chan.short_name == name


def test_channels_count(dci) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)

    for channel in channels:
        assert chlist.count(channel) == 1


def test_channels_is_sequence(dci) -> None:
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)

    assert isinstance(chlist, Sequence)
    assert issubclass(ChannelList, Sequence)


def test_names(dci) -> None:
    ex_inst_name = "dci"
    for channel in dci.channels:
        sub_channel = DummyChannel(channel, "subchannel", "subchannel")
        channel.add_submodule("somesubchannel", sub_channel)
    assert dci.name == ex_inst_name
    assert dci.full_name == ex_inst_name
    assert dci.short_name == ex_inst_name
    assert dci.name_parts == [ex_inst_name]

    # Parameters directly on instrument
    assert dci.IDN.name == "IDN"
    assert dci.IDN.full_name == f"{ex_inst_name}_IDN"
    for chan, name in zip(dci.channels, ["A", "B", "C", "D", "E", "F"]):
        ex_chan_name = f"Chan{name}"
        ex_chan_full_name = f"{ex_inst_name}_{ex_chan_name}"

        assert chan.short_name == ex_chan_name
        assert chan.name == ex_chan_full_name
        assert chan.full_name == ex_chan_full_name
        assert chan.name_parts == [ex_inst_name, ex_chan_name]

        ex_param_name = "temperature"
        assert chan.temperature.name == ex_param_name
        assert chan.temperature.full_name == f"{ex_chan_full_name}_{ex_param_name}"
        assert chan.temperature.short_name == ex_param_name
        assert chan.temperature.name_parts == [
            ex_inst_name,
            ex_chan_name,
            ex_param_name,
        ]

        ex_subchan_name = "subchannel"
        ex_subchan_full_name = f"{ex_chan_full_name}_{ex_subchan_name}"

        assert chan.somesubchannel.short_name == ex_subchan_name
        assert chan.somesubchannel.name == ex_subchan_full_name
        assert chan.somesubchannel.full_name == ex_subchan_full_name
        assert chan.somesubchannel.name_parts == [
            ex_inst_name,
            ex_chan_name,
            ex_subchan_name,
        ]

        assert chan.somesubchannel.temperature.name == ex_param_name
        assert (
            chan.somesubchannel.temperature.full_name
            == f"{ex_subchan_full_name}_{ex_param_name}"
        )
        assert chan.somesubchannel.temperature.short_name == ex_param_name
        assert chan.somesubchannel.temperature.name_parts == [
            ex_inst_name,
            ex_chan_name,
            ex_subchan_name,
            ex_param_name,
        ]


def test_root_instrument(dci) -> None:
    assert dci.root_instrument is dci
    for channel in dci.channels:
        assert channel.root_instrument is dci
        for parameter in channel.parameters.values():
            assert parameter.root_instrument is dci


def test_get_attr_on_empty_channellist_works_as_expected(empty_instrument) -> None:
    channels = ChannelTuple(empty_instrument, "channels", chan_type=DummyChannel)
    empty_instrument.add_submodule("channels", channels)

    with pytest.raises(
        AttributeError, match="'ChannelTuple' object has no attribute 'temperature'"
    ):
        _ = empty_instrument.channels.temperature


def test_channel_tuple_call_method_basic_test(dci) -> None:
    result = dci.channels.turn_on()
    assert result is None


def test_channel_tuple_call_method_called_as_expected(dci, mocker) -> None:
    for channel in dci.channels:
        channel.turn_on = mocker.MagicMock(return_value=1)

    result = dci.channels.turn_on("bar")
    # We never return the result (same for Function)
    assert result is None
    for channel in dci.channels:
        channel.turn_on.assert_called_with("bar")


def test_channel_tuple_names(dci: DummyChannelInstrument) -> None:
    assert dci.channels.short_name == "TempSensors"
    assert dci.channels.full_name == "dci_TempSensors"


def _verify_multiparam_data(data):
    assert "multi_setpoint_param_this_setpoint_set" in data.arrays.keys()
    assert_array_equal(
        data.arrays["multi_setpoint_param_this_setpoint_set"].ndarray,
        np.repeat(np.arange(5.0, 10).reshape(1, 5), 11, axis=0),
    )
    assert "dci_ChanA_multi_setpoint_param_this" in data.arrays.keys()
    assert_array_equal(
        data.arrays["dci_ChanA_multi_setpoint_param_this"].ndarray, np.zeros((11, 5))
    )
    assert "dci_ChanA_multi_setpoint_param_this" in data.arrays.keys()
    assert_array_equal(
        data.arrays["dci_ChanA_multi_setpoint_param_that"].ndarray, np.ones((11, 5))
    )
    assert "dci_ChanA_temperature_set" in data.arrays.keys()
    assert_array_equal(
        data.arrays["dci_ChanA_temperature_set"].ndarray, np.arange(0, 10.1, 1)
    )


def _verify_array_data(data, channels=("A",)):
    assert "array_setpoint_param_this_setpoint_set" in data.arrays.keys()
    assert_array_equal(
        data.arrays["array_setpoint_param_this_setpoint_set"].ndarray,
        np.repeat(np.arange(5.0, 10).reshape(1, 5), 11, axis=0),
    )
    for channel in channels:
        aname = f"dci_Chan{channel}_dummy_array_parameter"
        assert aname in data.arrays.keys()
        assert_array_equal(data.arrays[aname].ndarray, np.ones((11, 5)) + 1)
    assert "dci_ChanA_temperature_set" in data.arrays.keys()
    assert_array_equal(
        data.arrays["dci_ChanA_temperature_set"].ndarray, np.arange(0, 10.1, 1)
    )
