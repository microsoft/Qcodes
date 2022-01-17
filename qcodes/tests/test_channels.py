import logging

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose, assert_array_equal

from qcodes.data.location import FormatLocation
from qcodes.instrument.channel import ChannelList
from qcodes.instrument.parameter import Parameter
from qcodes.loops import Loop
from qcodes.tests.instrument_mocks import DummyChannel, DummyChannelInstrument
from qcodes.utils.validators import Numbers


@pytest.fixture(scope='function', name='dci')
def _make_dci():

    dci = DummyChannelInstrument(name='dci')
    try:
        yield dci
    finally:
        dci.close()


def test_channels_call_function(dci, caplog):
    """
    Test that dci.channels.some_function() calls
    some_function on each of the channels
    """
    with caplog.at_level(logging.DEBUG,
                         logger='qcodes.tests.instrument_mocks'):
        caplog.clear()
        dci.channels.log_my_name()
        mssgs = [rec.message for rec in caplog.records]
        names = [ch.name.replace('dci_', '') for ch in dci.channels]
        assert mssgs == names


def test_channels_get(dci):

    temperatures = dci.channels.temperature.get()
    assert len(temperatures) == 6


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(value=hst.floats(0, 300), channel=hst.integers(0, 3))
def test_channel_access_is_identical(dci, value, channel):
    channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
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


def test_add_channel(dci):
    n_channels = len(dci.channels)
    name = 'foo'
    channel = DummyChannel(dci, 'Chan'+name, name)
    dci.channels.append(channel)
    dci.add_submodule(name, channel)

    assert len(dci.channels) == n_channels+1

    dci.channels.lock()
    # after locking the channels it's not possible to add any more channels
    with pytest.raises(AttributeError):
        name = 'bar'
        channel = DummyChannel(dci, 'Chan' + name, name)
        dci.channels.append(channel)
    assert len(dci.channels) == n_channels + 1


def test_add_channels_from_generator(dci):
    n_channels = len(dci.channels)
    names = ('foo', 'bar', 'foobar')
    channels = (DummyChannel(dci, 'Chan'+name, name)
                for name in names)
    dci.channels.extend(channels)

    assert len(dci.channels) == n_channels + len(names)


def test_add_channels_from_tuple(dci):
    n_channels = len(dci.channels)
    names = ('foo', 'bar', 'foobar')
    channels = tuple(DummyChannel(dci, 'Chan'+name, name)
                     for name in names)
    dci.channels.extend(channels)

    assert len(dci.channels) == n_channels + len(names)


def test_extend_then_remove(dci):
    n_channels = len(dci.channels)
    names = ('foo', 'bar', 'foobar')
    channels = [DummyChannel(dci, 'Chan' + name, name)
                for name in names]
    dci.channels.extend(channels)

    assert len(dci.channels) == n_channels + len(names)
    last_channel = dci.channels[-1]
    dci.channels.remove(last_channel)
    assert last_channel not in dci.channels
    assert len(dci.channels) == n_channels + len(names) - 1


def test_insert_channel(dci):
    n_channels = len(dci.channels)
    name = 'foo'
    channel = DummyChannel(dci, 'Chan'+name, name)
    dci.channels.insert(1, channel)
    dci.add_submodule(name, channel)

    assert len(dci.channels) == n_channels+1
    assert dci.channels[1] is channel
    dci.channels.lock()
    # after locking the channels it's not possible to add any more channels
    with pytest.raises(AttributeError):
        name = 'bar'
        channel = DummyChannel(dci, 'Chan' + name, name)
        dci.channels.insert(2, channel)
    assert len(dci.channels) == n_channels + 1


def test_clear_channels(dci):
    channels = dci.channels
    channels.clear()
    assert len(channels) == 0


def test_clear_locked_channels(dci):
    channels = dci.channels
    original_length = len(channels)
    channels.lock()
    with pytest.raises(AttributeError):
        channels.clear()
    assert len(channels) == original_length


def test_remove_channel(dci):
    channels = dci.channels
    chan_a = dci.A
    original_length = len(channels.temperature())
    channels.remove(chan_a)
    with pytest.raises(AttributeError):
        getattr(channels, chan_a.short_name)
    assert len(channels) == original_length-1
    assert len(channels.temperature()) == original_length-1


def test_remove_locked_channel(dci):
    channels = dci.channels
    chan_a = dci.A
    channels.lock()
    with pytest.raises(AttributeError):
        channels.remove(chan_a)


def test_remove_tupled_channel(dci):
    channel_tuple = tuple(
        DummyChannel(dci, f'Chan{C}', C)
        for C in ('A', 'B', 'C', 'D', 'E', 'F')
    )
    channels = ChannelList(dci,
                           "TempSensorsTuple",
                           DummyChannel,
                           channel_tuple,
                           snapshotable=False)
    chan_a = channels.ChanA
    with pytest.raises(AttributeError):
        channels.remove(chan_a)


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(setpoints=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
def test_combine_channels(dci, setpoints):
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
@given(start=hst.integers(-8, 7), stop=hst.integers(-8, 7),
       step=hst.integers(1, 7))
def test_access_channels_by_slice(dci, start, stop, step):
    names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    channels = tuple(DummyChannel(dci,
                                  'Chan'+name, name) for name in names)
    chlist = ChannelList(dci, 'channels',
                         DummyChannel, channels)
    if stop < start:
        step = -step
    myslice = slice(start, stop, step)
    mychans = chlist[myslice]
    expected_channels = names[myslice]
    for chan, exp_chan in zip(mychans, expected_channels):
        assert chan.name == f'dci_Chan{exp_chan}'


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(myindexs=hst.lists(elements=hst.integers(-8, 7), min_size=1))
def test_access_channels_by_tuple(dci, myindexs):
    names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
    mytuple = tuple(myindexs)
    channels = tuple(DummyChannel(dci,
                                  'Chan'+name, name) for name in names)
    chlist = ChannelList(dci, 'channels',
                         DummyChannel, channels)

    mychans = chlist[mytuple]
    for chan, chanindex in zip(mychans, mytuple):
        assert chan.name == f"dci_Chan{names[chanindex]}"


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(myindexs=hst.lists(elements=hst.integers(0, 7), min_size=2))
def test_access_channels_by_name(dci, myindexs):
    names = ("A", "B", "C", "D", "E", "F", "G", "H")
    channels = tuple(DummyChannel(dci, "Chan" + name, name) for name in names)
    chlist = ChannelList(dci, "channels", DummyChannel, channels)

    channel_names = (f"Chan{names[i]}" for i in myindexs)
    mychans = chlist.get_channel_by_name(*channel_names)
    for chan, chanindex in zip(mychans, myindexs):
        assert chan.name == f'dci_Chan{names[chanindex]}'


def test_names(dci):
    ex_inst_name = 'dci'
    for channel in dci.channels:
        sub_channel = DummyChannel(channel, 'subchannel', 'subchannel')
        channel.add_submodule('somesubchannel', sub_channel)
    assert dci.name == ex_inst_name
    assert dci.full_name == ex_inst_name
    assert dci.short_name == ex_inst_name
    assert dci.name_parts == [ex_inst_name]

    # Parameters directly on instrument
    assert dci.IDN.name == 'IDN'
    assert dci.IDN.full_name == f"{ex_inst_name}_IDN"
    for chan, name in zip(dci.channels,
                          ['A', 'B', 'C', 'D', 'E', 'F']):
        ex_chan_name = f"Chan{name}"
        ex_chan_full_name = f"{ex_inst_name}_{ex_chan_name}"

        assert chan.short_name == ex_chan_name
        assert chan.name == ex_chan_full_name
        assert chan.full_name == ex_chan_full_name
        assert chan.name_parts == [ex_inst_name, ex_chan_name]

        ex_param_name = 'temperature'
        assert chan.temperature.name == ex_param_name
        assert chan.temperature.full_name ==\
               f'{ex_chan_full_name}_{ex_param_name}'
        assert chan.temperature.short_name == ex_param_name
        assert chan.temperature.name_parts == [ex_inst_name, ex_chan_name,
                                               ex_param_name]

        ex_subchan_name = f"subchannel"
        ex_subchan_full_name = f"{ex_chan_full_name}_{ex_subchan_name}"

        assert chan.somesubchannel.short_name == ex_subchan_name
        assert chan.somesubchannel.name == ex_subchan_full_name
        assert chan.somesubchannel.full_name == ex_subchan_full_name
        assert chan.somesubchannel.name_parts == [ex_inst_name,
                                                  ex_chan_name,
                                                  ex_subchan_name]

        assert chan.somesubchannel.temperature.name == ex_param_name
        assert chan.somesubchannel.temperature.full_name ==\
               f'{ex_subchan_full_name}_{ex_param_name}'
        assert chan.somesubchannel.temperature.short_name == ex_param_name
        assert chan.somesubchannel.temperature.name_parts == \
               [ex_inst_name, ex_chan_name, ex_subchan_name, ex_param_name]


def test_loop_simple(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopSimple'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 300, 10),
                0.001).each(dci.A.temperature)
    data = loop.run(location=loc_provider)
    assert_array_equal(data.dci_ChanA_temperature_set.ndarray,
                       data.dci_ChanA_temperature.ndarray)


def test_loop_measure_all_channels(dci):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'allChannels'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).\
        each(dci.channels.temperature)
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    assert len(data.arrays) == 7
    for chan in ['A', 'B', 'C', 'D', 'E', 'F']:
        assert getattr(
            data,
            f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)


def test_loop_measure_channels_individually(dci):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'channelsIndividually'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(dci.
                                                 channels[0].temperature,
                                                 dci.
                                                 channels[1].temperature,
                                                 dci.
                                                 channels[2].temperature,
                                                 dci.
                                                 channels[3].temperature)
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    for chan in ['A', 'B', 'C', 'D']:
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)


@given(values=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
@settings(max_examples=10, deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_loop_measure_channels_by_name(dci, values):
    p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None,
                   set_cmd=None)
    for i in range(4):
        dci.channels[i].temperature(values[i])
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'channelsByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(
        dci.A.temperature,
        dci.B.temperature,
        dci.C.temperature,
        dci.D.temperature
    )
    data = loop.run(location=loc_provider)
    assert data.p1_set.ndarray.shape == (21, )
    for i, chan in enumerate(['A', 'B', 'C', 'D']):
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.shape == (21,)
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.max() == values[i]
        assert getattr(
            data, f'dci_Chan{chan}_temperature'
        ).ndarray.min() == values[i]


@given(loop_channels=hst.lists(hst.integers(0, 3), min_size=2, max_size=2,
                               unique=True),
       measure_channel=hst.integers(0, 3))
@settings(max_examples=10, deadline=800,
          suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_nested_loop_over_channels(dci, loop_channels, measure_channel):
    channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'nestedLoopOverChannels'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[loop_channels[0]].temperature.
                sweep(0, 10, 0.5))
    loop = loop.loop(dci.channels[loop_channels[1]].temperature.
                     sweep(50, 51, 0.1))
    loop = loop.each(dci.channels[measure_channel].temperature)
    data = loop.run(location=loc_provider)

    assert getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[0]]}_temperature_set'
    ).ndarray.shape == (21,)
    assert getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[1]]}_temperature_set'
    ).ndarray.shape == (21, 11,)
    assert getattr(
        data,
        f'dci_Chan{channel_to_label[measure_channel]}_temperature'
    ).ndarray.shape == (21, 11)

    assert_array_equal(getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[0]]}_temperature_set'
    ).ndarray, np.arange(0, 10.1, 0.5))

    expected_array = np.repeat(np.arange(50, 51.01, 0.1).reshape(1, 11),
                               21, axis=0)
    array = getattr(
        data,
        f'dci_Chan{channel_to_label[loop_channels[1]]}_temperature_set'
    ).ndarray
    assert_allclose(array, expected_array)


def test_loop_slicing_multiparameter_raises(dci):
    with pytest.raises(NotImplementedError):
        loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
        loop.each(dci.channels[0:2].dummy_multi_parameter).run()


def test_loop_multiparameter_by_name(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'multiParamByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.A.dummy_multi_parameter)\
        .run(location=loc_provider)
    _verify_multiparam_data(data)
    assert 'multi_setpoint_param_this_setpoint_set' in data.arrays.keys()


def test_loop_multiparameter_by_index(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopByIndex'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 10, 1),
                0.1)
    data = loop.each(dci.A.dummy_multi_parameter)\
        .run(location=loc_provider)
    _verify_multiparam_data(data)


def test_loop_slicing_arrayparameter(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'loopSlicing'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.channels[0:2].dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data, channels=('A', 'B'))


def test_loop_arrayparameter_by_name(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'arrayParamByName'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.A.temperature.sweep(0, 10, 1), 0.1)
    data = loop.each(dci.A.dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data)


def test_loop_arrayparameter_by_index(dci):
    loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
    rcd = {'name': 'arrayParamByIndex'}
    loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
    loop = Loop(dci.channels[0].temperature.sweep(0, 10, 1),
                0.1)
    data = loop.each(dci.A.dummy_array_parameter)\
        .run(location=loc_provider)
    _verify_array_data(data)


def test_root_instrument(dci):
    assert dci.root_instrument is dci
    for channel in dci.channels:
        assert channel.root_instrument is dci
        for parameter in channel.parameters.values():
            assert parameter.root_instrument is dci


def _verify_multiparam_data(data):
    assert 'multi_setpoint_param_this_setpoint_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['multi_setpoint_param_this_setpoint_set'].ndarray,
        np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0)
    )
    assert 'dci_ChanA_multi_setpoint_param_this' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_multi_setpoint_param_this'].ndarray,
        np.zeros((11, 5))
    )
    assert 'dci_ChanA_multi_setpoint_param_this' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_multi_setpoint_param_that'].ndarray,
        np.ones((11, 5))
    )
    assert 'dci_ChanA_temperature_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_temperature_set'].ndarray,
        np.arange(0, 10.1, 1)
    )


def _verify_array_data(data, channels=('A',)):
    assert 'array_setpoint_param_this_setpoint_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['array_setpoint_param_this_setpoint_set'].ndarray,
        np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0)
    )
    for channel in channels:
        aname = f'dci_Chan{channel}_dummy_array_parameter'
        assert aname in data.arrays.keys()
        assert_array_equal(data.arrays[aname].ndarray, np.ones((11, 5))+1)
    assert 'dci_ChanA_temperature_set' in data.arrays.keys()
    assert_array_equal(
        data.arrays['dci_ChanA_temperature_set'].ndarray,
        np.arange(0, 10.1, 1)
    )
