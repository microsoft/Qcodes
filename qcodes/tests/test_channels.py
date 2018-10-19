import logging

from unittest import TestCase
import unittest
from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from qcodes.tests.instrument_mocks import DummyChannelInstrument, DummyChannel
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import ChannelList
from qcodes.loops import Loop


@pytest.fixture(scope='function')
def dci():

    dci = DummyChannelInstrument(name='dci')
    yield dci
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


class TestChannels(TestCase):

    def setUp(self):
        # print("setup")
        self.instrument = DummyChannelInstrument(name='testchanneldummy')

    def tearDown(self):

        self.instrument.close()
        del self.instrument
        # del self.instrument is not sufficient in general because the __del__ method is
        # first invoked when there are 0 (non weak) references to the instrument. If a test
        # fails the unittest framework will keep a reference to the instrument is removed from
        # the testcase and __del__ is not invoked until all the tests have run.

    def test_channels_get(self):

        temperatures = self.instrument.channels.temperature.get()
        self.assertEqual(len(temperatures), 6)

    @given(value=hst.floats(0, 300), channel=hst.integers(0, 3))
    def test_channel_access_is_identical(self, value, channel):
        channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
        label = channel_to_label[channel]
        channel_via_label = getattr(self.instrument, label)
        # set via labeled channel
        channel_via_label.temperature(value)
        self.assertEqual(channel_via_label.temperature(), value)
        self.assertEqual(self.instrument.channels[channel].temperature(), value)
        self.assertEqual(self.instrument.channels.temperature()[channel], value)
        # reset
        channel_via_label.temperature(0)
        self.assertEqual(channel_via_label.temperature(), 0)
        self.assertEqual(self.instrument.channels[channel].temperature(), 0)
        self.assertEqual(self.instrument.channels.temperature()[channel], 0)
        # set via index into list
        self.instrument.channels[channel].temperature(value)
        self.assertEqual(channel_via_label.temperature(), value)
        self.assertEqual(self.instrument.channels[channel].temperature(), value)
        self.assertEqual(self.instrument.channels.temperature()[channel], value)
        # it's not possible to set via self.instrument.channels.temperature as this is a multi parameter
        # that currently does not support set.

    def test_add_channel(self):
        n_channels = len(self.instrument.channels)
        name = 'foo'
        channel = DummyChannel(self.instrument, 'Chan'+name, name)
        self.instrument.channels.append(channel)
        self.instrument.add_submodule(name, channel)

        self.assertEqual(len(self.instrument.channels), n_channels+1)

        self.instrument.channels.lock()
        # after locking the channels it's not possible to add any more channels
        with self.assertRaises(AttributeError):
            name = 'bar'
            channel = DummyChannel(self.instrument, 'Chan' + name, name)
            self.instrument.channels.append(channel)
        self.assertEqual(len(self.instrument.channels), n_channels + 1)

    def test_add_channels_from_generator(self):
        n_channels = len(self.instrument.channels)
        names = ('foo', 'bar', 'foobar')
        channels = (DummyChannel(self.instrument, 'Chan'+name, name) for name in names)
        self.instrument.channels.extend(channels)

        self.assertEqual(len(self.instrument.channels), n_channels + len(names))

    def test_add_channels_from_tuple(self):
        n_channels = len(self.instrument.channels)
        names = ('foo', 'bar', 'foobar')
        channels = tuple(DummyChannel(self.instrument, 'Chan'+name, name) for name in names)
        self.instrument.channels.extend(channels)

        self.assertEqual(len(self.instrument.channels), n_channels + len(names))

    def test_insert_channel(self):
        n_channels = len(self.instrument.channels)
        name = 'foo'
        channel = DummyChannel(self.instrument, 'Chan'+name, name)
        self.instrument.channels.insert(1, channel)
        self.instrument.add_submodule(name, channel)

        self.assertEqual(len(self.instrument.channels), n_channels+1)
        self.assertIs(self.instrument.channels[1], channel)
        self.instrument.channels.lock()
        # after locking the channels it's not possible to add any more channels
        with self.assertRaises(AttributeError):
            name = 'bar'
            channel = DummyChannel(self.instrument, 'Chan' + name, name)
            self.instrument.channels.insert(2, channel)
        self.assertEqual(len(self.instrument.channels), n_channels + 1)

    def test_clear_channels(self):
        channels = self.instrument.channels
        channels.clear()
        self.assertEqual(len(channels), 0)

    def test_clear_locked_channels(self):
        channels = self.instrument.channels
        original_length = len(channels)
        channels.lock()
        with self.assertRaises(AttributeError):
            channels.clear()
        self.assertEqual(len(channels), original_length)

    def test_remove_channel(self):
        channels = self.instrument.channels
        chanA = self.instrument.A
        original_length = len(channels.temperature())
        channels.remove(chanA)
        with self.assertRaises(AttributeError):
            getattr(channels, chanA.short_name)
        self.assertEqual(len(channels), original_length-1)
        self.assertEqual(len(channels.temperature()), original_length-1)

    def test_remove_locked_channel(self):
        channels = self.instrument.channels
        chanA = self.instrument.A
        channels.lock()
        with self.assertRaises(AttributeError):
            channels.remove(chanA)

    def test_remove_tupled_channel(self):
        channel_tuple = tuple(
            DummyChannel(self.instrument, f'Chan{C}', C)
            for C in ('A', 'B', 'C', 'D', 'E', 'F')
        )
        channels = ChannelList(self.instrument,
                               "TempSensorsTuple",
                               DummyChannel,
                               channel_tuple,
                               snapshotable=False)
        chanA = channels.ChanA
        with self.assertRaises(AttributeError):
            channels.remove(chanA)

    @given(setpoints=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
    def test_combine_channels(self, setpoints):
        self.assertEqual(len(self.instrument.channels), 6)

        mychannels = self.instrument.channels[0:2] + self.instrument.channels[4:]

        self.assertEqual(len(mychannels), 4)
        self.assertIs(mychannels[0], self.instrument.A)
        self.assertIs(mychannels[1], self.instrument.B)
        self.assertIs(mychannels[2], self.instrument.E)
        self.assertIs(mychannels[3], self.instrument.F)

        for i, chan in enumerate(mychannels):
            chan.temperature(setpoints[i])

        expected = tuple(setpoints[0:2] + [0, 0] + setpoints[2:])
        self.assertEqual(self.instrument.channels.temperature(), expected)

    @given(start=hst.integers(-8,7), stop=hst.integers(-8,7), step=hst.integers(1,7))
    def test_access_channels_by_slice(self, start, stop, step):
        names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
        channels = tuple(DummyChannel(self.instrument,
                                      'Chan'+name, name) for name in names)
        chlist = ChannelList(self.instrument, 'channels',
                             DummyChannel, channels)
        if stop < start:
            step = -step
        myslice = slice(start, stop, step)
        mychans = chlist[myslice]
        expected_channels = names[myslice]
        for chan, exp_chan in zip(mychans, expected_channels):
            assert chan.name == f'testchanneldummy_Chan{exp_chan}'


    @given(myindexs=hst.lists(elements=hst.integers(-8,7), min_size=1))
    def test_access_channels_by_tuple(self, myindexs):
        names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
        mytuple = tuple(myindexs)
        channels = tuple(DummyChannel(self.instrument,
                                      'Chan'+name, name) for name in names)
        chlist = ChannelList(self.instrument, 'channels',
                             DummyChannel, channels)

        mychans = chlist[mytuple]
        for chan, chanindex in zip(mychans, mytuple):
            assert chan.name == f'testchanneldummy_Chan{names[chanindex]}'


    def test_names(self):
        ex_inst_name = 'testchanneldummy'
        for channel in self.instrument.channels:
            sub_channel = DummyChannel(channel, 'subchannel', 'subchannel')
            channel.add_submodule('somesubchannel', sub_channel)
        assert self.instrument.name == ex_inst_name
        assert self.instrument.full_name == ex_inst_name
        assert self.instrument.short_name == ex_inst_name
        assert self.instrument.name_parts == [ex_inst_name]

        # Parameters directly on instrument
        assert self.instrument.IDN.name == 'IDN'
        assert self.instrument.IDN.full_name == f"{ex_inst_name}_IDN"
        for chan, name in zip(self.instrument.channels,
                              ['A', 'B', 'C', 'D', 'E', 'F']):
            ex_chan_name = f"Chan{name}"
            ex_chan_full_name = f"{ex_inst_name}_{ex_chan_name}"

            assert chan.short_name == ex_chan_name
            assert chan.name == ex_chan_full_name
            assert chan.full_name == ex_chan_full_name
            assert chan.name_parts == [ex_inst_name, ex_chan_name]

            ex_param_name = 'temperature'
            assert chan.temperature.name == ex_param_name
            assert chan.temperature.full_name == f'{ex_chan_full_name}_{ex_param_name}'
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
            assert chan.somesubchannel.temperature.full_name == f'{ex_subchan_full_name}_{ex_param_name}'
            assert chan.somesubchannel.temperature.short_name == ex_param_name
            assert chan.somesubchannel.temperature.name_parts == [ex_inst_name,
                                                                  ex_chan_name,
                                                                  ex_subchan_name,
                                                                  ex_param_name]
class TestChannelsLoop(TestCase):

    def setUp(self):
        self.instrument = DummyChannelInstrument(name='testchanneldummy')

    def tearDown(self):
        self.instrument.close()
        del self.instrument

    def test_loop_simple(self):
        loop = Loop(self.instrument.channels[0].temperature.sweep(0, 300, 10),
                    0.001).each(self.instrument.A.temperature)
        data = loop.run()
        assert_array_equal(data.testchanneldummy_ChanA_temperature_set.ndarray,
                           data.testchanneldummy_ChanA_temperature.ndarray)

    def test_loop_measure_all_channels(self):
        p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None, set_cmd=None)
        loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(self.instrument.channels.temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        self.assertEqual(len(data.arrays), 7)
        for chan in ['A', 'B', 'C', 'D', 'E', 'F']:
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))

    def test_loop_measure_channels_individually(self):
        p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None, set_cmd=None)
        loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(self.instrument.channels[0].temperature,
                                                     self.instrument.channels[1].temperature,
                                                     self.instrument.channels[2].temperature,
                                                     self.instrument.channels[3].temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        for chan in ['A', 'B', 'C', 'D']:
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))

    @given(values=hst.lists(hst.floats(0, 300), min_size=4, max_size=4))
    @settings(max_examples=10, deadline=None)
    def test_loop_measure_channels_by_name(self, values):
        p1 = Parameter(name='p1', vals=Numbers(-10, 10), get_cmd=None, set_cmd=None)
        for i in range(4):
            self.instrument.channels[i].temperature(values[i])
        loop = Loop(p1.sweep(-10, 10, 1), 1e-6).each(self.instrument.A.temperature,
                                                     self.instrument.B.temperature,
                                                     self.instrument.C.temperature,
                                                     self.instrument.D.temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        for i, chan in enumerate(['A', 'B', 'C', 'D']):
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.max(), values[i])
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.min(), values[i])

    @given(loop_channels=hst.lists(hst.integers(0, 3), min_size=2, max_size=2, unique=True),
           measure_channel=hst.integers(0, 3))
    @settings(max_examples=10, deadline=800)
    def test_nested_loop_over_channels(self, loop_channels, measure_channel):
        channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
        loop = Loop(self.instrument.channels[loop_channels[0]].temperature.sweep(0, 10, 0.5))
        loop = loop.loop(self.instrument.channels[loop_channels[1]].temperature.sweep(50, 51, 0.1))
        loop = loop.each(self.instrument.channels[measure_channel].temperature)
        data = loop.run()

        self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature_set'.format(
            channel_to_label[loop_channels[0]])).ndarray.shape, (21,))
        self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature_set'.format(
            channel_to_label[loop_channels[1]])).ndarray.shape, (21, 11,))
        self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(
            channel_to_label[measure_channel])).ndarray.shape, (21, 11))

        assert_array_equal(getattr(data, 'testchanneldummy_Chan{}_temperature_set'.format(
            channel_to_label[loop_channels[0]])).ndarray,
                           np.arange(0, 10.1, 0.5))

        expected_array = np.repeat(np.arange(50, 51.01, 0.1).reshape(1, 11), 21, axis=0)
        array = getattr(data, 'testchanneldummy_Chan'
                              '{}_temperature_set'.format(channel_to_label[loop_channels[1]])).ndarray
        assert_allclose(array, expected_array)

    def test_loop_slicing_multiparameter_raises(self):
        with self.assertRaises(NotImplementedError):
            loop = Loop(self.instrument.A.temperature.sweep(0, 10, 1), 0.1)
            loop.each(self.instrument.channels[0:2].dummy_multi_parameter).run()

    def test_loop_multiparameter_by_name(self):
        loop = Loop(self.instrument.A.temperature.sweep(0, 10, 1), 0.1)
        data = loop.each(self.instrument.A.dummy_multi_parameter).run()
        self._verify_multiparam_data(data)
        self.assertIn('this_setpoint_set', data.arrays.keys())

    def test_loop_multiparameter_by_index(self):
        loop = Loop(self.instrument.channels[0].temperature.sweep(0, 10, 1), 0.1)
        data = loop.each(self.instrument.A.dummy_multi_parameter).run()
        self._verify_multiparam_data(data)

    def _verify_multiparam_data(self, data):
        self.assertIn('this_setpoint_set', data.arrays.keys())
        assert_array_equal(data.arrays['this_setpoint_set'].ndarray,
                           np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0))
        self.assertIn('testchanneldummy_ChanA_this', data.arrays.keys())
        assert_array_equal(data.arrays['testchanneldummy_ChanA_this'].ndarray, np.zeros((11, 5)))
        self.assertIn('testchanneldummy_ChanA_that', data.arrays.keys())
        assert_array_equal(data.arrays['testchanneldummy_ChanA_that'].ndarray, np.ones((11, 5)))
        self.assertIn('testchanneldummy_ChanA_temperature_set', data.arrays.keys())
        assert_array_equal(data.arrays['testchanneldummy_ChanA_temperature_set'].ndarray, np.arange(0, 10.1, 1))

    def test_loop_slicing_arrayparameter(self):
        loop = Loop(self.instrument.A.temperature.sweep(0, 10, 1), 0.1)
        data = loop.each(self.instrument.channels[0:2].dummy_array_parameter).run()
        self._verify_array_data(data, channels=('A', 'B'))

    def test_loop_arrayparameter_by_name(self):
        loop = Loop(self.instrument.A.temperature.sweep(0, 10, 1), 0.1)
        data = loop.each(self.instrument.A.dummy_array_parameter).run()
        self._verify_array_data(data)

    def test_loop_arrayparameter_by_index(self):
        loop = Loop(self.instrument.channels[0].temperature.sweep(0, 10, 1), 0.1)
        data = loop.each(self.instrument.A.dummy_array_parameter).run()
        self._verify_array_data(data)

    def _verify_array_data(self, data, channels=('A',)):
        self.assertIn('this_setpoint_set', data.arrays.keys())
        assert_array_equal(data.arrays['this_setpoint_set'].ndarray,
                           np.repeat(np.arange(5., 10).reshape(1, 5), 11, axis=0))
        for channel in channels:
            aname = 'testchanneldummy_Chan{}_dummy_array_parameter'.format(channel)
            self.assertIn(aname, data.arrays.keys())
            assert_array_equal(data.arrays[aname].ndarray, np.ones((11, 5))+1)
        self.assertIn('testchanneldummy_ChanA_temperature_set', data.arrays.keys())
        assert_array_equal(data.arrays['testchanneldummy_ChanA_temperature_set'].ndarray, np.arange(0, 10.1, 1))

    def test_root_instrument(self):
        assert self.instrument.root_instrument is self.instrument
        for channel in self.instrument.channels:
            assert channel.root_instrument is self.instrument
            for parameter in channel.parameters.values():
                assert parameter.root_instrument is self.instrument

if __name__ == '__main__':
    unittest.main()
