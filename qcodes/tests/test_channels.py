from unittest import TestCase
import unittest

from qcodes.tests.instrument_mocks import DummyChannelInstrument, DummyChannel
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import ManualParameter

from hypothesis import given, settings
import hypothesis.strategies as hst
from hypothesis.strategies import floats, integers
from qcodes.loops import Loop


from numpy.testing import assert_array_equal

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
        self.assertEqual(len(temperatures), 4)

    @given(value=floats(0, 300), channel=integers(0, 3))
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
        name = 'foo'
        channel = DummyChannel(self.instrument, 'Chan'+name, name)
        self.instrument.channels.append(channel)
        self.instrument.add_submodule(name, channel)

        self.assertEqual(len(self.instrument.channels), 5)

        self.instrument.channels.lock()
        # after locking the channels it's not possible to add any more channels
        with self.assertRaises(AttributeError):
            name = 'bar'
            channel = DummyChannel(self.instrument, 'Chan' + name, name)
            self.instrument.channels.append(channel)
            self.instrument.add_submodule(name, channel)



class TestChannelsLoop(TestCase):
    pass

    def setUp(self):
        self.instrument = DummyChannelInstrument(name='testchanneldummy')

    def tearDown(self):
        self.instrument.close()
        del self.instrument

    def test_loop_simple(self):
        loop = Loop(self.instrument.channels[0].temperature.sweep(0,300, 10), 0.001).each(self.instrument.A.temperature)
        data = loop.run()
        assert_array_equal(data.testchanneldummy_ChanA_temperature_set.ndarray,
                           data.testchanneldummy_ChanA_temperature.ndarray)

    def test_loop_measure_all_channels(self):
        p1 = ManualParameter(name='p1', vals=Numbers(-10, 10))
        loop = Loop(p1.sweep(-10,10,1), 1e-6).each(self.instrument.channels.temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        for i,chan in enumerate(['A', 'B', 'C', 'D']):
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))


    def test_loop_measure_channels_individually(self):
        p1 = ManualParameter(name='p1', vals=Numbers(-10, 10))
        loop = Loop(p1.sweep(-10,10,1), 1e-6).each(self.instrument.channels[0].temperature,
                                                   self.instrument.channels[1].temperature,
                                                   self.instrument.channels[2].temperature,
                                                   self.instrument.channels[3].temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        for i,chan in enumerate(['A', 'B', 'C', 'D']):
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))

    @given(values=hst.tuples(floats(0, 300), floats(0, 300), floats(0, 300), floats(0, 300)))
    @settings(max_examples=10)
    def test_loop_measure_channels_by_name(self, values):
        p1 = ManualParameter(name='p1', vals=Numbers(-10, 10))
        for i in range(4):
            self.instrument.channels[i].temperature(values[i])
        loop = Loop(p1.sweep(-10,10,1), 1e-6).each(self.instrument.A.temperature,
                                                      self.instrument.B.temperature,
                                                      self.instrument.C.temperature,
                                                      self.instrument.D.temperature)
        data = loop.run()
        self.assertEqual(data.p1_set.ndarray.shape, (21, ))
        for i,chan in enumerate(['A', 'B', 'C', 'D']):
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))
            self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.max(), values[i])
    #
    # @given(loop_channel=hst.lists(integers(0, 3), measure_channel=integers(0, 3))
    # @settings(max_examples=10)
    def test_nested_loop_over_channels(self, loop_channels=(0,1), measure_channel=2):
        channel_to_label = {0: 'A', 1: 'B', 2: 'C', 3: "D"}
        loop = Loop(self.instrument.channels[loop_channels[0]].temperature.sweep(0,10,1))
        loop = loop.loop(self.instrument.channels[loop_channels[1]].temperature.sweep(50,51,0.1))
        loop = loop.each(self.instrument.channels[measure_channel].temperature)
        data = loop.run()
        # for i,chan in enumerate(['A', 'B', 'C', 'D']):
        #     self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.shape, (21,))
        #     self.assertEqual(getattr(data, 'testchanneldummy_Chan{}_temperature'.format(chan)).ndarray.max(), values[i])



if __name__ == '__main__':
    unittest.main()