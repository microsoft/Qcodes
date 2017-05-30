from unittest import TestCase
import gc

from .instrument_mocks import DummyChannelInstrument, DummyChannel

from hypothesis import given
from hypothesis.strategies import floats, integers


class TestChannels(TestCase):

    def setUp(self):
        self.instrument = DummyChannelInstrument(name='testchanneldummy')

    def tearDown(self):
        # force gc run
        del self.instrument
        gc.collect()

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

