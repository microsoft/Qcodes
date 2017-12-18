import logging
import re
import warnings

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes.utils.validators import Ints

log = logging.getLogger(__name__)


class SwitchChannelBase(InstrumentChannel):
    def __init__(self, parent, name, channel_letter):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of
            name (str): the name of the channel
            channel_letter (str): channel letter ['a', 'b', 'c' or 'd'])
        """

        super().__init__(parent, name)
        self.channel_letter = channel_letter.upper()
        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.channel_number = _chanlist.index(channel_letter)

        self.add_parameter(
            'switch',
            label='switch {}'.format(self.channel_letter),
            set_cmd=self._set_switch,
            get_cmd=self._get_switch,
            vals=Ints(1, 2))

    def __call__(self, *args):
        if len(args) == 1:
            self.switch(args[0])
        elif len(args) == 0:
            return self.switch()
        else:
            raise RuntimeError(
                'Call channel with either one or zero arguments')

    def _set_switch(self, switch):
        raise NotImplementedError()

    def _get_switch(self):
        raise NotImplementedError()


class SPDT_Base(Instrument):
    @property
    def CHANNEL_CLASS(self):
        raise NotImplementedError

    def add_channels(self):
        channels = ChannelList(
            self, "Channels", self.CHANNEL_CLASS, snapshotable=False)

        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self._deprecated_attributes = {
            'channel_{}'.format(k): k
            for k in _chanlist
        }

        _max_channel_number = self.get_number_of_channels()
        _chanlist = _chanlist[0:_max_channel_number]

        for c in _chanlist:
            channel = self.CHANNEL_CLASS(self, 'channel_{}'.format(c), c)
            channels.append(channel)
            attribute_name = 'channel_{}'.format(c)
            self.add_submodule(attribute_name, channel)
            self.add_submodule(c, channel)
            self._deprecated_attributes[attribute_name] = c
        channels.lock()
        self.add_submodule('channels', channels)

    def all(self, switch_to):
        for c in self.channels:
            c.switch(switch_to)

    def __getattr__(self, key):
        if key in self._deprecated_attributes:
            warnings.warn(
                ("Using '{}' is deprecated and will be removed in future" +
                 "releases. Use '{}' instead").
                format(key, self._deprecated_attributes[key]), UserWarning)
        return super().__getattr__(key)

    def get_number_of_channels(self):
        model = self.get_idn()['model']
        model_parts = model.split('-')
        if len(model_parts) < 2:
            raise RuntimeError(
                ('The driver could not determine the number of channels of ' +
                 'the model \'{}\', it might not be supported').format(model)
            )
        if model_parts[0] not in ('RC', 'USB'):
            log.warning(
                ('The model with the name \'{}\' might not be supported by' +
                 ' the driver').format(model))
        channels = re.match('^[0-9]*', model_parts[1])[0]
        if not channels:
            raise RuntimeError(
                'The driver could not determine the number of channels of' +
                ' the model \'{}\', it might not be supported'.format(model)
            )
        return int(channels)
