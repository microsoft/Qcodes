"""
This module contains extensions to the QCoDeS InstrumentChannel and ChannelList
classes. On the N52xx instruments, we can define channels dynamically through
the API. The extension class defines how to discover, create and delete
these channels and keeps track of creation and deletion. Write and ask actions
are only allowed if the channels exists on the instrument.
"""

from typing import List, Optional, Sequence, Any

from qcodes import Instrument, InstrumentChannel, ChannelList
from qcodes.instrument.channel import MultiChannelInstrumentParameter


class N52xxInstrumentChannel(InstrumentChannel):
    """
    This is *NOT* the implementation of channels as defined in the programmers
    manual of these instruments. The latter is implemented in `N52xxChannel`.
    As stated above, this module extends the QCoDeS InstrumentChannel and
    ChannelList classes.
    """

    discover_command = None

    @classmethod
    def load_from_instrument(
            cls, parent: Instrument, **kwargs) ->List['N52xxInstrumentChannel']:
        """
        Discover and load channels from the instrument

        Args:
            parent (Instrument): The instrument through which the instrument
                channel is accessible
            **kwargs (dict): Keyword arguments needed to create the channels

        Returns:
            obj_list (list): List of instrument channels
        """

        obj_list = []
        for identifier in cls._discover_list_from_instrument(parent):
            obj = cls(parent, identifier=identifier, existence=True, **kwargs)
            obj_list.append(obj)

        return obj_list

    @classmethod
    def _discover_list_from_instrument(
            cls, parent: Instrument, **kwargs) ->List[Any]:
        """
        Discover channels on the instrument and return a list of unique
        identifiers for each channel

        Args:
            parent (Instrument): The instrument through which the instrument
                channel is accessible
            **kwargs (dict): Keyword arguments needed to discover the channels

        Returns:
              list: List of unique identifiers, e.g. channel numbers
        """
        if cls.discover_command is None:
            raise NotImplementedError("Please subclass")

        ans = parent.ask(cls.discover_command).strip().strip("\"").split(",")
        return [int(i) for i in ans if i != ""]

    @classmethod
    def make_unique_id(cls, parent: Instrument, **kwargs) ->Any:
        """
        Given a list of ID's, make a new unique ID. By default we assume that
        ID's are numbers (e.g. channel numbers). Simply return a number not
        already in the list

        Args:
            parent (Instrument): The instrument through which the instrument
                channel is accessible
        """
        existing_ids = cls._discover_list_from_instrument(parent)
        new_id = 1
        while new_id in existing_ids:
            new_id += 1

        return new_id

    def __init__(
            self, parent: Instrument, identifier: Any, existence: bool=False,
            channel_list: 'N52xxChannelList'=None, **kwargs) ->None:
        """
        Instantiate a channel object. Note that this is not the same as actually
        creating the channel on the instrument. Parameters defined on this
        channels will note be able to query/write to the instrument until it
        has been created

        Args:
            parent (Instrument): The instrument through which the instrument
                channel is accessible
            identifier: A unique identifier for this channel (e.g. a channel
                number)
            existence (bool): True if the channel exists on the instrument
            channel_list (N52xxChannelList): If the channel is deleted,
                delete `self` from the channel list
        """
        super().__init__(parent, name=identifier, **kwargs)
        self._exists_on_instrument = existence
        self._channel_list = channel_list
        self.base_instrument = parent.base_instrument

    def create(self) ->None:
        """Create the channel on the instrument"""
        self._create()
        self._exists_on_instrument = True

    def _create(self) ->None:
        raise NotImplementedError("Please subclass")

    def delete(self) ->None:
        """
        Delete the channel from the instrument and remove from channel list
        """
        self._delete()
        if self._channel_list is not None:
            self._channel_list.remove(self)
        self._exists_on_instrument = False

    def _delete(self) ->None:
        raise NotImplementedError("Please subclass")

    def _assert_existence(self) ->None:
        if not self._exists_on_instrument:
            raise RuntimeError(
                "Object does not exist (anymore) on the instrument")

    def write(self, cmd: str) ->Any:
        """
        Only write to the instrument if the channel is present on the instrument
        """
        self._assert_existence()
        return super().write(cmd)

    def ask(self, cmd: str) ->None:
        """
        Only query to the instrument if the channel is present on the instrument
        """
        self._assert_existence()
        return super().ask(cmd)

    @property
    def exists_on_instrument(self):
        return self._exists_on_instrument


class N52xxChannelList(ChannelList):
    """
    Extends the QCoDeS ChannelList class to add the following features:
    - Automatically create channel objects on initialization
    - Make a `add` method to create channel objects
    - Get channel objects by their short name

    Args:
        parent (Instrument): the instrument to which this channel
            should be attached

        name (string): the name of the channel list

        chan_type (N52xxInstrumentChannel): the type of channel contained
            within this list. This should be a subclass of
            N52xxInstrumentChannel

        chan_list (Iterable[chan_type]): An optional iterable of
            channels of type chan_type.  This will create a list and
            immediately lock the ChannelList.

        snapshotable (bool): Optionally disables taking of snapshots
            for a given channel list.  This is used when objects
            stored inside a channel list are accessible in multiple
            ways and should not be repeated in an instrument snapshot.

        multichan_paramclass (MultiChannelInstrumentParameter): The class of
            the object to be returned by the ChanneList's __getattr__ method.
            Should be a subclass of MultiChannelInstrumentParameter.

    Raises:
        ValueError: If chan_type is not a subclass of InstrumentChannel
        ValueError: If multichan_paramclass if not a subclass of
            MultiChannelInstrumentParameter (note that a class is a subclass
            of itself).
    """
    def __init__(
            self,
            parent: Instrument,
            name: str,
            chan_type: type,
            chan_list: Optional[Sequence['N52xxInstrumentChannel']] = None,
            snapshotable: bool = True,
            multichan_paramclass: type = MultiChannelInstrumentParameter,
            **kwargs
    ) ->None:

        super().__init__(parent, name, chan_type, chan_list, snapshotable,
                         multichan_paramclass)

        new_channels = self._chan_type.load_from_instrument(
            self._parent, channel_list=self, **kwargs)

        for channel in new_channels:
            # NB: There is a bug in `extend`. TODO: Make a PR!
            self.append(channel)

    def add(self, **kwargs) ->None:
        """
        Add a channel to the list
        """
        channel_number = self._chan_type.make_unique_id(self._parent, **kwargs)

        new_channel = self._chan_type(
            self._parent, identifier=str(channel_number), channel_list=self,
            **kwargs
        )

        new_channel.create()
        self.append(new_channel)
        return new_channel

    def __getitem__(self, item: Any) ->Any:
        """
        Get an item by its short name
        """
        if isinstance(item, str):
            return self._channel_mapping[item]

        return super().__getitem__(item)
