""" Base class for the channel of an instrument """
from typing import List, Tuple, Union

from .base import InstrumentBase, Instrument
from .parameter import MultiParameter, ArrayParameter
from ..utils.metadata import Metadatable
from ..utils.helpers import full_class


class InstrumentChannel(InstrumentBase):
    """
    Base class for a channel in an instrument

    Args:
        parent (Instrument): the instrument to which this channel should be
          attached

        name (str): the name of this channel

    Attributes:
        name (str): the name of this channel

        parameters (Dict[Parameter]): All the parameters supported by this
          channel. Usually populated via ``add_parameter``

        functions (Dict[Function]): All the functions supported by this
          channel. Usually populated via ``add_function``
    """

    def __init__(self, parent: Instrument, name: str, **kwargs):
        # Initialize base classes of Instrument. We will overwrite what we
        # want to do in the Instrument initializer
        super().__init__(name=name, **kwargs)

        self.name = "{}_{}".format(parent.name, str(name))
        self.short_name = str(name)
        self._meta_attrs = ['name']

        self._parent = parent

    def __repr__(self):
        """Custom repr to give parent information"""
        return '<{}: {} of {}: {}>'.format(type(self).__name__,
                                           self.name,
                                           type(self._parent).__name__,
                                           self._parent.name)

    # Pass any commands to read or write from the instrument up to the parent
    def write(self, cmd):
        return self._parent.write(cmd)

    def write_raw(self, cmd):
        return self._parent.write_raw(cmd)

    def ask(self, cmd):
        return self._parent.ask(cmd)

    def ask_raw(self, cmd):
        return self._parent.ask_raw(cmd)


class MultiChannelInstrumentParameter(MultiParameter):
    """
    Parameter to get or set multiple channels simultaneously.

    Will normally be created by a ChannelList and not directly by anything
    else.

    Args:
        channels(list[chan_type]): A list of channels which we can operate on
          simultaneously.

        param_name(str): Name of the multichannel parameter
    """
    def __init__(self,
                 channels: Union[List, Tuple],
                 param_name: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._param_name = param_name

    def get(self) -> tuple:
        """
        Return a tuple containing the data from each of the channels in the
        list
        """
        return tuple(chan.parameters[self._param_name].get() for chan
                     in self._channels)

    def set(self, value):
        """
        Set all parameters to this value

        Args:
            value (unknown): The value to set to. The type is given by the
            underlying parameter.
        """
        for chan in self._channels:
            getattr(chan, self._param_name).set(value)

    @property
    def full_names(self):
        """Overwrite full_names because the instrument name is already included
        in the name. This happens because the instrument name is included in
        the channel name merged into the parameter name above.
        """

        return self.names


class ChannelList(Metadatable):
    """
    Container for channelized parameters that allows for sweeps over
    all channels, as well as addressing of individual channels.

    Args:
        parent (Instrument): the instrument to which this channel
            should be attached

        name (string): the name of the channel list

        chan_type (InstrumentChannel): the type of channel contained
            within this list

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

    def __init__(self, parent: Instrument,
                 name: str,
                 chan_type: type,
                 chan_list: Union[List, Tuple, None]=None,
                 snapshotable: bool=True,
                 multichan_paramclass: type = MultiChannelInstrumentParameter):
        super().__init__()

        self._parent = parent
        self._name = name
        if (not isinstance(chan_type, type) or
                not issubclass(chan_type, InstrumentChannel)):
            raise ValueError("Channel Lists can only hold instances of type"
                             " InstrumentChannel")
        if (not isinstance(multichan_paramclass, type) or
                not issubclass(multichan_paramclass,
                               MultiChannelInstrumentParameter)):
            raise ValueError("multichan_paramclass must be a (subclass of) "
                             "MultiChannelInstrumentParameter")

        self._chan_type = chan_type
        self._snapshotable = snapshotable
        self._paramclass = multichan_paramclass

        self._channel_mapping = {}  # provide lookup of channels by name
        # If a list of channels is not provided, define a list to store
        # channels. This will eventually become a locked tuple.
        if chan_list is None:
            self._locked = False
            self._channels = []
        else:
            self._locked = True
            self._channels = tuple(chan_list)
            self._channel_mapping = {channel.short_name: channel
                                     for channel in self._channels}
            if not all(isinstance(chan, chan_type) for chan in self._channels):
                raise TypeError("All items in this channel list must be of "
                                "type {}.".format(chan_type.__name__))

    def __getitem__(self, i: Union[int, slice]):
        """
        Return either a single channel, or a new ChannelList containing only
        the specified channels

        Args:
            i (int/slice): Either a single channel index or a slice of channels
              to get
        """
        if isinstance(i, slice):
            return ChannelList(self._parent, self._name, self._chan_type,
                               self._channels[i],
                               multichan_paramclass=self._paramclass)
        return self._channels[i]

    def __iter__(self):
        return iter(self._channels)

    def __len__(self):
        return len(self._channels)

    def __repr__(self):
        return "ChannelList({!r}, {}, {!r})".format(self._parent,
                                                    self._chan_type.__name__,
                                                    self._channels)

    def __add__(self, other: 'ChannelList'):
        """
        Return a new channel list containing the channels from both
        ChannelList self and r.

        Both channel lists must hold the same type and have the same parent.

        Args:
            other(ChannelList): Right argument to add.
        """
        if not isinstance(self, ChannelList) or not isinstance(other,
                                                               ChannelList):
            raise TypeError("Can't add objects of type"
                            " {} and {} together".format(type(self).__name__,
                                                         type(other).__name__))
        if self._chan_type != other._chan_type:
            raise TypeError("Both l and r arguments to add must contain "
                            "channels of the same type."
                            " Adding channels of type "
                            "{} and {}.".format(self._chan_type.__name__,
                                                other._chan_type.__name__))
        if self._parent != other._parent:
            raise ValueError("Can only add channels from the same parent "
                             "together.")

        return ChannelList(self._parent, self._name, self._chan_type,
                           self._channels + other._channels)

    def append(self, obj: InstrumentChannel):
        """
        When initially constructing the channel list, a new channel to add to
        the end of the list

        Args:
            obj(chan_type): New channel to add to the list.
        """
        if self._locked:
            raise AttributeError("Cannot append to a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError("All items in a channel list must be of the same "
                            "type. Adding {} to a list of {}"
                            ".".format(type(obj).__name__,
                                       self._chan_type.__name__))
        self._channel_mapping[obj.short_name] = obj
        return self._channels.append(obj)

    def extend(self, objects):
        """
        Insert an iterable of objects into the list of channels.

        Args:
            objects(Iterable[chan_type]): A list of objects to add into the
              ChannelList.
        """
        # objects may be a generator but we need to iterate over it twice
        # below so copy it into a tuple just in case.
        objects = tuple(objects)
        if self._locked:
            raise AttributeError("Cannot extend a locked channel list")
        if not all(isinstance(obj, self._chan_type) for obj in objects):
            raise TypeError("All items in a channel list must be of the same "
                            "type.")
        return self._channels.extend(objects)

    def index(self, obj: InstrumentChannel):
        """
        Return the index of the given object

        Args:
            obj(chan_type): The object to find in the channel list.
        """
        return self._channels.index(obj)

    def insert(self, index: int, obj: InstrumentChannel):
        """
        Insert an object into the channel list at a specific index.

        Args:
            index(int): Index to insert object.

            obj(chan_type): Object of type chan_type to insert.
        """
        if self._locked:
            raise AttributeError("Cannot insert into a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError("All items in a channel list must be of the same "
                            "type. Adding {} to a list of {}"
                            ".".format(type(obj).__name__,
                                       self._chan_type.__name__))

        return self._channels.insert(index, obj)

    def lock(self):
        """
        Lock the channel list. Once this is done, the channel list is
        converted to a tuple and any future changes to the list are prevented.
        """
        if self._locked:
            return

        self._channels = tuple(self._channels)
        self._locked = True

    def snapshot_base(self, update: bool=False):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory..

        Returns:
            dict: base snapshot
        """
        if self._snapshotable:
            snap = {'channels': dict((chan.name, chan.snapshot(update=update))
                                     for chan in self._channels),
                    'snapshotable': self._snapshotable,
                    '__class__': full_class(self),
                    }
        else:
            snap = {'snapshotable': self._snapshotable,
                    '__class__': full_class(self),
                    }
        return snap

    def __getattr__(self, name: str):
        """
        Return a multi-channel function or parameter that we can use to get or
        set all items in a channel list simultaneously.

        Params:
            name(str): The name of the parameter or function that we want to
            operate on.
        """
        # Check if this is a valid parameter
        if name in self._channels[0].parameters:
            setpoints = None
            setpoint_names = None
            setpoint_labels = None
            setpoint_units = None
            # We need to construct a MultiParameter object to get each of the
            # values our of each parameter in our list, we don't currently try
            # to construct a multiparameter from a list of multi parameters
            if isinstance(self._channels[0].parameters[name], MultiParameter):
                raise NotImplementedError("Slicing is currently not "
                                          "supported for MultiParameters")
            names = tuple("{}_{}".format(chan.name, name)
                          for chan in self._channels)
            labels = tuple(chan.parameters[name].label
                           for chan in self._channels)
            units = tuple(chan.parameters[name].unit
                          for chan in self._channels)

            if isinstance(self._channels[0].parameters[name], ArrayParameter):
                shapes = tuple(chan.parameters[name].shape for
                               chan in self._channels)

                if self._channels[0].parameters[name].setpoints:
                    setpoints = tuple(chan.parameters[name].setpoints for
                                      chan in self._channels)
                if self._channels[0].parameters[name].setpoint_names:
                    setpoint_names = tuple(chan.parameters[name].setpoint_names
                                           for chan in self._channels)
                if self._channels[0].parameters[name].setpoint_labels:
                    setpoint_labels = tuple(
                        chan.parameters[name].setpoint_labels
                        for chan in self._channels)
                if self._channels[0].parameters[name].setpoint_units:
                    setpoint_units = tuple(chan.parameters[name].setpoint_units
                                           for chan in self._channels)
            else:
                shapes = tuple(() for _ in self._channels)

            param = self._paramclass(self._channels,
                                     param_name=name,
                                     name="Multi_{}".format(name),
                                     names=names,
                                     shapes=shapes,
                                     instrument=self._parent,
                                     labels=labels,
                                     units=units,
                                     setpoints=setpoints,
                                     setpoint_names=setpoint_names,
                                     setpoint_units=setpoint_units,
                                     setpoint_labels=setpoint_labels)
            return param

        # Check if this is a valid function
        if name in self._channels[0].functions:
            # We want to return a reference to a function that would call the
            # function for each of the channels in turn.
            def multi_func(*args, **kwargs):
                for chan in self._channels:
                    chan.functions[name](*args, **kwargs)
            return multi_func

        try:
            return self._channel_mapping[name]
        except KeyError:
            pass

        raise AttributeError('\'{}\' object has no attribute \'{}\''
                             ''.format(self.__class__.__name__, name))

    def __dir__(self) -> list:
        names = super().__dir__()
        if self._channels:
            names += list(self._channels[0].parameters.keys())
            names += list(self._channels[0].functions.keys())
            names += [channel.short_name for channel in self._channels]
        return sorted(set(names))
