""" Base class for the channel of an instrument """
from typing import (
    List, Union, Optional, Dict, Sequence,
    cast, Any
)

from .base import InstrumentBase, Instrument
from .parameter import MultiParameter, ArrayParameter, Parameter
from ..utils.validators import Validator
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

    def __init__(self,
                 parent: Union[Instrument, 'InstrumentChannel'],
                 name: str,
                 **kwargs) -> None:
        # need to specify parent before `super().__init__` so that the right
        # `full_name` is available in that scope. `full_name` is used for
        # registering the filter for the log messages. It is composed by
        # iteratively concatenating the full names of the parent instruments
        # scope of `Base`
        self._parent = parent
        super().__init__(name=name, **kwargs)
        # Naming insanity:
        # (see https://github.com/QCoDeS/Qcodes/issues/1140 for a nice table)
        # this has been a confusion about names. don't use name but
        # full_name, or short_name.
        self.name = "{}_{}".format(parent.name, str(name))



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

    @property
    def parent(self) -> InstrumentBase:
        return self._parent

    @property
    def root_instrument(self) -> InstrumentBase:
        return self._parent.root_instrument

    @property
    def name_parts(self) -> List[str]:
        name_parts = self._parent.name_parts
        name_parts.append(self.short_name)
        return name_parts


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
                 channels: Sequence[InstrumentChannel],
                 param_name: str,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._param_name = param_name

    def get_raw(self) -> tuple:
        """
        Return a tuple containing the data from each of the channels in the
        list
        """
        return tuple(chan.parameters[self._param_name].get() for chan
                     in self._channels)

    def set_raw(self, value):
        """
        Set all parameters to this value

        Args:
            value (Any): The value to set to. The type is given by the
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

        name (str): the name of the channel list

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
                 chan_list: Optional[Sequence[InstrumentChannel]]=None,
                 snapshotable: bool=True,
                 multichan_paramclass: type = MultiChannelInstrumentParameter) -> None:
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

        self._channel_mapping: Dict[str, InstrumentChannel] = {}
        # provide lookup of channels by name
        # If a list of channels is not provided, define a list to store
        # channels. This will eventually become a locked tuple.
        self._channels: Sequence[InstrumentChannel]
        if chan_list is None:
            self._locked = False
            self._channels = []
        else:
            self._locked = True
            self._channels = tuple(chan_list)
            if self._channels is None:
                raise RuntimeError("Empty channel list")
            self._channel_mapping = {channel.short_name: channel
                                     for channel in self._channels}
            if not all(isinstance(chan, chan_type) for chan in self._channels):
                raise TypeError("All items in this channel list must be of "
                                "type {}.".format(chan_type.__name__))

    def __getitem__(self, i: Union[int, slice, tuple]):
        """
        Return either a single channel, or a new ChannelList containing only
        the specified channels

        Args:
            i (int, slice): Either a single channel index or a slice of channels
              to get
        """
        if isinstance(i, slice):
            return ChannelList(self._parent, self._name, self._chan_type,
                               self._channels[i],
                               multichan_paramclass=self._paramclass)
        elif isinstance(i, tuple):
            return ChannelList(self._parent, self._name, self._chan_type,
                               [self._channels[j] for j in i],
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
                           list(self._channels) + list(other._channels))

    def append(self, obj: InstrumentChannel):
        """
        When initially constructing the channel list, a new channel to add to
        the end of the list

        Args:
            obj(chan_type): New channel to add to the list.
        """
        if (isinstance(self._channels, tuple) or self._locked):
            raise AttributeError("Cannot append to a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError("All items in a channel list must be of the same "
                            "type. Adding {} to a list of {}"
                            ".".format(type(obj).__name__,
                                       self._chan_type.__name__))
        self._channel_mapping[obj.short_name] = obj
        self._channels = cast(List[InstrumentChannel], self._channels)
        return self._channels.append(obj)

    def clear(self):
        """
        Clear all items from the channel list.
        """
        if self._locked:
            raise AttributeError("Cannot clear a locked channel list")
        self._channels.clear()
        self._channel_mapping.clear()

    def remove(self, obj: InstrumentChannel):
        """
        Removes obj from channellist if not locked.
        Args:
            obj: Channel to remove from the list.
        """
        if self._locked:
            raise AttributeError("Cannot remove from a locked channel list")
        else:
            self._channels = cast(List[InstrumentChannel], self._channels)
            self._channels.remove(obj)
            self._channel_mapping.pop(obj.short_name)

    def extend(self, objects: Sequence[InstrumentChannel]):
        """
        Insert an iterable of objects into the list of channels.

        Args:
            objects(Iterable[chan_type]): A list of objects to add into the
              ChannelList.
        """
        # objects may be a generator but we need to iterate over it twice
        # below so copy it into a tuple just in case.
        objects_tuple = tuple(objects)
        if self._locked:
            raise AttributeError("Cannot extend a locked channel list")
        if not all(isinstance(obj, self._chan_type) for obj in objects_tuple):
            raise TypeError("All items in a channel list must be of the same "
                            "type.")
        channels = cast(List[InstrumentChannel], self._channels)
        channels.extend(objects_tuple)
        self._channel_mapping.update({
            obj.short_name: obj for obj in objects
        })
        self._channels = channels

    def index(self, obj: InstrumentChannel):
        """
        Return the index of the given object

        Args:
            obj(chan_type): The object to find in the channel list.
        """
        return self._channels.index(obj)

    def insert(self, index: int, obj: InstrumentChannel) -> None:
        """
        Insert an object into the channel list at a specific index.

        Args:
            index(int): Index to insert object.

            obj(chan_type): Object of type chan_type to insert.
        """
        if (isinstance(self._channels, tuple) or self._locked):
            raise AttributeError("Cannot insert into a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError("All items in a channel list must be of the same "
                            "type. Adding {} to a list of {}"
                            ".".format(type(obj).__name__,
                                       self._chan_type.__name__))
        self._channels = cast(List[InstrumentChannel], self._channels)
        self._channels.insert(index, obj)

    def get_validator(self):
        """
        Returns a validator that checks that the returned object is a channel
        in this channel list
        """
        if not self._locked:
            raise AttributeError("Cannot create a validator for an unlocked channel list")
        return ChannelListValidator(self)

    def lock(self) -> None:
        """
        Lock the channel list. Once this is done, the channel list is
        converted to a tuple and any future changes to the list are prevented.
        """
        if self._locked:
            return

        self._channels = tuple(self._channels)
        self._locked = True

    def snapshot_base(self, update: bool=False, params_to_skip_update: Optional[Sequence[str]]=None):
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
            parameters = cast(List[Union[Parameter, ArrayParameter]],
                              [chan.parameters[name] for chan in self._channels])
            names = tuple("{}_{}".format(chan.name, name)
                          for chan in self._channels)
            labels = tuple(parameter.label
                           for parameter in parameters)
            units = tuple(parameter.unit
                          for parameter in parameters)

            if isinstance(parameters[0], ArrayParameter):
                arrayparameters = cast(List[ArrayParameter],parameters)
                shapes = tuple(parameter.shape for
                               parameter in arrayparameters)
                if arrayparameters[0].setpoints:
                    setpoints = tuple(parameter.setpoints for
                                      parameter in arrayparameters)
                if arrayparameters[0].setpoint_names:
                    setpoint_names = tuple(parameter.setpoint_names for
                                           parameter in arrayparameters)
                if arrayparameters[0].setpoint_labels:
                    setpoint_labels = tuple(
                        parameter.setpoint_labels
                        for parameter in arrayparameters)
                if arrayparameters[0].setpoint_units:
                    setpoint_units = tuple(parameter.setpoint_units
                                           for parameter in arrayparameters)
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
        names = list(super().__dir__())
        if self._channels:
            names += list(self._channels[0].parameters.keys())
            names += list(self._channels[0].functions.keys())
            names += [channel.short_name for channel in self._channels]
        return sorted(set(names))


class ChannelListValidator(Validator):
    """
    A validator that checks that the returned object is a member of the
    channel list with which the validator was constructed.

    This class will not normally be created directly, but created from a channel
    list using the `ChannelList.get_validator` method.

    Args:
        channel_list (ChannelList): the channel list that should be checked against.
            The channel list must be locked and populated before it can be used to
            construct a validator.
    """
    def __init__(self, channel_list: ChannelList) -> None:
        # Save the base parameter list
        if not isinstance(channel_list, ChannelList):
            raise ValueError("channel_list must be a ChannelList object containing the "
                "channels that should be validated")
        if not channel_list._locked:
            raise AttributeError("Channel list must be locked before it can be used "
                "to create a validator")
        self._channel_list = channel_list

    def validate(self, value, context: str='') -> None:
        """
        Checks to see that value is a member of the channel list referenced by this
        validator

        Args:
            value (InstrumentChannel): the value to be checked against the reference
                channel list.

            context (str): the context of the call, used as part of the exception
                raised.
        """
        if value not in self._channel_list:
            raise ValueError(
                '{} is not part of the expected channel list; {}'.format(
                    repr(value), context))


class AutoLoadableInstrumentChannel(InstrumentChannel):
    """
    This subclass provides extensions to auto-load channels
    from instruments and adds methods to create and delete
    channels when possible. Please note that `channel` in this
    context does not necessarily mean a physical instrument channel,
    but rather an instrument sub-module. For some instruments,
    these sub-modules can be created and deleted at will.
    """

    @classmethod
    def load_from_instrument(
            cls, parent: Instrument,
            channel_list: 'AutoLoadableChannelList'=None,
            **kwargs
    )->List['AutoLoadableInstrumentChannel']:
        """
        Load channels that already exist on the instrument

        Args:
            parent: The instrument through which the instrument
                channel is accessible
            channel_list: The channel list this
                channel is a part of
            **kwargs: Keyword arguments needed to create the channels

        Returns:
            List of instrument channel instances created for channels
            that already exist on the instrument
        """

        obj_list = []
        for new_kwargs in cls._discover_from_instrument(parent, **kwargs):
            obj = cls(
                parent, existence=True, channel_list=channel_list,
                **new_kwargs
            )
            obj_list.append(obj)

        return obj_list

    @classmethod
    def _discover_from_instrument(
            cls, parent: Instrument, **kwargs) ->List[dict]:
        """
        Discover channels on the instrument and return a list kwargs to create
        these channels in memory

        Args:
            parent: The instrument through which the instrument
                channel is accessible
            **kwargs: Keyword arguments needed to discover the channels

        Returns:
              List of keyword arguments for channel instance initialization
              for each channel that already exists on the physical instrument
        """
        raise NotImplementedError(
            "Please subclass and implement this method in the subclass")

    @classmethod
    def new_instance(
            cls, parent: Instrument, create_on_instrument: bool=True,
            channel_list: 'AutoLoadableChannelList'=None, **kwargs
    )->'AutoLoadableInstrumentChannel':
        """
        Create a new instance of the channel on the instrument: This involves
        finding initialization arguments which will create a channel with a
        unique name and create the channel on the instrument.

        Args:
            parent: The instrument through which the instrument
                channel is accessible
            create_on_instrument: When True, the channel is immediately
                created on the instrument
            channel_list: The channel list this
                channel is going to belong to
            **kwargs: Keyword arguments needed to create a new instance.
        """
        new_kwargs = cls._get_new_instance_kwargs(parent=parent, **kwargs)

        try:
            new_instance = cls(parent, channel_list=channel_list, **new_kwargs)
        except TypeError as err:
            # The 'new_kwargs' dict is malformed. Investigate more precisely
            # why and give the user a more helpful hint how this can be
            # solved.
            if "name" not in new_kwargs:
                raise TypeError(
                    "A 'name' argument should be supplied by the "
                    "'_get_new_instance_kwargs' method"
                ) from err
            if "parent" in new_kwargs:
                raise TypeError(
                    "A 'parent' argument should *not* be supplied by the "
                    "'_get_new_instance_kwargs' method"
                ) from err
            # Something else has gone wrong. Probably, not all mandatory keyword
            # arguments are supplied
            raise TypeError(
                "Probably, the '_get_new_instance_kwargs' method does not "
                "return all of the required keyword arguments") from err

        if create_on_instrument:
            new_instance.create()

        return new_instance

    @classmethod
    def _get_new_instance_kwargs(cls, parent: Instrument=None, **kwargs)->dict:
        """
        Returns a dictionary which is used as keyword args when instantiating a
        channel

        Args:
            parent: The instrument the new channel will belong to. Not all
                instruments need this so it is an optional argument
            **kwargs: Additional arguments which are needed to
                instantiate a channel can be given directly by the calling
                function.

        Returns:
            A keyword argument dictionary with at least a `name` key which is
            unique on the instrument. The parent instrument is passed as an
            argument in this function so we can query if the generated name is
            indeed unique.

        Notes:
            The init arguments `parent` and `channel_list` are automatically
            added by the `new_instance` method and should not be added in the
            kwarg dictionary returned here. Additionally, the argument
            `existence` either needs to be omitted or be False.
        """
        raise NotImplementedError(
            "Please subclass and implement this method in the subclass")

    def __init__(
            self,
            parent: Union[Instrument, 'InstrumentChannel'],
            name: str,
            exists_on_instrument: bool=False,
            channel_list: 'AutoLoadableChannelList'=None,
            **kwargs
    ) ->None:
        """
        Instantiate a channel object. Note that this is not the same as actually
        creating the channel on the instrument. Parameters defined on this
        channels will not be able to query/write to the instrument until it
        has been created on the instrument

        Args:
            parent: The instrument through which the instrument
                channel is accessible
            name: channel name
            exists_on_instrument: True if the channel exists on the instrument
            channel_list: Reference to the list that this channel is a member
                of; this is used when deleting the channel so that it can remove
                itself from the list
        """
        super().__init__(parent, name=name, **kwargs)
        self._exists_on_instrument = exists_on_instrument
        self._channel_list = channel_list

    def create(self) ->None:
        """Create the channel on the instrument"""
        if self._exists_on_instrument:
            raise RuntimeError("Channel already exists on instrument")

        self._create()
        self._exists_on_instrument = True

    def _create(self)->None:
        """
        (SCPI) commands needed to create the channel. Note that we
        need to use `self.root_instrument.write` to send commands,
        because self.write will cause _assert_existence to raise a
        runtime error.
        """
        raise NotImplementedError("Please subclass")

    def remove(self)->None:
        """
        Delete the channel from the instrument and remove from channel list
        """
        self._assert_existence()
        self._remove()
        if self._channel_list is not None and self in self._channel_list:
            self._channel_list.remove(self)
        self._exists_on_instrument = False

    def _remove(self) ->None:
        """
        (SCPI) commands needed to delete the channel from the instrument
        """
        raise NotImplementedError("Please subclass")

    def _assert_existence(self)->None:
        if not self._exists_on_instrument:
            raise RuntimeError(
                "Object does not exist (anymore) on the instrument")

    def write(self, cmd: str)->Any:
        """
        Write to the instrument only if the channel is present on the instrument
        """
        self._assert_existence()
        return super().write(cmd)

    def ask(self, cmd: str)->None:
        """
        Ask the instrument only if the channel is present on the instrument
        """
        self._assert_existence()
        return super().ask(cmd)

    @property
    def exists_on_instrument(self)->bool:
        return self._exists_on_instrument


class AutoLoadableChannelList(ChannelList):
    """
    Extends the QCoDeS ChannelList class to add the following features:
    - Automatically create channel objects on initialization
    - Make a `add` method to create channel objects

    Args:
        parent: the instrument to which this channel
            should be attached

        name: the name of the channel list

        chan_type: the type of channel contained
            within this list

        chan_list: An optional iterable of
            channels of type chan_type.  This will create a list and
            immediately lock the ChannelList.

        snapshotable: Optionally disables taking of snapshots
            for a given channel list.  This is used when objects
            stored inside a channel list are accessible in multiple
            ways and should not be repeated in an instrument snapshot.

        multichan_paramclass: The class of
            the object to be returned by the ChanneList's __getattr__ method.
            Should be a subclass of MultiChannelInstrumentParameter.

        **kwargs: Keyword arguments to be passed to the `load_from_instrument`
            method of the channel class. Note that the kwargs are *NOT* passed
            to the `__init__` of the super class.

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
            chan_list: Optional[Sequence['AutoLoadableInstrumentChannel']]=None,
            snapshotable: bool=True,
            multichan_paramclass: type=MultiChannelInstrumentParameter,
            **kwargs
    ) ->None:

        super().__init__(
            parent, name, chan_type, chan_list, snapshotable,
            multichan_paramclass
        )
        new_channels = self._chan_type.load_from_instrument(  # type: ignore
            self._parent, channel_list=self, **kwargs)

        self.extend(new_channels)

    def add(self, **kwargs) ->'AutoLoadableInstrumentChannel':
        """
        Add a channel to the list

        Args:
            kwargs: Keyword arguments passed to the `new_instance` method of
                the channel class

        Returns:
            Newly created instance of the channel class
        """
        new_channel = self._chan_type.new_instance(  # type: ignore
            self._parent,
            create_on_instrument=True,
            channel_list=self,
            **kwargs
        )

        self.append(new_channel)
        return new_channel
