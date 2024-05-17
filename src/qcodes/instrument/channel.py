""" Base class for the channel of an instrument """
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast, overload

from qcodes.metadatable import MetadatableWithName
from qcodes.parameters import (
    ArrayParameter,
    MultiChannelInstrumentParameter,
    MultiParameter,
    Parameter,
)
from qcodes.parameters.multi_channel_instrument_parameter import InstrumentModuleType
from qcodes.utils import full_class
from qcodes.validators import Validator

from .instrument_base import InstrumentBase

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .instrument import Instrument
    from .instrument_base import InstrumentBaseKWArgs


class InstrumentModule(InstrumentBase):
    """
    Base class for a module in an instrument.
    This could be in the form of a channel (e.g. something that
    the instrument has multiple instances of) or another logical grouping
    of parameters that you wish to group together separate from the rest of the
    instrument.

    Args:
        parent: The instrument to which this module should be
          attached.
        name: The name of this module.
        **kwargs: Forwarded to the base class.

    """

    def __init__(
        self, parent: InstrumentBase, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]
    ) -> None:
        # need to specify parent before `super().__init__` so that the right
        # `full_name` is available in that scope. `full_name` is used for
        # registering the filter for the log messages. It is composed by
        # iteratively concatenating the full names of the parent instruments
        # scope of `Base`
        self._parent = parent
        super().__init__(name=name, **kwargs)

    def __repr__(self) -> str:
        """Custom repr to give parent information"""
        return (
            f"<{type(self).__name__}: {self.name} of "
            f"{type(self._parent).__name__}: {self._parent.name}>"
        )

    # Pass any commands to read or write from the instrument up to the parent
    def write(self, cmd: str) -> None:
        return self._parent.write(cmd)

    def write_raw(self, cmd: str) -> None:
        return self._parent.write_raw(cmd)

    def ask(self, cmd: str) -> str:
        return self._parent.ask(cmd)

    def ask_raw(self, cmd: str) -> str:
        return self._parent.ask_raw(cmd)

    @property
    def parent(self) -> InstrumentBase:
        return self._parent

    @property
    def root_instrument(self) -> InstrumentBase:
        return self._parent.root_instrument

    @property
    def name_parts(self) -> list[str]:
        name_parts = list(self._parent.name_parts)
        name_parts.append(self.short_name)
        return name_parts


class InstrumentChannel(InstrumentModule):
    pass


T = TypeVar("T", bound="ChannelTuple")


class ChannelTuple(MetadatableWithName, Sequence[InstrumentModuleType]):
    """
    Container for channelized parameters that allows for sweeps over
    all channels, as well as addressing of individual channels.

    This behaves like a python tuple i.e. it implements the
    :class:`collections.abc.Sequence` interface.

    Args:
        parent: The instrument to which this ChannelTuple
            should be attached.

        name: The name of the ChannelTuple.

        chan_type: The type of channel contained
            within this tuple.

        chan_list: An optional iterable of
            channels of type ``chan_type``.

        snapshotable: Optionally disables taking of snapshots
            for a given ChannelTuple. This is used when objects
            stored inside a ChannelTuple are accessible in multiple
            ways and should not be repeated in an instrument snapshot.

        multichan_paramclass: The class of
            the object to be returned by the :meth:`__getattr__`
            method of :class:`ChannelTuple`.
            Should be a subclass of :class:`.MultiChannelInstrumentParameter`.
            Defaults to :class:`.MultiChannelInstrumentParameter` if None.


    Raises:
        ValueError: If ``chan_type`` is not a subclass of
            :class:`InstrumentChannel`
        ValueError: If ``multichan_paramclass`` is not a subclass of
            :class:`.MultiChannelInstrumentParameter` (note that a class is a
            subclass of itself).

    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        chan_type: type[InstrumentModuleType],
        chan_list: Sequence[InstrumentModuleType] | None = None,
        snapshotable: bool = True,
        multichan_paramclass: type[MultiChannelInstrumentParameter] | None = None,
    ):
        if multichan_paramclass is None:
            multichan_paramclass = MultiChannelInstrumentParameter

        super().__init__()

        self._parent = parent
        self._name = name
        if not isinstance(chan_type, type) or not issubclass(
            chan_type, InstrumentChannel
        ):
            raise ValueError(
                "ChannelTuple can only hold instances of type InstrumentChannel"
            )
        if not isinstance(multichan_paramclass, type) or not issubclass(
            multichan_paramclass, MultiChannelInstrumentParameter
        ):
            raise ValueError(
                "multichan_paramclass must be a (subclass of) "
                "MultiChannelInstrumentParameter"
            )

        self._chan_type = chan_type
        self._snapshotable = snapshotable
        self._paramclass = multichan_paramclass

        self._channel_mapping: dict[str, InstrumentModuleType] = {}
        # provide lookup of channels by name
        # If a list of channels is not provided, define a list to store
        # channels. This will eventually become a locked tuple.
        self._channels: list[InstrumentModuleType]
        if chan_list is None:
            self._channels = []
        else:
            self._channels = list(chan_list)
            self._channel_mapping = {channel.short_name: channel
                                     for channel in self._channels}
            if not all(isinstance(chan, chan_type) for chan in self._channels):
                raise TypeError(
                    f"All items in this ChannelTuple must be of "
                    f"type {chan_type.__name__}."
                )

    @overload
    def __getitem__(self, i: int) -> InstrumentModuleType:
        ...

    @overload
    def __getitem__(self: T, i: slice | tuple[int, ...]) -> T:
        ...

    def __getitem__(
        self: T, i: int | slice | tuple[int, ...]
    ) -> InstrumentModuleType | T:
        """
        Return either a single channel, or a new :class:`ChannelTuple`
        containing only the specified channels

        Args:
            i: Either a single channel index or a slice of channels
              to get
        """
        if isinstance(i, slice):
            return type(self)(
                self._parent,
                self._name,
                self._chan_type,
                self._channels[i],
                multichan_paramclass=self._paramclass,
                snapshotable=self._snapshotable,
            )
        elif isinstance(i, tuple):
            return type(self)(
                self._parent,
                self._name,
                self._chan_type,
                [self._channels[j] for j in i],
                multichan_paramclass=self._paramclass,
                snapshotable=self._snapshotable,
            )
        return self._channels[i]

    def __iter__(self) -> Iterator[InstrumentModuleType]:
        return iter(self._channels)

    def __reversed__(self) -> Iterator[InstrumentModuleType]:
        return reversed(self._channels)

    def __len__(self) -> int:
        return len(self._channels)

    def __contains__(self, item: object) -> bool:
        return item in self._channels

    def __repr__(self) -> str:
        return (
            f"ChannelTuple({self._parent!r}, "
            f"{self._chan_type.__name__}, {self._channels!r})"
        )

    def __add__(self: T, other: ChannelTuple) -> T:
        """
        Return a new ChannelTuple containing the channels from both
        :class:`ChannelTuple` self and r.

        Both ChannelTuple must hold the same type and have the same parent.

        Args:
            other: Right argument to add.
        """
        if not isinstance(self, ChannelTuple) or not isinstance(other, ChannelTuple):
            raise TypeError(
                f"Can't add objects of type"
                f" {type(self).__name__} and {type(other).__name__} together"
            )
        if self._chan_type != other._chan_type:
            raise TypeError(
                f"Both l and r arguments to add must contain "
                f"channels of the same type."
                f" Adding channels of type "
                f"{self._chan_type.__name__} and {other._chan_type.__name__}."
            )
        if self._parent != other._parent:
            raise ValueError("Can only add channels from the same parent "
                             "together.")

        return type(self)(
            self._parent,
            self._name,
            self._chan_type,
            list(self._channels) + list(other._channels),
            snapshotable=self._snapshotable,
        )

    @property
    def short_name(self) -> str:
        return self._name

    @property
    def full_name(self) -> str:
        return "_".join(self.name_parts)

    @property
    def name_parts(self) -> list[str]:
        """
        List of the parts that make up the full name of this function
        """
        if self._parent is not None:
            name_parts = getattr(self._parent, "name_parts", [])
            if name_parts == []:
                # add fallback for the case where someone has bound
                # the function to something that is not an instrument
                # but perhaps it has a name anyway?
                name = getattr(self._parent, "name", None)
                if name is not None:
                    name_parts = [name]
        else:
            name_parts = []

        name_parts.append(self.short_name)
        return name_parts

    # the parameter obj should be called value but that would
    # be an incompatible change
    def index(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self,
        obj: InstrumentModuleType,
        start: int = 0,
        stop: int = sys.maxsize,
    ) -> int:
        """
        Return the index of the given object

        Args:
            obj: The object to find in the channel list.
            start: Index to start searching from.
            stop: Index to stop searching at.
        """
        return self._channels.index(obj, start, stop)

    def count(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self, obj: InstrumentModuleType
    ) -> int:
        """Returns number of instances of the given object in the list

        Args:
            obj: The object to find in the ChannelTuple.
        """
        return self._channels.count(obj)

    def get_channel_by_name(self: T, *names: str) -> InstrumentModuleType | T:
        """
        Get a channel by name, or a ChannelTuple if multiple names are given.

        Args:
            *names: channel names
        """
        if len(names) == 0:
            raise Exception("one or more names must be given")
        if len(names) == 1:
            return self._channel_mapping[names[0]]
        selected_channels = tuple(self._channel_mapping[name] for name in names)
        return type(self)(
            self._parent,
            self._name,
            self._chan_type,
            selected_channels,
            self._snapshotable,
            self._paramclass,
        )

    def get_validator(self) -> ChannelTupleValidator:
        """
        Returns a validator that checks that the returned object is a channel
        in this ChannelTuple
        """
        return ChannelTupleValidator(self)

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`.NumpyJSONEncoder` supports).

        Args:
            update: If True, update the state by querying the
                instrument. If None only update if the state is known to be
                invalid. If False, just use the latest values in memory
                and never update.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac). If you want to skip the
                update of certain parameters in all snapshots, use the
                ``snapshot_get``  attribute of those parameters instead.

        Returns:
            dict: base snapshot
        """
        if self._snapshotable:
            snap = {'channels': {chan.name: chan.snapshot(update=update)
                                     for chan in self._channels},
                    'snapshotable': self._snapshotable,
                    '__class__': full_class(self),
                    }
        else:
            snap = {'snapshotable': self._snapshotable,
                    '__class__': full_class(self),
                    }
        return snap

    def __getattr__(
        self, name: str
    ) -> (MultiChannelInstrumentParameter | Callable[..., None] | InstrumentModuleType):
        """
        Look up an attribute by name. If this is the name of a parameter or
        a function on the channel type contained in this container return a
        multi-channel function or parameter that can be used to get or
        set all items in a channel list simultaneously. If this is the
        name of a channel, return that channel.

        Args:
            name: The name of the parameter, function or channel that we want to
                operate on.
        """
        if len(self) > 0:
            # Check if this is a valid parameter
            if name in self._channels[0].parameters:
                param = self._construct_multiparam(name)
                return param

            # Check if this is a valid function
            if name in self._channels[0].functions:
                # We want to return a reference to a function that would call the
                # function for each of the channels in turn.
                def multi_func(*args: Any) -> None:
                    for chan in self._channels:
                        chan.functions[name](*args)

                return multi_func

            # check if this is a method on the channels in the
            # sequence
            maybe_callable = getattr(self._channels[0], name, None)
            if callable(maybe_callable):

                def multi_callable(*args: Any) -> None:
                    for chan in self._channels:
                        getattr(chan, name)(*args)

                return multi_callable

        try:
            return self._channel_mapping[name]
        except KeyError:
            pass

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _construct_multiparam(self, name: str) -> MultiChannelInstrumentParameter:
        setpoints = None
        setpoint_names = None
        setpoint_labels = None
        setpoint_units = None
        # We need to construct a MultiParameter object to get each of the
        # values our of each parameter in our list, we don't currently try
        # to construct a multiparameter from a list of multi parameters
        if isinstance(self._channels[0].parameters[name], MultiParameter):
            raise NotImplementedError(
                "Slicing is currently not supported for MultiParameters"
            )
        parameters = cast(
            list[Union[Parameter, ArrayParameter]],
            [chan.parameters[name] for chan in self._channels],
        )
        names = tuple(f"{chan.name}_{name}" for chan in self._channels)
        labels = tuple(parameter.label for parameter in parameters)
        units = tuple(parameter.unit for parameter in parameters)
        if isinstance(parameters[0], ArrayParameter):
            arrayparameters = cast(list[ArrayParameter], parameters)
            shapes = tuple(parameter.shape for parameter in arrayparameters)
            if arrayparameters[0].setpoints:
                setpoints = tuple(parameter.setpoints for parameter in arrayparameters)
            if arrayparameters[0].setpoint_names:
                setpoint_names = tuple(
                    parameter.setpoint_names for parameter in arrayparameters
                )
            if arrayparameters[0].setpoint_labels:
                setpoint_labels = tuple(
                    parameter.setpoint_labels for parameter in arrayparameters
                )
            if arrayparameters[0].setpoint_units:
                setpoint_units = tuple(
                    parameter.setpoint_units for parameter in arrayparameters
                )
        else:
            shapes = tuple(() for _ in self._channels)
        param = self._paramclass(
            self._channels,
            param_name=name,
            name=f"Multi_{name}",
            names=names,
            shapes=shapes,
            instrument=self._parent,
            labels=labels,
            units=units,
            setpoints=setpoints,
            setpoint_names=setpoint_names,
            setpoint_units=setpoint_units,
            setpoint_labels=setpoint_labels,
            bind_to_instrument=False,
        )
        return param

    def __dir__(self) -> list[Any]:
        names = list(super().__dir__())
        if self._channels:
            names += list(self._channels[0].parameters.keys())
            names += list(self._channels[0].functions.keys())
            names += [channel.short_name for channel in self._channels]
        return sorted(set(names))

    def print_readable_snapshot(self, update: bool = False,
                                max_chars: int = 80) -> None:
        if self._snapshotable:
            for channel in self._channels:
                channel.print_readable_snapshot(update=update,
                                                max_chars=max_chars)

    def invalidate_cache(self) -> None:
        """
        Invalidate the cache of all parameters on the ChannelTuple.
        """
        for chan in self._channels:
            chan.invalidate_cache()

# we ignore a mypy error here since the __getitem__ signature above
# taking a tuple is not compatible with MutableSequence
# for some reason this does not happen with Sequence
class ChannelList(ChannelTuple, MutableSequence[InstrumentModuleType]):  # type: ignore[misc]
    """
    Mutable Container for channelized parameters that allows for sweeps over
    all channels, as well as addressing of individual channels.

    This behaves like a python list i.e. it implements the
    :class:`collections.abc.MutableSequence` interface.

    Note it may be useful to use the mutable ChannelList while constructing it.
    E.g. adding channels as they are created, but in most use cases it is recommended
    to convert this to a :class:`ChannelTuple` before adding it to an instrument.
    This can be done using the :meth:`to_channel_tuple` method.

    Args:
        parent: The instrument to which this :class:`ChannelList`
            should be attached.

        name: The name of the :class:`ChannelList`.

        chan_type: The type of channel contained
            within this list.

        chan_list: An optional iterable of
            channels of type ``chan_type``.  This will create a list and
            immediately lock the :class:`ChannelList`.

        snapshotable: Optionally disables taking of snapshots
            for a given ChannelList. This is used when objects
            stored inside a ChannelList are accessible in multiple
            ways and should not be repeated in an instrument snapshot.

        multichan_paramclass: The class of
            the object to be returned by the :meth:`__getattr__`
            method of :class:`ChannelList`.
            Should be a subclass of :class:`.MultiChannelInstrumentParameter`.
            Defaults to :class:`.MultiChannelInstrumentParameter` if None.

    Raises:
        ValueError: If ``chan_type`` is not a subclass of
            :class:`InstrumentChannel`
        ValueError: If ``multichan_paramclass`` is not a subclass of
            :class:`.MultiChannelInstrumentParameter` (note that a class is a
            subclass of itself).

    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        chan_type: type[InstrumentModuleType],
        chan_list: Sequence[InstrumentModuleType] | None = None,
        snapshotable: bool = True,
        multichan_paramclass: type[MultiChannelInstrumentParameter] | None = None,
    ):
        if multichan_paramclass is None:
            multichan_paramclass = MultiChannelInstrumentParameter
        super().__init__(
            parent, name, chan_type, chan_list, snapshotable, multichan_paramclass
        )
        if len(self._channels) > 0:
            self._locked = True
        else:
            self._locked = False

    @overload
    def __delitem__(self, key: int) -> None:
        ...

    @overload
    def __delitem__(self, key: slice) -> None:
        ...

    def __delitem__(self, key: int | slice) -> None:
        if self._locked:
            raise AttributeError("Cannot delete from a locked channel list")
        self._channels.__delitem__(key)
        self._channel_mapping = {
            channel.short_name: channel for channel in self._channels
        }

    @overload
    def __setitem__(self, index: int, value: InstrumentModuleType) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[InstrumentModuleType]) -> None:
        ...

    def __setitem__(
        self,
        index: int | slice,
        value: InstrumentModuleType | Iterable[InstrumentModuleType],
    ) -> None:
        if self._locked:
            raise AttributeError("Cannot set item in a locked channel list")
        # update mapping
        # asserts added to work around https://github.com/python/mypy/issues/7858
        if isinstance(index, int):
            assert isinstance(value, InstrumentModule)
            self._channels[index] = value
        else:
            assert not isinstance(value, InstrumentModule)
            self._channels[index] = value
        self._channel_mapping = {
            channel.short_name: channel for channel in self._channels
        }

    def append(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self, obj: InstrumentModuleType
    ) -> None:
        """
        Append a Channel to this list. Requires that the ChannelList is not
        locked and that the channel is of the same type as the ones in the list.

        Args:
            obj: New channel to add to the list.
        """
        if self._locked:
            raise AttributeError("Cannot append to a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError(
                f"All items in a channel list must be of the same "
                f"type. Adding {type(obj).__name__} to a "
                f"list of {self._chan_type.__name__}."
            )
        self._channel_mapping[obj.short_name] = obj
        self._channels.append(obj)

    def clear(self) -> None:
        """
        Clear all items from the ChannelList.
        """
        if self._locked:
            raise AttributeError("Cannot clear a locked ChannelList")
        # when not locked the _channels seq is a list
        self._channels.clear()
        self._channel_mapping.clear()

    def remove(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self, obj: InstrumentModuleType
    ) -> None:
        """
        Removes obj from ChannelList if not locked.

        Args:
            obj: Channel to remove from the list.
        """
        if self._locked:
            raise AttributeError("Cannot remove from a locked channel list")
        else:
            self._channels.remove(obj)
            self._channel_mapping.pop(obj.short_name)

    def extend(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self, objects: Iterable[InstrumentModuleType]
    ) -> None:
        """
        Insert an iterable of objects into the list of channels.

        Args:
            objects: A list of objects to add into the
              :class:`ChannelList`.
        """
        # objects may be a generator but we need to iterate over it twice
        # below so copy it into a tuple just in case.
        if self._locked:
            raise AttributeError("Cannot extend a locked channel list")
        objects_tuple = tuple(objects)
        if not all(isinstance(obj, self._chan_type) for obj in objects_tuple):
            raise TypeError("All items in a channel list must be of the same type.")
        self._channels.extend(objects_tuple)
        self._channel_mapping.update({obj.short_name: obj for obj in objects_tuple})

    def insert(  #  pyright: ignore[reportIncompatibleMethodOverride]
        self, index: int, obj: InstrumentModuleType
    ) -> None:
        """
        Insert an object into the ChannelList at a specific index.

        Args:
            index: Index to insert object.
            obj: Object of type chan_type to insert.
        """
        if self._locked:
            raise AttributeError("Cannot insert into a locked channel list")
        if not isinstance(obj, self._chan_type):
            raise TypeError(
                f"All items in a channel list must be of the same "
                f"type. Adding {type(obj).__name__} to a list of {self._chan_type.__name__}."
            )
        self._channels.insert(index, obj)
        self._channel_mapping[obj.short_name] = obj

    def get_validator(self) -> ChannelTupleValidator:
        """
        Returns a validator that checks that the returned object is a channel
        in this ChannelList.

        Raises:
            AttributeError: If the ChannelList is not locked.
        """
        if not self._locked:
            raise AttributeError(
                "Cannot create a validator for an unlocked ChannelList"
            )
        return super().get_validator()

    def lock(self) -> None:
        """
        Lock the channel list. Once this is done, the ChannelList is
        locked and any future changes to the list are prevented.
        Note this is not recommended and may be deprecated in the future.
        Use ``to_channel_tuple`` to convert this into a tuple instead.
        """
        if self._locked:
            return
        self._locked = True

    def to_channel_tuple(self) -> ChannelTuple:
        """
        Returns a ChannelTuple build from this ChannelList containing the
        same channels but without the ability to be modified.
        """
        return ChannelTuple(
            self._parent,
            self._name,
            self._chan_type,
            self._channels,
            multichan_paramclass=self._paramclass,
            snapshotable=self._snapshotable,
        )

    def __repr__(self) -> str:
        return (
            f"ChannelList({self._parent!r}, "
            f"{self._chan_type.__name__}, {self._channels!r})"
        )


class ChannelTupleValidator(Validator[InstrumentChannel]):
    """
    A validator that checks that the returned object is a member of the
    ChannelTuple with which the validator was constructed.

    This class will not normally be created directly, but created from a channel
    list using the ``ChannelTuple.get_validator`` method.

    Args:
        channel_list: the ChannelTuple that should be checked
            against. The channel list must be locked and populated before it
            can be used to construct a validator.
    """

    def __init__(self, channel_list: ChannelTuple) -> None:
        # Save the base parameter list
        if not isinstance(channel_list, ChannelTuple):
            raise ValueError(
                "channel_list must be a ChannelTuple "
                "object containing the "
                "channels that should be validated"
            )
        if isinstance(channel_list, ChannelList) and not channel_list._locked:
            raise AttributeError(
                "channel_list must be locked before it can "
                "be used to create a validator"
            )
        self._channel_list = channel_list

    def validate(self, value: InstrumentChannel, context: str = '') -> None:
        """
        Checks to see that value is a member of the ChannelTuple referenced by
        this validator

        Args:
            value: the value to be checked against the
                reference channel list.
            context: the context of the call, used as part of the exception
                raised.
        """
        if value not in self._channel_list:
            raise ValueError(
                f"{value!r} is not part of the expected channel list; {context}"
            )


class ChannelListValidator(ChannelTupleValidator):
    """Alias for backwards compatibility. Do not use"""
    pass


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
        cls,
        parent: Instrument,
        channel_list: AutoLoadableChannelList | None = None,
        **kwargs: Any,
    ) -> list[AutoLoadableInstrumentChannel]:
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
        cls, parent: Instrument, **kwargs: Any
    ) -> list[dict[Any, Any]]:
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
        cls,
        parent: Instrument,
        create_on_instrument: bool = True,
        channel_list: AutoLoadableChannelList | None = None,
        **kwargs: Any,
    ) -> AutoLoadableInstrumentChannel:
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
    def _get_new_instance_kwargs(
        cls, parent: Instrument | None = None, **kwargs: Any
    ) -> dict[Any, Any]:
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
            A keyword argument dictionary with at least a ``name`` key which is
            unique on the instrument. The parent instrument is passed as an
            argument in this function so we can query if the generated name is
            indeed unique.

        Notes:
            The init arguments ``parent`` and ``channel_list`` are automatically
            added by the ``new_instance`` method and should not be added in the
            kwarg dictionary returned here. Additionally, the argument
            ``existence`` either needs to be omitted or be False.
        """
        raise NotImplementedError(
            "Please subclass and implement this method in the subclass")

    def __init__(
        self,
        parent: Instrument | InstrumentChannel,
        name: str,
        exists_on_instrument: bool = False,
        channel_list: AutoLoadableChannelList | None = None,
        **kwargs: Any,
    ):
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
            **kwargs: Keyword passed to the super class.
        """
        super().__init__(parent, name=name, **kwargs)
        self._exists_on_instrument = exists_on_instrument
        self._channel_list = channel_list

    def create(self) -> None:
        """Create the channel on the instrument"""
        if self._exists_on_instrument:
            raise RuntimeError("Channel already exists on instrument")

        self._create()
        self._exists_on_instrument = True

    def _create(self) -> None:
        """
        (SCPI) commands needed to create the channel. Note that we
        need to use ``self.root_instrument.write`` to send commands,
        because ``self.write`` will cause ``_assert_existence`` to raise a
        runtime error.
        """
        raise NotImplementedError("Please subclass")

    def remove(self) -> None:
        """
        Delete the channel from the instrument and remove from channel list
        """
        self._assert_existence()
        self._remove()
        if self._channel_list is not None and self in self._channel_list:
            self._channel_list.remove(self)
        self._exists_on_instrument = False

    def _remove(self) -> None:
        """
        (SCPI) commands needed to delete the channel from the instrument
        """
        raise NotImplementedError("Please subclass")

    def _assert_existence(self) -> None:
        if not self._exists_on_instrument:
            raise RuntimeError(
                "Object does not exist (anymore) on the instrument")

    def write(self, cmd: str) -> None:
        """
        Write to the instrument only if the channel is present on the instrument
        """
        self._assert_existence()
        return super().write(cmd)

    def ask(self, cmd: str) -> str:
        """
        Ask the instrument only if the channel is present on the instrument
        """
        self._assert_existence()
        return super().ask(cmd)

    @property
    def exists_on_instrument(self) -> bool:
        return self._exists_on_instrument


class AutoLoadableChannelList(ChannelList):
    """
    Extends the QCoDeS :class:`ChannelList` class to add the following features:
    - Automatically create channel objects on initialization
    - Make a ``add`` method to create channel objects

    Args:
        parent: the instrument to which this channel
            should be attached

        name: the name of the channel list

        chan_type: the type of channel contained
            within this list

        chan_list: An optional iterable of
            channels of type chan_type.  This will create a list and
            immediately lock the :class:`ChannelList`.

        snapshotable: Optionally disables taking of snapshots
            for a given channel list.  This is used when objects
            stored inside a channel list are accessible in multiple
            ways and should not be repeated in an instrument snapshot.

        multichan_paramclass: The class of
            the object to be returned by the
            :class:`ChannelList` ``__getattr__`` method.
            Should be a subclass of :class:`MultiChannelInstrumentParameter`.

        **kwargs: Keyword arguments to be passed to the ``load_from_instrument``
            method of the channel class. Note that the kwargs are *NOT* passed
            to the ``__init__`` of the super class.

    Raises:
        ValueError: If :class:`chan_type` is not a subclass of
            :class:`InstrumentChannel`
        ValueError: If ``multichan_paramclass`` is not a subclass of
            :class:`MultiChannelInstrumentParameter` (note that a class is a
            subclass of itself).
    """
    def __init__(
        self,
        parent: Instrument,
        name: str,
        chan_type: type,
        chan_list: Sequence[AutoLoadableInstrumentChannel] | None = None,
        snapshotable: bool = True,
        multichan_paramclass: type = MultiChannelInstrumentParameter,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            parent, name, chan_type, chan_list, snapshotable,
            multichan_paramclass
        )
        new_channels = self._chan_type.load_from_instrument(  # type: ignore[attr-defined]
            self._parent, channel_list=self, **kwargs)

        self.extend(new_channels)

    def add(self, **kwargs: Any) -> AutoLoadableInstrumentChannel:
        """
        Add a channel to the list

        Args:
            kwargs: Keyword arguments passed to the ``new_instance`` method of
                the channel class

        Returns:
            Newly created instance of the channel class
        """
        new_channel = self._chan_type.new_instance(  # type: ignore[attr-defined]
            self._parent,
            create_on_instrument=True,
            channel_list=self,
            **kwargs
        )

        self.append(new_channel)
        return new_channel
