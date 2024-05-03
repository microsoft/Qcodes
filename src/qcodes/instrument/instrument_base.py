"""Base class for Instrument and InstrumentModule"""
from __future__ import annotations

import collections.abc
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from typing_extensions import TypedDict, TypeVar, deprecated

from qcodes.logger import get_instrument_logger
from qcodes.metadatable import Metadatable, MetadatableWithName
from qcodes.parameters import Function, Parameter, ParameterBase
from qcodes.utils import DelegateAttributes, full_class

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from typing_extensions import NotRequired

    from qcodes.instrument.channel import ChannelTuple, InstrumentModule
    from qcodes.logger.instrument_logger import InstrumentLoggerAdapter

from qcodes.utils import QCoDeSDeprecationWarning

log = logging.getLogger(__name__)

TParameter = TypeVar("TParameter", bound=ParameterBase, default=Parameter)


class InstrumentBaseKWArgs(TypedDict):
    """
    This TypedDict defines the type of the kwargs that can be passed to the InstrumentBase class.
    A subclass of VisaInstrument should take ``**kwargs: Unpack[InstrumentBaseKWArgs]`` as input
    and forward this to the super class to ensure that it can accept all the arguments defined here.
    """

    metadata: NotRequired[Mapping[Any, Any] | None]
    """
    Additional static metadata to add to this
    instrument's JSON snapshot.
    """
    label: NotRequired[str | None]
    """
    Nicely formatted name of the instrument; if None,
    the ``name`` is used.
    """

class InstrumentBase(MetadatableWithName, DelegateAttributes):
    """
    Base class for all QCodes instruments and instrument channels

    Args:
        name: an identifier for this instrument, particularly for
            attaching it to a Station.
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.
        label: nicely formatted name of the instrument; if None, the
            ``name`` is used.
    """

    def __init__(
        self,
        name: str,
        metadata: Mapping[Any, Any] | None = None,
        label: str | None = None,
    ) -> None:
        name = self._replace_hyphen(name)
        self._short_name = name
        self._is_valid_identifier(self.full_name)

        self.label = name if label is None else label
        self._label: str

        self.parameters: dict[str, ParameterBase] = {}
        """
        All the parameters supported by this instrument.
        Usually populated via :py:meth:`add_parameter`.
        """
        self.functions: dict[str, Function] = {}
        """
        All the functions supported by this
        instrument. Usually populated via :py:meth:`add_function`.
        """
        self.submodules: dict[str, InstrumentModule | ChannelTuple] = {}
        """
        All the submodules of this instrument
        such as channel lists or logical groupings of parameters.
        Usually populated via :py:meth:`add_submodule`.
        """
        self.instrument_modules: dict[str, InstrumentModule] = {}
        """
        All the :class:`InstrumentModule` of this instrument
        Usually populated via :py:meth:`add_submodule`.
        """

        self._channel_lists: dict[str, ChannelTuple] = {}
        """
        All the ChannelTuples of this instrument
        Usually populated via :py:meth:`add_submodule`.
        This is private until the correct name has been decided.
        """

        super().__init__(metadata)

        # This is needed for snapshot method to work
        self._meta_attrs = ["name", "label"]

        self.log: InstrumentLoggerAdapter = get_instrument_logger(self, __name__)

    @property
    def label(self) -> str:
        """
        Nicely formatted label of the instrument.
        """
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label

    def add_parameter(
        self,
        name: str,
        parameter_class: type[TParameter] | None = None,
        **kwargs: Any,
    ) -> TParameter:
        """
        Bind one Parameter to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get.

        Args:
            name: How the parameter will be stored within
                :attr:`.parameters` and also how you address it using the
                shortcut methods: ``instrument.set(param_name, value)`` etc.

            parameter_class: You can construct the parameter
                out of any class. Default :class:`.parameters.Parameter`.

            **kwargs: Constructor arguments for ``parameter_class``.

        Raises:
            KeyError: If this instrument already has a parameter with this
                name and the parameter being replaced is not an abstract
                parameter.

            ValueError: If there is an existing abstract parameter and the
                unit of the new parameter is inconsistent with the existing
                one.
        """
        if parameter_class is None:
            parameter_class = cast(type[TParameter], Parameter)

        if "bind_to_instrument" not in kwargs.keys():
            kwargs["bind_to_instrument"] = True

        try:
            param = parameter_class(name=name, instrument=self, **kwargs)
        except TypeError:
            kwargs.pop("bind_to_instrument")
            warnings.warn(
                f"Parameter {name} on instrument {self.name} does "
                f"not correctly pass kwargs to its baseclass. A "
                f"Parameter class must take `**kwargs` and forward "
                f"them to its baseclass.",
                QCoDeSDeprecationWarning,
            )
            param = parameter_class(name=name, instrument=self, **kwargs)

        existing_parameter = self.parameters.get(name, None)
        if not existing_parameter:
            warnings.warn(
                f"Parameter {name} did not correctly register itself on instrument"
                f" {self.name}. Please check that `instrument` argument is passed "
                f"from {parameter_class!r} all the way to `ParameterBase`. "
                "This will be an error in the future.",
                QCoDeSDeprecationWarning,
            )
            self.parameters[name] = param
        return param

    def add_function(self, name: str, **kwargs: Any) -> None:
        """
        Bind one ``Function`` to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real function of the instrument.

        This functionality is meant for simple cases, principally things that
        map to simple commands like ``*RST`` (reset) or those with just a few
        arguments. It requires a fixed argument count, and positional args
        only.

        Note:
            We do not recommend the usage of Function for any new driver.
            Function does not add any significant features over a method
            defined on the class.

        Args:
            name: How the Function will be stored within
                ``instrument.Functions`` and also how you  address it using the
                shortcut methods: ``instrument.call(func_name, *args)`` etc.
            **kwargs: constructor kwargs for ``Function``

        Raises:
            KeyError: If this instrument already has a function with this
                name.
        """
        if name in self.functions:
            raise KeyError(f"Duplicate function name {name}")
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

    def add_submodule(
        self, name: str, submodule: InstrumentModule | ChannelTuple
    ) -> None:
        """
        Bind one submodule to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        method for every submodule of the instrument.

        Submodules can effectively be considered as instruments within
        the main instrument, and should at minimum be
        snapshottable. For example, they can be used to either store
        logical groupings of parameters, which may or may not be
        repeated, or channel lists. They should either be an instance
        of an ``InstrumentModule`` or a ``ChannelTuple``.

        Args:
            name: How the submodule will be stored within
                ``instrument.submodules`` and also how it can be
                addressed.
            submodule: The submodule to be stored.

        Raises:
            KeyError: If this instrument already contains a submodule with this
                name.
            TypeError: If the submodule that we are trying to add is
                not an instance of an ``Metadatable`` object.
        """
        if name in self.submodules:
            raise KeyError(f"Duplicate submodule name {name}")
        if not isinstance(submodule, Metadatable):
            raise TypeError("Submodules must be metadatable.")
        self.submodules[name] = submodule

        if isinstance(submodule, collections.abc.Sequence):
            # this is channel_list like:
            # We cannot check against ChannelsList itself since that
            # would introduce a circular dependency.
            self._channel_lists[name] = submodule
        else:
            self.instrument_modules[name] = submodule

    def get_component(self, full_name: str) -> MetadatableWithName:
        """
        Recursively get a component of the instrument by full_name.

        Args:
            full_name: The name of the component to get.

        Returns:
            The component with the given name.

        Raises:
            KeyError: If the component does not exist.
        """
        name_parts = full_name.split("_")
        name_parts.reverse()

        component = self._get_component_by_name(name_parts.pop(), name_parts)
        return component

    def _get_component_by_name(
        self, potential_top_level_name: str, remaining_name_parts: list[str]
    ) -> MetadatableWithName:

        log.debug(
            "trying to find component %s on %s, remaining %s",
            potential_top_level_name,
            self.full_name,
            remaining_name_parts,
        )
        component: MetadatableWithName | None = None

        sub_component_name_map = {
            sub_component.short_name: sub_component
            for sub_component in self.submodules.values()
        }

        channel_name_map: dict[str, InstrumentModule] = {}
        for channel_list in self._channel_lists.values():
            local_channels_name_map: dict[str, InstrumentModule] = {
                channel.short_name: channel for channel in channel_list
            }
            channel_name_map.update(local_channels_name_map.items())

        if potential_top_level_name in self.parameters:
            component = self.parameters[potential_top_level_name]
        elif potential_top_level_name in self.functions:
            component = self.functions[potential_top_level_name]
        elif potential_top_level_name in self.submodules:
            # recursive call on found component
            component = self.submodules[potential_top_level_name]
            if len(remaining_name_parts) > 0:
                remaining_name_parts.reverse()
                remaining_name = "_".join(remaining_name_parts)
                component = component.get_component(remaining_name)
                remaining_name_parts = []
        elif potential_top_level_name in sub_component_name_map:
            component = sub_component_name_map[potential_top_level_name]
            if len(remaining_name_parts) > 0:
                remaining_name_parts.reverse()
                remaining_name = "_".join(remaining_name_parts)
                component = component.get_component(remaining_name)
                remaining_name_parts = []
        elif potential_top_level_name in channel_name_map:
            component = channel_name_map[potential_top_level_name]
            if len(remaining_name_parts) > 0:
                remaining_name_parts.reverse()
                remaining_name = "_".join(remaining_name_parts)
                component = component.get_component(remaining_name)
                remaining_name_parts = []

        if component is not None:
            if len(remaining_name_parts) == 0:
                return component

        if len(remaining_name_parts) == 0:
            raise KeyError(
                f"Found component {self.full_name} but could not "
                f"match {potential_top_level_name} part."
            )

        new_potential_top_level_name = (
            f"{potential_top_level_name}_{remaining_name_parts.pop()}"
        )

        component = self._get_component_by_name(
            new_potential_top_level_name, remaining_name_parts
        )

        return component

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`.NumpyJSONEncoder`
        supports).

        Args:
            update: If ``True``, update the state by querying the
                instrument. If None update the state if known to be invalid.
                If ``False``, just use the latest values in memory and never
                update state.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac). If you want to skip the
                update of certain parameters in all snapshots, use the
                ``snapshot_get`` attribute of those parameters instead.

        Returns:
            dict: base snapshot
        """

        if params_to_skip_update is None:
            params_to_skip_update = []

        snap: dict[str, Any] = {
            "functions": {
                name: func.snapshot(update=update)
                for name, func in self.functions.items()
            },
            "submodules": {
                name: subm.snapshot(update=update)
                for name, subm in self.submodules.items()
            },
            "parameters": {},
            "__class__": full_class(self),
        }

        for name, param in self.parameters.items():
            if param.snapshot_exclude:
                continue
            if params_to_skip_update and name in params_to_skip_update:
                update_par: bool | None = False
            else:
                update_par = update
            try:
                snap["parameters"][name] = param.snapshot(update=update_par)
            except Exception:
                # really log this twice. Once verbose for the UI and once
                # at lower level with more info for file based loggers
                self.log.warning("Snapshot: Could not update parameter: %s", name)
                self.log.info("Details for Snapshot:", exc_info=True)
                snap["parameters"][name] = param.snapshot(update=False)

        for attr in set(self._meta_attrs):
            val = getattr(self, attr, None)
            if val is not None:
                if isinstance(val, Metadatable):
                    snap[attr] = val.snapshot(update=update)
                else:
                    snap[attr] = val

        return snap

    def print_readable_snapshot(
        self, update: bool = False, max_chars: int = 80
    ) -> None:
        """
        Prints a readable version of the snapshot.
        The readable snapshot includes the name, value and unit of each
        parameter.
        A convenience function to quickly get an overview of the
        status of an instrument.

        Args:
            update: If ``True``, update the state by querying the
                instrument. If ``False``, just use the latest values in memory.
                This argument gets passed to the snapshot function.
            max_chars: the maximum number of characters per line. The
                readable snapshot will be cropped if this value is exceeded.
                Defaults to 80 to be consistent with default terminal width.
        """
        floating_types = (float, np.integer, np.floating)
        snapshot = self.snapshot(update=update)

        par_lengths = [len(p) for p in snapshot["parameters"]]
        # handle the case of no parameters
        par_lengths = par_lengths or [0]

        # Min of 50 is to prevent a super long parameter name to break this
        # function
        par_field_len = min(max(par_lengths) + 1, 50)

        print(self.name + ":")
        print("{0:<{1}}".format("\tparameter ", par_field_len) + "value")
        print("-" * max_chars)
        for par in sorted(snapshot["parameters"]):
            name = snapshot["parameters"][par]["name"]
            msg = "{0:<{1}}:".format(name, par_field_len)

            # in case of e.g. ArrayParameters, that usually have
            # snapshot_value == False, the parameter may not have
            # a value in the snapshot
            val = snapshot["parameters"][par].get("value", "Not available")

            unit = snapshot["parameters"][par].get("unit", None)
            if unit is None:
                # this may be a multi parameter
                unit = snapshot["parameters"][par].get("units", None)
            if isinstance(val, floating_types):
                msg += f"\t{val:.5g} "
                # numpy float and int types format like builtins
            else:
                msg += f"\t{val} "
            if unit != "":  # corresponds to no unit
                msg += f"({unit})"
            # Truncate the message if it is longer than max length
            if len(msg) > max_chars and not max_chars == -1:
                msg = msg[0 : max_chars - 3] + "..."
            print(msg)

        for submodule in self.submodules.values():
            submodule.print_readable_snapshot(update=update, max_chars=max_chars)

    def invalidate_cache(self) -> None:
        """
        Invalidate the cache of all parameters on the instrument.
        Calling this method will recursively mark the cache of all parameters
        on the instrument and any parameter on instrument modules as invalid.

        This is useful if you have performed manual operations
        (e.g. using the frontpanel)
        which changes the state of the instrument outside QCoDeS.

        This in turn means that the next snapshot of the instrument will trigger
        a (potentially slow) reread of all parameters of the instrument if you pass
        `update=None` to snapshot.
        """
        for parameter in self.parameters.values():
            parameter.cache.invalidate()

        for submodule in self.submodules.values():
            submodule.invalidate_cache()

    @property
    def parent(self) -> InstrumentBase | None:
        """
        The parent instrument. By default, this is ``None``.
        Any SubInstrument should subclass this to return the parent instrument.
        """
        return None

    @property
    def ancestors(self) -> tuple[InstrumentBase, ...]:
        """
        Ancestors in the form of a list of :class:`InstrumentBase`

        The list starts with the current module
        then the parent and the parents parent
        until the root instrument is reached.
        """
        if self.parent is not None:
            return (self,) + self.parent.ancestors
        else:
            return (self,)

    @property
    def root_instrument(self) -> InstrumentBase:
        """
        The topmost parent of this module.

        For the ``root_instrument`` this is ``self``.
        """

        return self

    @property
    def name_parts(self) -> list[str]:
        """
        A list of all the parts of the instrument name from :meth:`root_instrument`
        to the current :class:`InstrumentModule`.
        """
        return [self.short_name]

    @property
    def full_name(self) -> str:
        """
        Full name of the instrument.

        For an :class:`InstrumentModule` this includes
        all parents separated by ``_``
        """
        return "_".join(self.name_parts)

    @property
    def name(self) -> str:
        """
        Full name of the instrument

        This is equivalent to :meth:`full_name` for backwards compatibility.
        """
        return self.full_name

    @property
    @deprecated(
        "The private attribute `_name` is deprecated and will be removed. Use `full_name` instead.",
        category=QCoDeSDeprecationWarning,
    )
    def _name(self) -> str:
        """
        Private alias kept here for backwards compatibility
        see https://github.com/zhinst/zhinst-qcodes/issues/27
        """
        return self.full_name

    @property
    def short_name(self) -> str:
        """
        Short name of the instrument.

        For an :class:`InstrumentModule` this does
        not include any parent names.
        """
        return self._short_name

    @staticmethod
    def _is_valid_identifier(name: str) -> None:
        """Check whether given name is a valid instrument identifier."""
        if not name.isidentifier():
            raise ValueError(f"{name} invalid instrument identifier")

    @staticmethod
    def _replace_hyphen(name: str) -> str:
        """Replace - in name with _ and warn if any is found."""
        new_name = str(name).replace("-", "_")
        if name != new_name:
            warnings.warn(f"Changed {name} to {new_name} for instrument identifier")

        return new_name

    def _is_abstract(self) -> bool:
        """
        This method is run after the initialization of an instrument but
        before the instrument is registered. It recursively checks that there are
        no abstract parameters defined on the instrument or any instrument channels.
        """
        is_abstract = False
        abstract_parameters = [
            parameter.name
            for parameter in self.parameters.values()
            if parameter.abstract
        ]
        if any(abstract_parameters):
            is_abstract = True

        for submodule in self.instrument_modules.values():
            if submodule._is_abstract():
                is_abstract = True

        for chanel_list in self._channel_lists.values():
            for channel in chanel_list:
                if channel._is_abstract():
                    is_abstract = True

        return is_abstract

    #
    # shortcuts to parameters & setters & getters                           #
    #
    # instrument['someparam'] === instrument.parameters['someparam']        #
    # instrument.someparam === instrument.parameters['someparam']           #
    # instrument.get('someparam') === instrument['someparam'].get()         #
    # etc...                                                                #
    #
    delegate_attr_dicts: ClassVar[list[str]] = ["parameters", "functions", "submodules"]

    def __getitem__(self, key: str) -> Callable[..., Any] | Parameter:
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def set(self, param_name: str, value: Any) -> None:
        """
        Shortcut for setting a parameter from its name and new value.

        Args:
            param_name: The name of a parameter of this instrument.
            value: The new value to set.
        """
        self.parameters[param_name].set(value)

    def get(self, param_name: str) -> Any:
        """
        Shortcut for getting a parameter from its name.

        Args:
            param_name: The name of a parameter of this instrument.

        Returns:
            The current value of the parameter.
        """
        return self.parameters[param_name].get()

    def call(self, func_name: str, *args: Any) -> Any:
        """
        Shortcut for calling a function from its name.

        Args:
            func_name: The name of a function of this instrument.
            *args: any arguments to the function.

        Returns:
            The return value of the function.
        """
        return self.functions[func_name].call(*args)

    def __getstate__(self) -> None:
        """Prevent pickling instruments, and give a nice error message."""
        raise RuntimeError(
            f"Error when pickling instrument {self.name}. "
            f"QCoDeS instruments can not be pickled."
        )

    def validate_status(self, verbose: bool = False) -> None:
        """Validate the values of all gettable parameters

        The validation is done for all parameters that have both a get and
        set method.

        Arguments:
            verbose: If ``True``, then information about the
                parameters that are being check is printed.

        """
        for k, p in self.parameters.items():
            if p.gettable and p.settable:
                value = p.get()
                if verbose:
                    print(f"validate_status: param {k}: {value}")
                p.validate(value)
