"""Instrument base class."""
import logging
import time
import warnings
import weakref
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import numpy as np

from qcodes.logger.instrument_logger import get_instrument_logger
from qcodes.utils.helpers import DelegateAttributes, full_class, strip_attrs
from qcodes.utils.metadata import Metadatable
from qcodes.utils.validators import Anything

from .function import Function
from .parameter import Parameter, _BaseParameter

if TYPE_CHECKING:
    from qcodes.instrument.channel import ChannelList
    from qcodes.logger.instrument_logger import InstrumentLoggerAdapter

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

log = logging.getLogger(__name__)


class InstrumentBase(Metadatable, DelegateAttributes):
    """
    Base class for all QCodes instruments and instrument channels

    Args:
        name: an identifier for this instrument, particularly for
            attaching it to a Station.
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.
    """

    def __init__(self, name: str, metadata: Optional[Mapping[Any, Any]] = None) -> None:
        self._name = str(name)
        self._short_name = str(name)

        self.parameters: Dict[str, _BaseParameter] = {}
        """
        All the parameters supported by this instrument.
        Usually populated via :py:meth:`add_parameter`.
        """
        self.functions: Dict[str, Function] = {}
        """
        All the functions supported by this
        instrument. Usually populated via :py:meth:`add_function`.
        """
        self.submodules: Dict[str, Union['InstrumentBase',
                                         'ChannelList']] = {}
        """
        All the submodules of this instrument
        such as channel lists or logical groupings of parameters.
        Usually populated via :py:meth:`add_submodule`.
        """

        super().__init__(metadata)

        # This is needed for snapshot method to work
        self._meta_attrs = ['name']

        self.log = get_instrument_logger(self, __name__)

    @property
    def name(self) -> str:
        """Name of the instrument"""
        return self._name

    @property
    def short_name(self) -> str:
        """Short name of the instrument"""
        return self._short_name

    def add_parameter(
        self, name: str, parameter_class: type = Parameter, **kwargs: Any
    ) -> None:
        """
        Bind one Parameter to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get.

        Args:
            name: How the parameter will be stored within
                ``instrument.parameters`` and also how you address it using the
                shortcut methods: ``instrument.set(param_name, value)`` etc.

            parameter_class: You can construct the parameter
                out of any class. Default :class:`.parameter.Parameter`.

            **kwargs: Constructor arguments for ``parameter_class``.

        Raises:
            KeyError: If this instrument already has a parameter with this
                name and the parameter being replaced is not an abstract
                parameter.

            ValueError: If there is an existing abstract parameter and the
                unit of the new parameter is inconsistent with the existing
                one.
        """
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
                f"from {parameter_class!r} all the way to `_BaseParameter`. "
                "This will be an error in the future.",
                QCoDeSDeprecationWarning,
            )
            self.parameters[name] = param

    def add_function(self, name: str, **kwargs: Any) -> None:
        """
        Bind one ``Function`` to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real function of the instrument.

        This functionality is meant for simple cases, principally things that
        map to simple commands like ``*RST`` (reset) or those with just a few
        arguments. It requires a fixed argument count, and positional args
        only. If your case is more complicated, you're probably better off
        simply making a new method in your ``Instrument`` subclass definition.

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
            raise KeyError(f'Duplicate function name {name}')
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

    def add_submodule(self, name: str,
                      submodule:  Union['InstrumentBase',
                                        'ChannelList']) -> None:
        """
        Bind one submodule to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        method for every submodule of the instrument.

        Submodules can effectively be considered as instruments within
        the main instrument, and should at minimum be
        snapshottable. For example, they can be used to either store
        logical groupings of parameters, which may or may not be
        repeated, or channel lists.

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
            raise KeyError(f'Duplicate submodule name {name}')
        if not isinstance(submodule, Metadatable):
            raise TypeError('Submodules must be metadatable.')
        self.submodules[name] = submodule

    def snapshot_base(self, update: Optional[bool] = False,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`qcodes.utils.helpers.NumpyJSONEncoder`
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

        snap: Dict[str, Any] = {
            "functions": {name: func.snapshot(update=update)
                          for name, func in self.functions.items()},
            "submodules": {name: subm.snapshot(update=update)
                           for name, subm in self.submodules.items()},
            "__class__": full_class(self)
        }

        snap['parameters'] = {}
        for name, param in self.parameters.items():
            if param.snapshot_exclude:
                continue
            if params_to_skip_update and name in params_to_skip_update:
                update_par: Optional[bool] = False
            else:
                update_par = update
            try:
                snap['parameters'][name] = param.snapshot(update=update_par)
            except:
                # really log this twice. Once verbose for the UI and once
                # at lower level with more info for file based loggers
                self.log.warning(f"Snapshot: Could not update "
                                 f"parameter: {name}")
                self.log.info(f"Details for Snapshot:", exc_info=True)
                snap['parameters'][name] = param.snapshot(update=False)

        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    def print_readable_snapshot(self, update: bool = False,
                                max_chars: int = 80) -> None:
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

        par_lengths = [len(p) for p in snapshot['parameters']]

        # Min of 50 is to prevent a super long parameter name to break this
        # function
        par_field_len = min(max(par_lengths)+1, 50)

        print(self.name + ':')
        print('{0:<{1}}'.format('\tparameter ', par_field_len) + 'value')
        print('-'*max_chars)
        for par in sorted(snapshot['parameters']):
            name = snapshot['parameters'][par]['name']
            msg = '{0:<{1}}:'.format(name, par_field_len)

            # in case of e.g. ArrayParameters, that usually have
            # snapshot_value == False, the parameter may not have
            # a value in the snapshot
            val = snapshot['parameters'][par].get('value', 'Not available')

            unit = snapshot['parameters'][par].get('unit', None)
            if unit is None:
                # this may be a multi parameter
                unit = snapshot['parameters'][par].get('units', None)
            if isinstance(val, floating_types):
                msg += f'\t{val:.5g} '
                # numpy float and int types format like builtins
            else:
                msg += f'\t{val} '
            if unit != '':  # corresponds to no unit
                msg += f'({unit})'
            # Truncate the message if it is longer than max length
            if len(msg) > max_chars and not max_chars == -1:
                msg = msg[0:max_chars-3] + '...'
            print(msg)

        for submodule in self.submodules.values():
            submodule.print_readable_snapshot(update=update,
                                              max_chars=max_chars)

    @property
    def parent(self) -> Optional['InstrumentBase']:
        """
        Returns the parent instrument. By default this is ``None``.
        Any SubInstrument should subclass this to return the parent instrument.
        """
        return None

    @property
    def ancestors(self) -> List['InstrumentBase']:
        """
        Returns a list of instruments, starting from the current instrument
        and following to the parent instrument and the parents parent
        instrument until the root instrument is reached.
        """
        if self.parent is not None:
            return [self] + self.parent.ancestors
        else:
            return [self]

    @property
    def root_instrument(self) -> 'InstrumentBase':
        return self

    @property
    def name_parts(self) -> List[str]:
        name_parts = [self.short_name]
        return name_parts

    @property
    def full_name(self) -> str:
        return "_".join(self.name_parts)
    #
    # shortcuts to parameters & setters & getters                           #
    #
    # instrument['someparam'] === instrument.parameters['someparam']        #
    # instrument.someparam === instrument.parameters['someparam']           #
    # instrument.get('someparam') === instrument['someparam'].get()         #
    # etc...                                                                #
    #
    delegate_attr_dicts = ['parameters', 'functions', 'submodules']

    def __getitem__(self, key: str) -> Union[Callable[..., Any], Parameter]:
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
            f'Error when pickling instrument {self.name}. '
            f'QCoDeS instruments can not be pickled.')

    def validate_status(self, verbose: bool = False) -> None:
        """ Validate the values of all gettable parameters

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
                    print(f'validate_status: param {k}: {value}')
                p.validate(value)


class AbstractInstrument(ABC):
    """ABC that is useful for defining mixin classes for Instrument class"""
    log: 'InstrumentLoggerAdapter'  # instrument logging

    @abstractmethod
    def ask(self, cmd: str) -> str:
        pass


class Instrument(InstrumentBase, AbstractInstrument):

    """
    Base class for all QCodes instruments.

    Args:
        name: an identifier for this instrument, particularly for
            attaching it to a Station.
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.

    """

    shared_kwargs = ()

    _all_instruments: "Dict[str, weakref.ref[Instrument]]" = {}
    _type = None
    _instances: "List[weakref.ref[Instrument]]" = []

    def __init__(
            self,
            name: str,
            metadata: Optional[Mapping[Any, Any]] = None
    ) -> None:

        self._t0 = time.time()

        super().__init__(name, metadata)

        self.add_parameter('IDN', get_cmd=self.get_idn,
                           vals=Anything())
        self.record_instance(self)

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Parse a standard VISA ``*IDN?`` response into an ID dict.

        Even though this is the VISA standard, it applies to various other
        types as well, such as IPInstruments, so it is included here in the
        Instrument base class.

        Override this if your instrument does not support ``*IDN?`` or
        returns a nonstandard IDN string. This string is supposed to be a
        comma-separated list of vendor, model, serial, and firmware, but
        semicolon and colon are also common separators so we accept them here
        as well.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        idstr = ''  # in case self.ask fails
        try:
            idstr = self.ask('*IDN?')
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            idparts: List[Optional[str]]
            for separator in ',;:':
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except:
            self.log.debug('Error getting or interpreting *IDN?: '
                           + repr(idstr))
            idparts = [None, self.name, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def connect_message(self, idn_param: str = 'IDN',
                        begin_time: Optional[float] = None) -> None:
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param: Name of parameter that returns ID dict.
                Default ``IDN``.
            begin_time: ``time.time()`` when init started.
                Default is ``self._t0``, set at start of ``Instrument.__init__``.
        """
        # start with an empty dict, just in case an instrument doesn't
        # heed our request to return all 4 fields.
        idn = {'vendor': None, 'model': None,
               'serial': None, 'firmware': None}
        idn.update(self.get(idn_param))
        t = time.time() - (begin_time or self._t0)

        con_msg = ('Connected to: {vendor} {model} '
                   '(serial:{serial}, firmware:{firmware}) '
                   'in {t:.2f}s'.format(t=t, **idn))
        print(con_msg)
        self.log.info(f"Connected to instrument: {idn}")

    def __repr__(self) -> str:
        """Simplified repr giving just the class and name."""
        return f"<{type(self).__name__}: {self.name}>"

    def __del__(self) -> None:
        """Close the instrument and remove its instance record."""
        try:
            self.close()
        except BaseException:
            pass

    def close(self) -> None:
        """
        Irreversibly stop this instrument and free its resources.

        Subclasses should override this if they have other specific
        resources to close.
        """
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self, whitelist=['_name'])
        self.remove_instance(self)

    @classmethod
    def close_all(cls) -> None:
        """
        Try to close all instruments registered in
        ``_all_instruments`` This is handy for use with atexit to
        ensure that all instruments are closed when a python session is
        closed.

        Examples:
            >>> atexit.register(qc.Instrument.close_all())
        """
        log.info("Closing all registered instruments")
        for inststr in list(cls._all_instruments):
            try:
                inst = cls.find_instrument(inststr)
                log.info(f"Closing {inststr}")
                inst.close()
            except:
                log.exception(f"Failed to close {inststr}, ignored")
                pass

    @classmethod
    def record_instance(cls, instance: 'Instrument') -> None:
        """
        Record (a weak ref to) an instance in a class's instance list.

        Also records the instance in list of *all* instruments, and verifies
        that there are no other instruments with the same name.

        Args:
            instance: Instance to record.

        Raises:
            KeyError: If another instance with the same name is already present.
        """
        wr = weakref.ref(instance)
        name = instance.name
        # First insert this instrument in the record of *all* instruments
        # making sure its name is unique
        existing_wr = cls._all_instruments.get(name)
        if existing_wr and existing_wr():
            raise KeyError(f'Another instrument has the name: {name}')

        cls._all_instruments[name] = wr

        # Then add it to the record for this specific subclass, using ``_type``
        # to make sure we're not recording it in a base class instance list
        if getattr(cls, '_type', None) is not cls:
            cls._type = cls
            cls._instances = []
        cls._instances.append(wr)

    @classmethod
    def instances(cls) -> List['Instrument']:
        """
        Get all currently defined instances of this instrument class.

        You can use this to get the objects back if you lose track of them,
        and it's also used by the test system to find objects to test against.

        Returns:
            A list of instances.
        """
        if getattr(cls, '_type', None) is not cls:
            # only instances of a superclass - we want instances of this
            # exact class only
            return []
        return [wr() for wr in getattr(cls, '_instances', []) if wr()]

    @classmethod
    def remove_instance(cls, instance: 'Instrument') -> None:
        """
        Remove a particular instance from the record.

        Args:
            instance: The instance to remove
        """
        wr = weakref.ref(instance)
        if wr in getattr(cls, "_instances", []):
            cls._instances.remove(wr)

        # remove from all_instruments too, but don't depend on the
        # name to do it, in case name has changed or been deleted
        all_ins = cls._all_instruments
        for name, ref in list(all_ins.items()):
            if ref is wr:
                del all_ins[name]

    @classmethod
    def find_instrument(cls, name: str,
                        instrument_class: Optional[type] = None) -> 'Instrument':
        """
        Find an existing instrument by name.

        Args:
            name: Name of the instrument.
            instrument_class: The type of instrument you are looking for.

        Returns:
            The instrument found.

        Raises:
            KeyError: If no instrument of that name was found, or if its
                reference is invalid (dead).
            TypeError: If a specific class was requested but a different
                type was found.
        """
        if name not in cls._all_instruments:
            raise KeyError(f"Instrument with name {name} does not exist")
        ins = cls._all_instruments[name]()

        if ins is None:
            del cls._all_instruments[name]
            raise KeyError(f'Instrument {name} has been removed')
        if instrument_class is not None:
            if not isinstance(ins, instrument_class):
                raise TypeError(
                    'Instrument {} is {} but {} was requested'.format(
                        name, type(ins), instrument_class))

        return cast('Instrument', ins)

    @staticmethod
    def exist(name: str, instrument_class: Optional[type] = None) -> bool:
        """
        Check if an instrument with a given names exists (i.e. is already
        instantiated).

        Args:
            name: Name of the instrument.
            instrument_class: The type of instrument you are looking for.
        """
        instrument_exists = True

        try:
            _ = Instrument.find_instrument(
                name, instrument_class=instrument_class)

        except KeyError as exception:
            instrument_is_not_found = \
                any(str_ in str(exception)
                    for str_ in [name, 'has been removed'])

            if instrument_is_not_found:
                instrument_exists = False
            else:
                raise exception

        return instrument_exists

    @staticmethod
    def is_valid(instr_instance: 'Instrument') -> bool:
        """
        Check if a given instance of an instrument is valid: if an instrument
        has been closed, its instance is not longer a "valid" instrument.

        Args:
            instr_instance: Instance of an Instrument class or its subclass.
        """
        if isinstance(instr_instance, Instrument) \
                and instr_instance in instr_instance.instances():
            # note that it is important to call `instances` on the instance
            # object instead of `Instrument` class, because instances of
            # Instrument subclasses are recorded inside their subclasses; see
            # `instances` for more information
            return True
        return False

    # `write_raw` and `ask_raw` are the interface to hardware                #
    # `write` and `ask` are standard wrappers to help with error reporting   #
    #

    def write(self, cmd: str) -> None:
        """
        Write a command string with NO response to the hardware.

        Subclasses that transform ``cmd`` should override this method, and in
        it call ``super().write(new_cmd)``. Subclasses that define a new
        hardware communication should instead override ``write_raw``.

        Args:
            cmd: The string to send to the instrument.

        Raises:
            Exception: Wraps any underlying exception with extra context,
                including the command and the instrument.
        """
        try:
            self.write_raw(cmd)
        except Exception as e:
            inst = repr(self)
            e.args = e.args + ('writing ' + repr(cmd) + ' to ' + inst,)
            raise e

    def write_raw(self, cmd: str) -> None:
        """
        Low level method to write a command string to the hardware.

        Subclasses that define a new hardware communication should override
        this method. Subclasses that transform ``cmd`` should instead
        override ``write``.

        Args:
            cmd: The string to send to the instrument.
        """
        raise NotImplementedError(
            'Instrument {} has not defined a write method'.format(
                type(self).__name__))

    def ask(self, cmd: str) -> str:
        """
        Write a command string to the hardware and return a response.

        Subclasses that transform ``cmd`` should override this method, and in
        it call ``super().ask(new_cmd)``. Subclasses that define a new
        hardware communication should instead override ``ask_raw``.

        Args:
            cmd: The string to send to the instrument.

        Returns:
            response

        Raises:
            Exception: Wraps any underlying exception with extra context,
                including the command and the instrument.
        """
        try:
            answer = self.ask_raw(cmd)

            return answer

        except Exception as e:
            inst = repr(self)
            e.args = e.args + ('asking ' + repr(cmd) + ' to ' + inst,)
            raise e

    def ask_raw(self, cmd: str) -> str:
        """
        Low level method to write to the hardware and return a response.

        Subclasses that define a new hardware communication should override
        this method. Subclasses that transform ``cmd`` should instead
        override ``ask``.

        Args:
            cmd: The string to send to the instrument.
        """
        raise NotImplementedError(
            'Instrument {} has not defined an ask method'.format(
                type(self).__name__))


def find_or_create_instrument(instrument_class: Type[Instrument],
                              name: str,
                              *args: Any,
                              recreate: bool = False,
                              **kwargs: Any
                              ) -> Instrument:
    """
    Find an instrument with the given name of a given class, or create one if
    it is not found. In case the instrument was found, and `recreate` is True,
    the instrument will be re-instantiated.

    Note that the class of the existing instrument has to be equal to the
    instrument class of interest. For example, if an instrument with the same
    name but of a different class exists, the function will raise an exception.

    This function is very convenient because it allows not to bother about
    which instruments are already instantiated and which are not.

    If an instrument is found, a connection message is printed, as if the
    instrument has just been instantiated.

    Args:
        instrument_class: Class of the instrument to find or create.
        name: Name of the instrument to find or create.
        recreate: When ``True``, the instruments gets recreated if it is found.

    Returns:
        The found or created instrument.
    """
    if not Instrument.exist(name, instrument_class=instrument_class):
        instrument = instrument_class(name, *args, **kwargs)
    else:
        instrument = Instrument.find_instrument(
            name, instrument_class=instrument_class)

        if recreate:
            instrument.close()
            instrument = instrument_class(name, *args, **kwargs)
        else:
            instrument.connect_message()  # prints the message

    return instrument
