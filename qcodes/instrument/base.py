"""Instrument base class."""
import logging
import time
import warnings
import weakref
from typing import Sequence, Optional, Dict, Union, Callable, Any, List

from qcodes.instrument.parameter_node import ParameterNode
from qcodes.utils.helpers import strip_attrs
from qcodes.utils.validators import Anything
from .parameter import Parameter


log = logging.getLogger(__name__)


class InstrumentBase(ParameterNode):
    def __init__(self, *args, **kwargs):
        warnings.warn('InstrumentBase is deprecated, please use ParameterNode')
        super().__init__(*args, **kwargs)


class Instrument(ParameterNode):

    """
    Base class for all QCodes instruments.

    Args:
        name: an identifier for this instrument, particularly for
            attaching it to a Station.
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.


    Attributes:
        name (str): an identifier for this instrument, particularly for
            attaching it to a Station.

        parameters (Dict[Parameter]): All the parameters supported by this
            instrument. Usually populated via ``add_parameter``

        functions (Dict[Function]): All the functions supported by this
            instrument. Usually populated via ``add_function``

        submodules (Dict[Metadatable]): All the submodules of this instrument
            such as channel lists or logical groupings of parameters.
            Usually populated via ``add_submodule``
    """

    shared_kwargs = ()

    _all_instruments = {}

    def __init__(self, name: str,
                 metadata: Optional[Dict]=None, **kwargs) -> None:
        self._t0 = time.time()
        if kwargs.pop('server_name', False):
            warnings.warn("server_name argument not supported any more",
                          stacklevel=0)
        super().__init__(name, **kwargs)

        self.IDN = Parameter(get_cmd=self.get_idn,
                             vals=Anything())

        self.record_instance(self)

    def get_idn(self) -> Dict:
        """
        Parse a standard VISA '\*IDN?' response into an ID dict.

        Even though this is the VISA standard, it applies to various other
        types as well, such as IPInstruments, so it is included here in the
        Instrument base class.

        Override this if your instrument does not support '\*IDN?' or
        returns a nonstandard IDN string. This string is supposed to be a
        comma-separated list of vendor, model, serial, and firmware, but
        semicolon and colon are also common separators so we accept them here
        as well.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        try:
            idstr = ''  # in case self.ask fails
            idstr = self.ask('*IDN?')
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            for separator in ',;:':
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except:
            log.debug('Error getting or interpreting *IDN?: '
                      + repr(idstr))
            idparts = [None, self.name, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def is_testing(self):
        """Return True if we are testing"""
        return self._testing

    def get_mock_messages(self):
        """
        For testing purposes we might want to get log messages from the mocker.

        Returns:
            mocker_messages: list, str
        """
        if not self._testing:
            raise ValueError("Cannot get mock messages if not in testing mode")
        return self.mocker.get_log_messages()

    def connect_message(self, idn_param: str='IDN',
                        begin_time: float=None) -> None:
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param: name of parameter that returns ID dict.
                Default 'IDN'.
            begin_time: time.time() when init started.
                Default is self._t0, set at start of Instrument.__init__.
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

    def __repr__(self):
        """Simplified repr giving just the class and name."""
        return '<{}: {}>'.format(type(self).__name__, self.name)

    def __del__(self):
        """Close the instrument and remove its instance record."""
        try:
            wr = weakref.ref(self)
            if wr in getattr(self, '_instances', []):
                self._instances.remove(wr)
            self.close()
        except:
            pass

    def close(self) -> None:
        """
        Irreversibly stop this instrument and free its resources.

        Subclasses should override this if they have other specific
        resources to close.
        """
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self, whitelist=['name'])
        self.remove_instance(self)
        log.debug(f'Closing instrument {self.name}')


    @classmethod
    def close_all(cls) -> None:
        """
        Try to close all instruments registered in
        `_all_instruments` This is handy for use with atexit to
        ensure that all instruments are closed when a python session is
        closed.

        Examples:
            >>> atexit.register(qc.Instrument.close_all())
        """
        for inststr in list(cls._all_instruments):
            try:
                inst = cls.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass

    @classmethod
    def record_instance(cls, instance: 'Instrument') -> None:
        """
        Record (a weak ref to) an instance in a class's instance list.

        Also records the instance in list of *all* instruments, and verifies
        that there are no other instruments with the same name.

        Args:
            instance: Instance to record

        Raises:
            KeyError: if another instance with the same name is already present
        """
        wr = weakref.ref(instance)
        name = instance.name
        # First insert this instrument in the record of *all* instruments
        # making sure its name is unique
        existing_wr = cls._all_instruments.get(name)
        if existing_wr and existing_wr():
            raise KeyError('Another instrument has the name: {}'.format(name))

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
            A list of instances
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
            The instance to remove
        """
        wr = weakref.ref(instance)
        if wr in cls._instances:
            cls._instances.remove(wr)

        # remove from all_instruments too, but don't depend on the
        # name to do it, in case name has changed or been deleted
        all_ins = cls._all_instruments
        for name, ref in list(all_ins.items()):
            if ref is wr:
                del all_ins[name]

    @classmethod
    def find_instrument(cls, name: str,
                        instrument_class: Optional[type]=None) -> 'Instrument':
        """
        Find an existing instrument by name.

        Args:
            name: name of the instrument
            instrument_class: The type of instrument you are looking for.

        Returns:
            Union[Instrument]

        Raises:
            KeyError: if no instrument of that name was found, or if its
                reference is invalid (dead).
            TypeError: if a specific class was requested but a different
                type was found
        """
        ins = cls._all_instruments[name]()

        if ins is None:
            del cls._all_instruments[name]
            raise KeyError('Instrument {} has been removed'.format(name))

        if instrument_class is not None:
            if not isinstance(ins, instrument_class):
                raise TypeError(
                    'Instrument {} is {} but {} was requested'.format(
                        name, type(ins), instrument_class))

        return ins

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
            cmd: the string to send to the instrument

        Raises:
            Exception: wraps any underlying exception with extra context,
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
            cmd: the string to send to the instrument
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
            cmd: the string to send to the instrument

        Returns:
            response (str, normally)

        Raises:
            Exception: wraps any underlying exception with extra context,
                including the command and the instrument.
        """
        try:
            answer = self.ask_raw(cmd)

            return answer

        except Exception as e:
            inst = repr(self)
            e.args = e.args + ('asking ' + repr(cmd) + ' to ' + inst,)
            raise e

    def ask_raw(self, cmd: str) -> None:
        """
        Low level method to write to the hardware and return a response.

        Subclasses that define a new hardware communication should override
        this method. Subclasses that transform ``cmd`` should instead
        override ``ask``.

        Args:
            cmd: the string to send to the instrument
        """
        raise NotImplementedError(
            'Instrument {} has not defined an ask method'.format(
                type(self).__name__))
