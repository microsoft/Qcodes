"""Instrument base class."""
from __future__ import annotations

import logging
import time
import weakref
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

from qcodes.utils import strip_attrs
from qcodes.validators import Anything

from .instrument_base import InstrumentBase, InstrumentBaseKWArgs
from .instrument_meta import InstrumentMeta

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.logger.instrument_logger import InstrumentLoggerAdapter

log = logging.getLogger(__name__)


class InstrumentProtocol(Protocol):
    """Protocol that is useful for defining mixin classes for Instrument class"""

    log: InstrumentLoggerAdapter  # instrument logging

    def ask(self, cmd: str) -> str:
        ...

    def write(self, cmd: str) -> None:
        ...


T = TypeVar("T", bound="Instrument")

# a metaclass that overrides __call__ means that we lose
# both the args and return type hints.
# Since our metaclass does not modify the signature
# is is safe simply not to use that metaclass in typechecking context.
# See https://github.com/microsoft/pyright/discussions/5561 and
# https://github.com/microsoft/pyright/issues/5488
if TYPE_CHECKING:
    instrument_meta_class = type
else:
    instrument_meta_class = InstrumentMeta


class Instrument(InstrumentBase, metaclass=instrument_meta_class):
    """
    Base class for all QCodes instruments.

    Args:
        name: an identifier for this instrument, particularly for
            attaching it to a Station.
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.
        label: nicely formatted name of the instrument; if None, the
            ``name`` is used.
    """

    _all_instruments: weakref.WeakValueDictionary[
        str, Instrument
    ] = weakref.WeakValueDictionary()
    _type: type[Instrument] | None = None
    _instances: weakref.WeakSet[Instrument] = weakref.WeakSet()

    def __init__(self, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]) -> None:

        self._t0 = time.time()

        super().__init__(name=name, **kwargs)

        self.add_parameter("IDN", get_cmd=self.get_idn, vals=Anything())

    def get_idn(self) -> dict[str, str | None]:
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
        idstr = ""  # in case self.ask fails
        try:
            idstr = self.ask("*IDN?")
            idparts: list[str | None] = []
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            for separator in ",;:":
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except Exception:
            self.log.warning(
                f"Error getting or interpreting *IDN?: {idstr!r}", exc_info=True
            )
            idparts = [None, self.name, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith("model"):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(("vendor", "model", "serial", "firmware"), idparts))

    def connect_message(
        self, idn_param: str = "IDN", begin_time: float | None = None
    ) -> None:
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
        idn = {"vendor": None, "model": None, "serial": None, "firmware": None}
        idn.update(self.get(idn_param))
        t = time.time() - (begin_time or self._t0)

        con_msg = (
            "Connected to: {vendor} {model} "
            "(serial:{serial}, firmware:{firmware}) "
            "in {t:.2f}s".format(t=t, **idn)
        )
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
        if hasattr(self, "connection") and hasattr(self.connection, "close"):
            self.connection.close()

        # check for the existense first since this may already
        # have been striped e.g. if the instrument has been closed once before
        if hasattr(self, "instrument_modules"):
            for module in self.instrument_modules.values():
                strip_attrs(module, whitelist=["_short_name", "_parent"])

        if hasattr(self, "_channel_lists"):
            for channellist in self._channel_lists.values():
                for channel in channellist:
                    strip_attrs(channel, whitelist=["_short_name", "_parent"])

        strip_attrs(self, whitelist=["_short_name"])
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
                inst: Instrument = cls.find_instrument(inststr)
                log.info("Closing %s", inststr)
                inst.close()
            except Exception:
                log.exception("Failed to close %s, ignored", inststr)

    @classmethod
    def record_instance(cls, instance: Instrument) -> None:
        """
        Record (a weak ref to) an instance in a class's instance list.

        Also records the instance in list of *all* instruments, and verifies
        that there are no other instruments with the same name.

        This method is called after initialization of the instrument is completed.

        Args:
            instance: Instance to record.

        Raises:
            KeyError: If another instance with the same name is already present.
        """
        name = instance.name
        # First insert this instrument in the record of *all* instruments
        # making sure its name is unique
        existing_instr = cls._all_instruments.get(name)
        if existing_instr:
            raise KeyError(f"Another instrument has the name: {name}")

        cls._all_instruments[name] = instance

        # Then add it to the record for this specific subclass, using ``_type``
        # to make sure we're not recording it in a base class instance list
        if getattr(cls, "_type", None) is not cls:
            cls._type = cls
            cls._instances = weakref.WeakSet()
        cls._instances.add(instance)

    @classmethod
    def instances(cls: type[T]) -> list[T]:
        """
        Get all currently defined instances of this instrument class.

        You can use this to get the objects back if you lose track of them,
        and it's also used by the test system to find objects to test against.

        Returns:
            A list of instances.
        """
        if getattr(cls, "_type", None) is not cls:
            # only instances of a superclass - we want instances of this
            # exact class only
            return []
        return list(getattr(cls, "_instances", weakref.WeakSet()))

    @classmethod
    def remove_instance(cls, instance: Instrument) -> None:
        """
        Remove a particular instance from the record.

        Args:
            instance: The instance to remove
        """
        if instance in getattr(cls, "_instances", weakref.WeakSet()):
            cls._instances.remove(instance)

        # remove from all_instruments too, but don't depend on the
        # name to do it, in case name has changed or been deleted
        all_ins = cls._all_instruments
        for name, ref in list(all_ins.items()):
            if ref is instance:
                del all_ins[name]

    @overload
    @classmethod
    def find_instrument(cls, name: str, instrument_class: None = None) -> Instrument:
        ...

    @overload
    @classmethod
    def find_instrument(cls, name: str, instrument_class: type[T]) -> T:
        ...

    @classmethod
    def find_instrument(
        cls, name: str, instrument_class: type[T] | None = None
    ) -> T | Instrument:
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
        internal_instrument_class = instrument_class or Instrument

        if name not in cls._all_instruments:
            raise KeyError(f"Instrument with name {name} does not exist")
        ins = cls._all_instruments[name]
        if ins is None:
            del cls._all_instruments[name]
            raise KeyError(f"Instrument {name} has been removed")

        if not isinstance(ins, internal_instrument_class):
            raise TypeError(
                f"Instrument {name} is {type(ins)} but "
                f"{internal_instrument_class} was requested"
            )

        return ins

    @staticmethod
    def exist(name: str, instrument_class: type[Instrument] | None = None) -> bool:
        """
        Check if an instrument with a given names exists (i.e. is already
        instantiated).

        Args:
            name: Name of the instrument.
            instrument_class: The type of instrument you are looking for.
        """
        instrument_exists = True

        try:
            _ = Instrument.find_instrument(name, instrument_class=instrument_class)

        except KeyError as exception:
            instrument_is_not_found = any(
                str_ in str(exception) for str_ in [name, "has been removed"]
            )

            if instrument_is_not_found:
                instrument_exists = False
            else:
                raise exception

        return instrument_exists

    @staticmethod
    def is_valid(instr_instance: Instrument) -> bool:
        """
        Check if a given instance of an instrument is valid: if an instrument
        has been closed, its instance is not longer a "valid" instrument.

        Args:
            instr_instance: Instance of an Instrument class or its subclass.
        """
        if (
            isinstance(instr_instance, Instrument)
            and instr_instance in instr_instance.instances()
        ):
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
            e.args = e.args + ("writing " + repr(cmd) + " to " + inst,)
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
            f"Instrument {type(self).__name__} has not defined a write method"
        )

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
            e.args = e.args + ("asking " + repr(cmd) + " to " + inst,)
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
            f"Instrument {type(self).__name__} has not defined an ask method"
        )


def find_or_create_instrument(
    instrument_class: type[T],
    name: str,
    *args: Any,
    recreate: bool = False,
    **kwargs: Any,
) -> T:
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
        *args: Positional arguments passed to the instrument class.
        recreate: When ``True``, the instruments gets recreated if it is found.
        **kwargs: Keyword arguments passed to the instrument class.

    Returns:
        The found or created instrument.
    """
    if not Instrument.exist(name, instrument_class=instrument_class):
        instrument = instrument_class(name, *args, **kwargs)
    else:
        instrument = Instrument.find_instrument(name, instrument_class=instrument_class)

        if recreate:
            instrument.close()
            instrument = instrument_class(name, *args, **kwargs)
        else:
            instrument.connect_message()  # prints the message

    return instrument
