"""Instrument base class."""
import logging
import time
import warnings
import weakref

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import DelegateAttributes, strip_attrs, full_class
from qcodes.utils.nested_attrs import NestedAttrAccess
from qcodes.utils.validators import Anything
from .parameter import StandardParameter
from .function import Function
from .metaclass import InstrumentMetaclass


class Instrument(Metadatable, DelegateAttributes, NestedAttrAccess,
                 metaclass=InstrumentMetaclass):

    """
    Base class for all QCodes instruments.

    Args:
        name (str): an identifier for this instrument, particularly for
            attaching it to a Station.

        server_name (Optional[str]): If not ``None``, this instrument starts a
            separate server process (or connects to one, if one already exists
            with the same name) and all hardware calls are made there.

            Default '', then we call classmethod ``default_server_name``,
            passing in all the constructor kwargs, to determine the name.
            If not overridden, this just gives 'Instruments'.

            **see subclass constructors below for more on ``server_name``**

            Use None to operate without a server - but then this Instrument
            will not work with qcodes Loops or other multiprocess procedures.

            If a server is used, the ``Instrument`` you asked for is
            instantiated on the server, and the object you get in the main
            process is actually a ``RemoteInstrument`` that proxies all method
            calls, ``Parameters``, and ``Functions`` to the server.

            The metaclass ``InstrumentMetaclass`` handles making either the
            requested class or its RemoteInstrument proxy.

        metadata (Optional[Dict]): additional static metadata to add to this
            instrument's JSON snapshot.


    Any unpicklable objects that are inputs to the constructor must be set
    on server initialization, and must be shared between all instruments
    that reside on the same server. To make this happen, set the
    ``shared_kwargs`` class attribute to a tuple of kwarg names that should
    be treated this way.

    It is an error to initialize two instruments on the same server with
    different keys or values for ``shared_kwargs``, unless the later
    instruments have NO ``shared_kwargs`` at all.

    subclass constructors: ``server_name`` and any ``shared_kwargs`` must be
    available as kwargs and kwargs ONLY (not positional) in all subclasses,
    and not modified in the inheritance chain. This is because we need to
    create the server before instantiating the actual instrument. The easiest
    way to manage this is to accept ``**kwargs`` in your subclass and pass them
    on to ``super().__init()``.

    Attributes:
        name (str): an identifier for this instrument, particularly for
            attaching it to a Station.

        parameters (Dict[Parameter]): All the parameters supported by this
            instrument. Usually populated via ``add_parameter``

        functions (Dict[Function]): All the functions supported by this
            instrument. Usually populated via ``add_function``
    """

    shared_kwargs = ()

    _all_instruments = {}

    def __init__(self, name, server_name=None, **kwargs):
        self._t0 = time.time()
        super().__init__(**kwargs)
        self.parameters = {}
        self.functions = {}

        self.name = str(name)

        self.add_parameter('IDN', get_cmd=self.get_idn,
                           vals=Anything())

        self._meta_attrs = ['name']

        self._no_proxy_methods = {'__getstate__'}

    def get_idn(self):
        """
        Parse a standard VISA '*IDN?' response into an ID dict.

        Even though this is the VISA standard, it applies to various other
        types as well, such as IPInstruments, so it is included here in the
        Instrument base class.

        Override this if your instrument does not support '*IDN?' or
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
            logging.warn('Error getting or interpreting *IDN?: ' + repr(idstr))
            idparts = [None, None, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    @classmethod
    def default_server_name(cls, **kwargs):
        """
        Generate a default name for the server to host this instrument.

        Args:
            **kwargs: the constructor kwargs, used if necessary to choose a
                name.

        Returns:
            str: The default server name for the specific instrument instance
                we are constructing.
        """
        return 'Instruments'

    def connect_message(self, idn_param='IDN', begin_time=None):
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param (str): name of parameter that returns ID dict.
                Default 'IDN'.
            begin_time (number): time.time() when init started.
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
            if wr in getattr(self, '_instances', {}):
                self._instances.remove(wr)
            self.close()
        except:
            pass

    def close(self):
        """
        Irreversibly stop this instrument and free its resources.

        Subclasses should override this if they have other specific
        resources to close.
        """
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self, whitelist=['name'])
        self.remove_instance(self)

    @classmethod
    def record_instance(cls, instance):
        """
        Record (a weak ref to) an instance in a class's instance list.

        Also records the instance in list of *all* instruments, and verifies
        that there are no other instruments with the same name.

        Args:
            instance (Union[Instrument, RemoteInstrument]): Note: we *do not*
                check that instance is actually an instance of ``cls``. This is
                important, because a ``RemoteInstrument`` should function as an
                instance of the instrument it proxies.

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
    def instances(cls):
        """
        Get all currently defined instances of this instrument class.

        You can use this to get the objects back if you lose track of them,
        and it's also used by the test system to find objects to test against.

        Note:
            Will also include ``RemoteInstrument`` instances that proxy
            instruments of this class.

        Returns:
            List[Union[Instrument, RemoteInstrument]]
        """
        if getattr(cls, '_type', None) is not cls:
            # only instances of a superclass - we want instances of this
            # exact class only
            return []
        return [wr() for wr in getattr(cls, '_instances', []) if wr()]

    @classmethod
    def remove_instance(cls, instance):
        """
        Remove a particular instance from the record.

        Args:
            instance (Union[Instrument, RemoteInstrument])
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
    def find_instrument(cls, name, instrument_class=None):
        """
        Find an existing instrument by name.

        Args:
            name (str)
            instrument_class (Optional[class]): The type of instrument
                you are looking for.

        Returns:
            Union[Instrument, RemoteInstrument]

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

    @classmethod
    def find_component(cls, name_attr, instrument_class=None):
        """
        Find a component of an existing instrument by name and attribute.

        Args:
            name_attr (str): A string in nested attribute format:
                <name>.<attribute>[.<subattribute>] and so on.
                For example, <attribute> can be a parameter name,
                or a method name.
            instrument_class (Optional[class]): The type of instrument
                you are looking for this component within.

        Returns:
            Any: The component requested.
        """

        if '.' in name_attr:
            name, attr = name_attr.split('.', 1)
            ins = cls.find_instrument(name, instrument_class=instrument_class)
            return ins.getattr(attr)

        else:
            # allow find_component to return the whole instrument,
            # if no attribute was specified, for maximum generality.
            return cls.find_instrument(name_attr,
                                       instrument_class=instrument_class)

    def add_parameter(self, name, parameter_class=StandardParameter,
                      **kwargs):
        """
        Bind one Parameter to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get

        Args:
            name (str): How the parameter will be stored within
                ``instrument.parameters`` and also how you address it using the
                shortcut methods: ``instrument.set(param_name, value)`` etc.

            parameter_class (Optional[type]): You can construct the parameter
                out of any class. Default ``StandardParameter``.

            **kwargs: constructor arguments for ``parameter_class``.

        Returns:
            dict: attribute information. Only used if you add parameters
                from the ``RemoteInstrument`` rather than at construction, to
                properly construct the proxy for this parameter.

        Raises:
            KeyError: if this instrument already has a parameter with this
                name.
        """
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))
        param = parameter_class(name=name, instrument=self, **kwargs)
        self.parameters[name] = param

        # for use in RemoteInstruments to add parameters to the server
        # we return the info they need to construct their proxy
        return param.get_attrs()

    def add_function(self, name, **kwargs):
        """
        Bind one Function to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real function of the instrument.

        This functionality is meant for simple cases, principally things that
        map to simple commands like '*RST' (reset) or those with just a few
        arguments. It requires a fixed argument count, and positional args
        only. If your case is more complicated, you're probably better off
        simply making a new method in your ``Instrument`` subclass definition.

        Args:
            name (str): how the Function will be stored within
            ``instrument.Functions`` and also how you  address it using the
            shortcut methods: ``instrument.call(func_name, *args)`` etc.

            **kwargs: constructor kwargs for ``Function``

        Returns:
            A dict of attribute information. Only used if you add functions
            from the ``RemoteInstrument`` rather than at construction, to
            properly construct the proxy for this function.

        Raises:
            KeyError: if this instrument already has a function with this
                name.
        """
        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

        # for use in RemoteInstruments to add functions to the server
        # we return the info they need to construct their proxy
        return func.get_attrs()

    def snapshot_base(self, update=False):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
        """
        snap = {'parameters': dict((name, param.snapshot(update=update))
                                   for name, param in self.parameters.items()),
                'functions': dict((name, func.snapshot(update=update))
                                  for name, func in self.functions.items()),
                '__class__': full_class(self),
                }
        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    #
    # `write_raw` and `ask_raw` are the interface to hardware                #
    # `write` and `ask` are standard wrappers to help with error reporting   #
    #

    def write(self, cmd):
        """
        Write a command string with NO response to the hardware.

        Subclasses that transform ``cmd`` should override this method, and in
        it call ``super().write(new_cmd)``. Subclasses that define a new
        hardware communication should instead override ``write_raw``.

        Args:
            cmd (str): the string to send to the instrument

        Raises:
            Exception: wraps any underlying exception with extra context,
                including the command and the instrument.
        """
        try:
            self.write_raw(cmd)
        except Exception as e:
            e.args = e.args + ('writing ' + repr(cmd) + ' to ' + repr(self),)
            raise e

    def write_raw(self, cmd):
        """
        Low level method to write a command string to the hardware.

        Subclasses that define a new hardware communication should override
        this method. Subclasses that transform ``cmd`` should instead
        override ``write``.

        Args:
            cmd (str): the string to send to the instrument
        """
        raise NotImplementedError(
            'Instrument {} has not defined a write method'.format(
                type(self).__name__))

    def ask(self, cmd):
        """
        Write a command string to the hardware and return a response.

        Subclasses that transform ``cmd`` should override this method, and in
        it call ``super().ask(new_cmd)``. Subclasses that define a new
        hardware communication should instead override ``ask_raw``.

        Args:
            cmd (str): the string to send to the instrument

        Returns:
            response (str, normally)

        Raises:
            Exception: wraps any underlying exception with extra context,
                including the command and the instrument.
        """
        try:
            return self.ask_raw(cmd)
        except Exception as e:
            e.args = e.args + ('asking ' + repr(cmd) + ' to ' + repr(self),)
            raise e

    def ask_raw(self, cmd):
        """
        Low level method to write to the hardware and return a response.

        Subclasses that define a new hardware communication should override
        this method. Subclasses that transform ``cmd`` should instead
        override ``ask``.

        Args:
            cmd (str): the string to send to the instrument
        """
        raise NotImplementedError(
            'Instrument {} has not defined an ask method'.format(
                type(self).__name__))

    #
    # shortcuts to parameters & setters & getters                            #
    #
    # instrument['someparam'] === instrument.parameters['someparam']        #
    # instrument.someparam === instrument.parameters['someparam']           #
    # instrument.get('someparam') === instrument['someparam'].get()         #
    # etc...                                                                #
    #

    delegate_attr_dicts = ['parameters', 'functions']

    def __getitem__(self, key):
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def set(self, param_name, value):
        """
        Shortcut for setting a parameter from its name and new value.

        Args:
            param_name (str): The name of a parameter of this instrument.
            value (any): The new value to set.
        """
        self.parameters[param_name].set(value)

    def get(self, param_name):
        """
        Shortcut for getting a parameter from its name.

        Args:
            param_name (str): The name of a parameter of this instrument.

        Returns:
            any: The current value of the parameter.
        """
        return self.parameters[param_name].get()

    def call(self, func_name, *args):
        """
        Shortcut for calling a function from its name.

        Args:
            func_name (str): The name of a function of this instrument.
            *args: any arguments to the function.

        Returns:
            any: The return value of the function.
        """
        return self.functions[func_name].call(*args)

    #
    # info about what's in this instrument, to help construct the remote     #
    #

    def connection_attrs(self, new_id):
        """
        Collect info to reconstruct the instrument API in the RemoteInstrument.

        Args:
            new_id (int): The ID of this instrument on its server.
                This is how the RemoteInstrument points its calls to the
                correct server instrument when it calls the server.

        Returns:
            dict: Dictionary of name: str, id: int, parameters: dict,
                functions: dict, _methods: dict
                parameters, functions, and _methods are dictionaries of
                name: List(str) of attributes to be proxied in the remote.
        """
        return {
            'name': self.name,
            'id': new_id,
            'parameters': {name: p.get_attrs()
                           for name, p in self.parameters.items()},
            'functions': {name: f.get_attrs()
                          for name, f in self.functions.items()},
            '_methods': self._get_method_attrs()
        }

    def _get_method_attrs(self):
        """
        Construct a dict of methods this instrument has.

        Returns:
            dict: Dictionary of method names : list of attributes each method
                has that should be proxied. As of now, this is just its
                docstring, if it has one.
        """
        out = {}

        for attr in dir(self):
            value = getattr(self, attr)
            if ((not callable(value)) or
                    value is self.parameters.get(attr) or
                    value is self.functions.get(attr) or
                    attr in self._no_proxy_methods):
                # Functions and Parameters are callable and they show up in
                # dir(), but they have their own listing.
                continue

            out[attr] = ['__doc__'] if hasattr(value, '__doc__') else []

        return out

    def __getstate__(self):
        """Prevent pickling instruments, and give a nice error message."""
        raise RuntimeError(
            'qcodes Instruments should not be pickled. Likely this means you '
            'were trying to use a local instrument (defined with '
            'server_name=None) in a background Loop. Local instruments can '
            'only be used in Loops with background=False.')
