"""Instrument base class."""
import weakref
import time

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import DelegateAttributes, strip_attrs, full_class
from qcodes.utils.nested_attrs import NestedAttrAccess
from qcodes.utils.validators import Anything
from .parameter import StandardParameter
from .function import Function
from .remote import RemoteInstrument


class Instrument(Metadatable, DelegateAttributes, NestedAttrAccess):

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

            ** see SUBCLASS CONSTRUCTORS below for more on ``server_name`` **

            Use None to operate without a server - but then this Instrument
            will not work with qcodes Loops or other multiprocess procedures.

            If a server is used, the ``Instrument`` you asked for is
            instantiated on the server, and the object you get in the main
            process is actually a ``RemoteInstrument`` that proxies all method
            calls, ``Parameters``, and ``Functions`` to the server.

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

    SUBCLASS CONSTRUCTORS: ``server_name`` and any ``shared_kwargs`` must be
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

    def __new__(cls, *args, server_name='', **kwargs):
        """Figure out whether to create a base instrument or proxy."""
        if server_name is None:
            return super().__new__(cls)
        else:
            return RemoteInstrument(*args, instrument_class=cls,
                                    server_name=server_name, **kwargs)

    def __init__(self, name, server_name=None, **kwargs):
        self._t0 = time.time()
        super().__init__(**kwargs)
        self.parameters = {}
        self.functions = {}

        self.name = str(name)

        self.add_parameter('IDN', get_cmd=self.get_idn,
                           vals=Anything())

        self._meta_attrs = ['name']

        self.record_instance(self)

    def get_idn(self):
        """
        Placeholder for instrument ID parameter getter.

        Subclasses should override this.

        Returns:
            A dict containing (at least) these 4 fields:
                vendor
                model
                serial
                firmware
        """
        return {'vendor': None, 'model': None,
                'serial': None, 'firmware': None}

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

        strip_attrs(self)
        self.remove_instance(self)

    @classmethod
    def record_instance(cls, instance):
        """
        Record (a weak ref to) an instance in a class's instance list.

        Args:
            instance (Union[Instrument, RemoteInstrument]): Note: we *do not*
                check that instance is actually an instance of ``cls``. This is
                important, because a ``RemoteInstrument`` should function as an
                instance of the instrument it proxies.
        """
        if getattr(cls, '_type', None) is not cls:
            cls._type = cls
            cls._instances = []
        cls._instances.append(weakref.ref(instance))

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

        ``Metadatable`` adds metadata, if any, to this snapshot.

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

    ##########################################################################
    # `write_raw` and `ask_raw` are the interface to hardware                #
    # `write` and `ask` are standard wrappers to help with error reporting   #
    ##########################################################################

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

    ##########################################################################
    # shortcuts to parameters & setters & getters                            #
    #                                                                        #
    #  instrument['someparam'] === instrument.parameters['someparam']        #
    #  instrument.someparam === instrument.parameters['someparam']           #
    #  instrument.get('someparam') === instrument['someparam'].get()         #
    #  etc...                                                                #
    ##########################################################################

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

    ##########################################################################
    # info about what's in this instrument, to help construct the remote     #
    ##########################################################################

    def connection_attrs(self, new_id):
        """
        Collect info to reconstruct the instrument API in the RemoteInstrument.

        Args:
            new_id (int): The ID of this instrument on its server.
                This is how the RemoteInstrument points its calls to the
                correct server instrument when it calls the server.
        """
        return {
            'name': self.name,
            'id': new_id,
            'parameters': {name: p.get_attrs()
                           for name, p in self.parameters.items()},
            'functions': {name: f.get_attrs()
                          for name, f in self.functions.items()},
            'methods': self._get_method_attrs()
        }

    def _get_method_attrs(self):
        """
        Construct a dict of methods this instrument has.

        Each value is itself a dict of attribute dictionaries.
        """
        out = {}

        for attr in dir(self):
            value = getattr(self, attr)
            if ((not callable(value)) or
                    value is self.parameters.get(attr) or
                    value is self.functions.get(attr)):
                # Functions and Parameters are callable and they show up in
                # dir(), but they have their own listing.
                continue

            attrs = out[attr] = {}

            if hasattr(value, '__doc__'):
                attrs['__doc__'] = value.__doc__

        return out
