import weakref
import time

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import DelegateAttributes, strip_attrs, full_class
from qcodes.utils.nested_attrs import NestedAttrAccess
from qcodes.utils.validators import Anything
from .parameter import StandardParameter
from .function import Function
from .remote import RemoteInstrument
import logging


class NoDefault:
    '''
    empty class to provide a missing default to getattr
    '''
    pass


class Instrument(Metadatable, DelegateAttributes, NestedAttrAccess):
    '''
    Base class for all QCodes instruments

    name: an identifier for this instrument, particularly for attaching it to
        a Station.

    server_name: this instrument starts a separate server process (or connects
        to one, if one already exists with the same name) and all hardware
        calls are made there. If '' (default), then we call classmethod
        `default_server_name`, passing in all the constructor kwargs, to
        determine the name. If not overridden, this is just 'Instruments'.

        ** see notes below about `server_name` in SUBCLASS CONSTRUCTORS **

        Use None to operate without a server - but then this Instrument
        will not work with qcodes Loops or other multiprocess procedures.

        If a server is used, the Instrument you asked for is instantiated
        on the server, and the object you get in the main process is actually
        a RemoteInstrument that proxies all method calls, Parameters, and
        Functions to the server.

    kwargs: any that make it all the way to this base class get stored as
        metadata with this instrument


    Any unpicklable objects that are inputs to the constructor must be set
    on server initialization, and must be shared between all instruments
    that reside on the same server. To make this happen, set the
    `shared_kwargs` class attribute to a list of kwarg names that should
    be treated this way.

    It is an error to initialize two instruments on the same server with
    different keys or values for `shared_kwargs`, unless the later
    instruments have NO shared_kwargs at all.

    SUBCLASS CONSTRUCTORS: `server_name` and any `shared_kwargs` must be
    available as kwargs and kwargs ONLY (not positional) in all subclasses,
    and not modified in the inheritance chain. This is because we need to
    create the server before instantiating the actual instrument. The easiest
    way to manage this is to accept **kwargs in your subclass and pass them
    on to super().__init()
    '''
    shared_kwargs = []

    def __new__(cls, *args, server_name='', **kwargs):
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

        Subclasses should override this and return dicts containing
        at least these 4 fields:
            vendor
            model
            serial
            firmware
        """
        return {'vendor': None, 'model': None,
                'serial': None, 'firmware': None}

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'Instruments'

    def connect_message(self, idn_param='IDN', begin_time=None):
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param (str): name of parameter that returns ID dict.
                Default 'IDN'.
            begin_time (number, optional): time.time() when init started.
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
        return '<{}: {}>'.format(type(self).__name__, self.name)

    def __del__(self):
        try:
            wr = weakref.ref(self)
            if wr in getattr(self, '_instances', {}):
                self._instances.remove(wr)
            self.close()
        except:
            pass

    def close(self):
        '''
        Irreversibly stop this instrument and free its resources
        subclasses should override this if they have other specific
        resources to close.
        '''
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self)
        self.remove_instance(self)

    @classmethod
    def record_instance(cls, instance):
        '''
        record (a weak ref to) an instance in a class's instance list

        note that we *do not* check that instance is actually an instance of
        cls. This is important, because a RemoteInstrument should function as
        an instance of the real Instrument it is connected to on the server.
        '''
        if getattr(cls, '_type', None) is not cls:
            cls._type = cls
            cls._instances = []
        cls._instances.append(weakref.ref(instance))

    @classmethod
    def instances(cls):
        '''
        returns all currently defined instances of this instrument class
        you can use this to get the objects back if you lose track of them,
        and it's also used by the test system to find objects to test against.
        '''
        if getattr(cls, '_type', None) is not cls:
            # only instances of a superclass - we want instances of this
            # exact class only
            return []
        return [wr() for wr in getattr(cls, '_instances', []) if wr()]

    @classmethod
    def remove_instance(cls, instance):
        wr = weakref.ref(instance)
        if wr in cls._instances:
            cls._instances.remove(wr)

    def add_parameter(self, name, parameter_class=StandardParameter,
                      **kwargs):
        '''
        binds one Parameter to this instrument.

        instrument subclasses can call this repeatedly in their __init__
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get

        `name` is how the Parameter will be stored within
        instrument.parameters and also how you address it using the
        shortcut methods:
        instrument.set(param_name, value) etc.

        `parameter_class` can be used to construct the parameter out of
            something other than StandardParameter

        kwargs: see StandardParameter (or `parameter_class`)
        '''
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))
        param = parameter_class(name=name, instrument=self, **kwargs)
        self.parameters[name] = param

        # for use in RemoteInstruments to add parameters to the server
        # we return the info they need to construct their proxy
        return param.get_attrs()

    def add_function(self, name, **kwargs):
        '''
        binds one Function to this instrument.

        instrument subclasses can call this repeatedly in their __init__
        for every real function of the instrument.

        In this sense, functions are actions of the instrument, that typically
        transcend any one parameter, such as reset, activate, or trigger.

        `name` is how the Function will be stored within instrument.functions
        and also how you  address it using the shortcut methods:
        instrument.call(func_name, *args) etc.

        see Function for the list of kwargs and notes on its limitations.
        '''
        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

        # for use in RemoteInstruments to add functions to the server
        # we return the info they need to construct their proxy
        return func.get_attrs()

    def snapshot_base(self, update=False):
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

        Returns:
            None
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
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def set(self, param_name, value):
        self.parameters[param_name].set(value)

    def get(self, param_name):
        return self.parameters[param_name].get()

    def param_getattr(self, param_name, attr, default=NoDefault):
        if default is NoDefault:
            return getattr(self.parameters[param_name], attr)
        else:
            return getattr(self.parameters[param_name], attr, default)

    def param_setattr(self, param_name, attr, value):
        setattr(self.parameters[param_name], attr, value)

    def param_call(self, param_name, method_name, *args, **kwargs):
        func = getattr(self.parameters[param_name], method_name)
        return func(*args, **kwargs)

    # and one lonely one for Functions
    def call(self, func_name, *args):
        return self.functions[func_name].call(*args)

    ##########################################################################
    # info about what's in this instrument, to help construct the remote     #
    ##########################################################################

    def _get_method_attrs(self):
        '''
        grab all methods of the instrument, and return them
        as a dictionary of attribute dictionaries
        '''
        out = {}

        for attr in dir(self):
            value = getattr(self, attr)
            if ((not callable(value)) or
                    value is self.parameters.get(attr) or
                    value is self.functions.get(attr)):
                # Functions are callable, and they show up in dir(),
                # but we don't want them included in methods, they have
                # their own listing. But we don't want to just exclude
                # all Functions, in case a Function conflicts with a method,
                # then we want the method to win because the function can still
                # be called via instrument.call or instrument.functions
                continue

            attrs = out[attr] = {}

            if hasattr(value, '__doc__'):
                attrs['__doc__'] = value.__doc__

        return out
