import weakref
import time

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import DelegateAttributes, strip_attrs
from .parameter import Parameter, StandardParameter
from .function import Function
from .remote import RemoteInstrument


class NoDefault:
    '''
    empty class to provide a missing default to getattr
    '''
    pass


class Instrument(Metadatable, DelegateAttributes):
    '''
    Base class for all QCodes instruments

    name: an identifier for this instrument, particularly for attaching it to
        a Station.

    server_name: this instrument starts a separate server process (or connects
        to one, if one already exists with the same name) and all hardware
        calls are made there. If '' (default), then we call classmethod
        `default_server_name`, passing in all the constructor kwargs, to
        determine the name. If not overridden, this is just 'Instruments'.

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

    shared_kwargs must be provided ONLY as kwargs when constructing
    instruments that need them, you CANNOT provide them as positional args.

    It is an error to initialize two instruments on
    the same server with different keys or values for these kwargs, unless
    the later instruments have NO shared_kwargs at all.
    '''
    shared_kwargs = []

    def __new__(cls, *args, server_name='', **kwargs):
        if server_name is None:
            return super().__new__(cls)
        else:
            return RemoteInstrument(*args, instrument_class=cls,
                                    server_name=server_name, **kwargs)

    def __init__(self, name, server_name=None, **kwargs):
        super().__init__(**kwargs)

        # you can call add_parameter and add_function *before* calling
        # super().__init__(...) from a subclass, since they contain this
        # hasattr check as well. We just put it here too so we're sure these
        # dicts get created even if they are empty.
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        if not hasattr(self, 'functions'):
            self.functions = {}

        self.name = str(name)

        self.record_instance(self)

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'Instruments'

    def connect_message(self, param_name, begin_time):
        '''
        standard message on initial connection to an instrument

        put `t0 = time.time()` at the start of your subclass __init__,
        and eg `self.connect_message('IDN', t0)` at the end (if you've
        defined a parameter 'IDN' that gives the instrument ID)
        '''
        idn = self.get(param_name).replace(',', ', ').replace('\n', ' ')
        t1 = time.time()
        return 'Connected to: ', idn, 'in %.2fs' % (t1 - begin_time)

    def getattr(self, attr, default=NoDefault):
        '''
        Get an attribute of this Instrument.
        Exact proxy for getattr if attr is a string, but can also
        get parts from nested items if attr is a sequence.

        attr: a string or sequence
            if a string, this behaves exactly as normal getattr
            if a sequence, treats the parts as diving into a nested dictionary,
                or attribute lookup for any part starting with '.' (the first
                part is always an attribute and doesn't need a '.').
                if a default is provided, it will be returned if
                the lookup fails at any level of the nesting, otherwise
                an AttributeError or KeyError will be raised
                NOTE: even with a default, if an intermediate nesting
                encounters a non-container, a TypeError will be raised.
                for example if obj.d = {'a': 1} and we call
                obj.getattr(('d','a','b'), None)

        default: value to return if the lookup fails
        '''
        try:
            if isinstance(attr, str):
                # simply attribute lookup
                return getattr(self, attr)

            else:
                # nested dictionary lookup
                obj = getattr(self, attr[0])
                for key in attr[1:]:
                    if str(key).startswith('.'):
                        obj = getattr(obj, key[1:])
                    else:
                        obj = obj[key]
                return obj

        except (AttributeError, KeyError):
            if default is NoDefault:
                raise
            else:
                return default

    def setattr(self, attr, value):
        '''
        Set an attribute of this Instrument
        Exact proxy for setattr if attr is a string, but can also
        set parts in nested items if attr is a sequence.

        attr: a string or sequence
            if a string, this behaves exactly as normal setattr
            if a sequence, treats the parts as diving into a nested dictionary,
                or attribute lookup for any part starting with '.' (the first
                part is always an attribute and doesn't need a '.').
                if any level is missing it will be created
                NOTE: if an intermediate nesting encounters a non-container,
                a TypeError will be raised.
                for example if obj.d = {'a': 1} and we call
                obj.setattr(('d','a','b'), 2)

        value: the value to store
        '''
        if isinstance(attr, str):
            setattr(self, attr, value)
        elif len(attr) == 1:
            setattr(self, attr[0], value)
        else:
            if not hasattr(self, attr[0]):
                setattr(self, attr[0], {})
            obj = getattr(self, attr[0])

            for key in attr[1: -1]:
                if str(key).startswith('.'):
                    # we don't make intermediate attributes, only
                    # intermediate dicts.
                    obj = getattr(obj, key)
                else:
                    if key not in obj:
                        obj[key] = {}
                    obj = obj[key]

            if str(attr[-1]).startswith('.'):
                setattr(obj, attr[-1][1:], value)
            else:
                obj[attr[-1]] = value

    def delattr(self, attr, prune=True):
        '''
        Delete an attribute from this Instrument
        Exact proxy for __delattr__ if attr is a string, but can also
        remove parts of nested items if attr is a sequence, in which case
        it may prune empty containers of the final attribute

        attr: a string or sequence
            if a string, this behaves exactly as normal __delattr__
            if a sequence, treats the parts as diving into a nested dictionary,
                or attribute lookup for any part starting with '.' (the first
                part is always an attribute and doesn't need a '.').
        prune: if True (default) and attr is a sequence, will try to remove
            any containing levels which have become empty
        '''
        if isinstance(attr, str):
            delattr(self, attr)
        elif len(attr) == 1:
            delattr(self, attr[0])
        else:
            obj = getattr(self, attr[0])
            # dive into the nesting, saving what we did
            tree = []
            for key in attr[1:-1]:
                if str(key).startswith('.'):
                    newobj = getattr(obj, key[1:])
                else:
                    newobj = obj[key]
                tree.append((newobj, obj, key))
                obj = newobj
            # delete the leaf
            del obj[attr[-1]]
            # work back out, deleting branches if we can
            if prune:
                for child, parent, key in reversed(tree):
                    if not child:
                        if str(key).startswith('.'):
                            delattr(parent, key[1:])
                        else:
                            del parent[key]
                    else:
                        break
                if not getattr(self, attr[0]):
                    delattr(self, attr[0])

    def __del__(self):
        wr = weakref.ref(self)
        if wr in getattr(self, '_instances', {}):
            self._instances.remove(wr)
        self.close()

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
        if not hasattr(self, 'parameters'):
            self.parameters = {}

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
        if not hasattr(self, 'functions'):
            self.functions = {}

        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

        # for use in RemoteInstruments to add functions to the server
        # we return the info they need to construct their proxy
        return func.get_attrs()

    def snapshot_base(self, update=False):
        if update:
            for par in self.parameters.values():
                par.get()
        state = self.getattr('param_state', {})
        return {
            'parameters': dict((name, param.snapshot(state=state.get(name)))
                               for name, param in self.parameters.items()),
            'functions': dict((name, func.snapshot())
                              for name, func in self.functions.items())
        }

    ##########################################################################
    # `write`, `read`, and `ask` are the interface to hardware               #
    # Override these in a subclass.                                          #
    ##########################################################################

    def write(self, cmd):
        raise NotImplementedError(
            'Instrument {} has not defined a write method'.format(
                type(self).__name__))

    def read(self):
        raise NotImplementedError(
            'Instrument {} has not defined a read method'.format(
                type(self).__name__))

    def ask(self, cmd):
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

    _keep_attrs = []

    def _get_method_attrs(self):
        '''
        grab all methods of the instrument, and return them
        as a dictionary of attribute dictionaries
        '''
        out = {}

        for attr in dir(self):
            if (attr[0] == '_' and attr not in self._keep_attrs):
                continue
            value = getattr(self, attr)
            if (not callable(value)) or isinstance(value, Function):
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
