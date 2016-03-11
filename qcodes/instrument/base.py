import asyncio
import weakref
from uuid import uuid4

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import wait_for_async
from qcodes.utils.helpers import DelegateAttributes, strip_attrs
from .parameter import StandardParameter
from .function import Function
from .server import connect_instrument_server, ask_server, write_server


class NoDefault:
    '''
    empty class to provide a missing default to getattr
    '''
    pass


class Instrument(Metadatable, DelegateAttributes):
    '''
    Base class for all QCodes instruments

    server_name: if provided (and one generally should be) then this instrument
        starts a separate server process (or connects to one, if one already
        exists with the same name) and all hardware calls are made there.

    server_extras: a dictionary of objects to be passed to the server, and
        from there attached as attributes to each instrument that connects to
        it. Intended for things like extra queues that can't be sent over a
        queue themselves.

    kwargs: metadata to store with this instrument

    The object on the server is simply a copy of the Instrument itself. Any
    methods that are decorated with either @ask_server or @write_server
    (from qcodes.instrument.server) will execute on the server regardless of
    which process calls them.

    Subclasses, in their __init__, should first set up any methods/attributes
    that need to exist in BOTH server and local copies of the instrument (such
    as decorated methods), eg:
        self.write = self._default_write
    then call super init:
        super().__init__(name, server_name, server_extras, **kwargs)
    which loads this instrument into the server.
    After that, __init__ should not set any attributes of self directly if the
    hardware connection needs them, only through @ask_server or @write_server,
    to prepare the connection and hardware.

    Subclasses should override at least one each of write/write_async,
    ask/ask_async, and potentially read/read_async, decorating each with
    ask_server or write_server as appropriate (from qcodes.instrument.server).

    Any other methods that interact with hardware must also be decorated.
    It's OK if a decorated method calls another decorated method
    '''
    connection = None

    def __init__(self, name, server_name=None, server_extras={}, **kwargs):
        super().__init__(**kwargs)
        self.functions = {}
        self.parameters = {}

        self.uuid = uuid4().hex

        self.name = str(name)

        # keep a (weak) record of all instances of this Instrument
        # instancetype is there to make sure we aren't using instances
        # from a superclass that has been instantiated previously
        if getattr(type(self), '_type', None) is not type(self):
            type(self)._type = type(self)
            type(self)._instances = []
        self._instances.append(weakref.ref(self))

        if server_name is not None:
            self.connection = connect_instrument_server(server_name, self,
                                                        server_extras)

    @ask_server
    def getattr(self, attr, default=NoDefault):
        '''
        Get an attribute of the server copy of this Instrument.
        Exact proxy for getattr if attr is a string, but can also
        get parts from nested items if attr is a sequence.

        attr: a string or sequence
            if a string, this behaves exactly as normal getattr
            if a sequence, treats the parts as diving into a nested dictionary.
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
                    obj = obj[key]
                return obj

        except (AttributeError, KeyError):
            if default is NoDefault:
                raise
            else:
                return default

    @write_server
    def setattr(self, attr, value):
        '''
        Set an attribute of the server copy of this Instrument
        Exact proxy for setattr if attr is a string, but can also
        set parts in nested items if attr is a sequence.

        attr: a string or sequence
            if a string, this behaves exactly as normal setattr
            if a sequence, treats the parts as diving into a nested dictionary.
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
                if key not in obj:
                    obj[key] = {}
                obj = obj[key]

            obj[attr[-1]] = value

    @write_server
    def delattr(self, attr, prune=True):
        '''
        Delete an attribute from the server copy of this Instrument
        Exact proxy for __delattr__ if attr is a string, but can also
        remove parts of nested items if attr is a sequence, in which case
        it may prune empty containers of the final attribute

        attr: a string or sequence
            if a string, this behaves exactly as normal __delattr__
            if a sequence, treats the parts as diving into a nested dictionary.
        prune: if True (default) and attr is a sequence, will try to remove
            any containing levels which have become empty
        '''
        if isinstance(attr, str):
            self.__delattr__(attr)
        elif len(attr) == 1:
            self.__delattr__(attr[0])
        else:
            obj = getattr(self, attr[0])
            # dive into the nesting, saving what we did
            tree = []
            for key in attr[1:-1]:
                newobj = obj[key]
                tree.append((newobj, obj, key))
                obj = newobj
            # delete the leaf
            del obj[attr[-1]]
            # work back out, deleting branches if we can
            if prune:
                for child, parent, key in reversed(tree):
                    if not child:
                        del parent[key]
                    else:
                        break
                if not getattr(self, attr[0]):
                    self.__delattr__(attr[0])

    def __del__(self):
        wr = weakref.ref(self)
        if wr in getattr(self, '_instances', {}):
            self._instances.remove(wr)
        self.close()

    def close(self):
        '''
        Irreversibly stop this instrument and free its resources
        '''
        if self.connection:
            self.connection.close()

        strip_attrs(self)

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
        self.parameters[name] = parameter_class(name=name, instrument=self,
                                                **kwargs)

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
        self.functions[name] = Function(name=name, instrument=self, **kwargs)

    def snapshot_base(self, update=False):
        if update:
            for par in self.parameters:
                self[par].get()
        state = self.getattr('param_state', {})
        return {
            'parameters': dict((name, param.snapshot(state=state.get(name)))
                               for name, param in self.parameters.items()),
            'functions': dict((name, func.snapshot())
                              for name, func in self.functions.items())
        }

    ##########################################################################
    # write, read, and ask are the interface to hardware                     #
    #                                                                        #
    # at least one (sync or async) of each pair should be overridden by a    #
    # subclass. These defaults simply convert between sync and async if only #
    # one is defined, but raise an error if neither is.                      #
    ##########################################################################

    def write(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .write
        converts to the async version if the subclass supplies one.
        '''
        wait_for_async(self.write_async, cmd)

    @asyncio.coroutine
    def write_async(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .write_async
        converts to the sync version if the subclass supplies one.
        '''
        # check if the paired function is still from the base class (so we'd
        # have a recursion loop) notice that we only have to do this in one
        # of the pair, because the other will call this one.
        if self.write.__func__ is Instrument.write:
            raise NotImplementedError(
                'instrument {} has no write method defined'.format(self.name))
        self.write(cmd)

    def read(self):
        '''
        The Instrument base class has no hardware connection. This .read
        converts to the async version if the subclass supplies one.
        '''
        return wait_for_async(self.read_async)

    @asyncio.coroutine
    def read_async(self):
        '''
        The Instrument base class has no hardware connection. This .read_async
        converts to the sync version if the subclass supplies one.
        '''
        if self.read.__func__ is Instrument.read:
            raise NotImplementedError(
                'instrument {} has no read method defined'.format(self.name))
        return self.read()

    def ask(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .ask
        converts to the async version if the subclass supplies one.
        '''
        return wait_for_async(self.ask_async, cmd)

    @asyncio.coroutine
    def ask_async(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .ask_async
        converts to the sync version if the subclass supplies one.
        '''
        if self.ask.__func__ is Instrument.ask:
            raise NotImplementedError(
                'instrument {} has no ask method defined'.format(self.name))
        return self.ask(cmd)

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

    @asyncio.coroutine
    def set_async(self, param_name, value):
        yield from self.parameters[param_name].set_async(value)

    def get(self, param_name):
        return self.parameters[param_name].get()

    @asyncio.coroutine
    def get_async(self, param_name):
        return (yield from self.parameters[param_name].get_async())

    def call(self, func_name, *args):
        return self.functions[func_name].call(*args)

    @asyncio.coroutine
    def call_async(self, func_name, *args):
        return (yield from self.functions[func_name].call_async(*args))
