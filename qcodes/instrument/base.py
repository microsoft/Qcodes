import asyncio
import functools
import weakref
from uuid import uuid4

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import wait_for_async, mock_async
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

    name: an identifier for this instrument, particularly for attaching it to
        a Station.

    server_name: this instrument starts a separate server process (or connects
        to one, if one already exists with the same name) and all hardware
        calls are made there. default 'Instruments'.
        Use None operate without a server - but then this Instrument
        will not work with qcodes Loops or other multiprocess procedures.

    server_extras: a dictionary of objects to be passed to the server, and
        from there attached as self.server_extras to each instrument that
        connects to it. Intended for unpicklable things like extra queues.
        Note that the Instrument to *start* the server must provide the extras
        for *all* instruments that will use the server.

    kwargs: metadata to store with this instrument


    When you use a server, the object on the server is simply a copy of
    this one, but they execute different initialization code. Any method
    decorated with either @qcodes.ask_server or @qcodes.write_server
    will execute on the server regardless of which process calls them. This
    should include any method that interacts directly with hardware.
    In particular, subclasses should override at least one each of:
    - write or write_async
    - ask or ask_async
    - optionally read or read_async
    decorating each with @ask_server or @write_server as appropriate.

    Unlike most subclass initializations, the normal pattern with Instrument
    subclasses is to call super().__init__(...) at the END of __init__.
    There are three places you can put initialization code in subclasses,
    and they all have different purposes:

    - In self.__init__ BEFORE super().__init__(...):
        Attributes set here will exist in both the local and server copies.
        Such attributes MUST be picklable - NO hardware connections.
        add_parameter and add_function may be called here, if you would like
        the server copy to be able to use them in custom methods.

    - In self.on_connect:
        Attributes set here will exist ONLY in the server copy.
        This is where you should make the actual hardware connections.
        on_connect is called from the local copy, so should either be decorated
        with write_server (or ask_server to ensure that it finishes before you
        do anything else locally) or call decorated methods. Do not add
        parameters or functions here, as they will not be accessible outside
        the server process.

    - In self.__init__ AFTER super().__init__(...):
        Not normally used at all.
        Attributes set here will exist ONLY in the local copy, but it is still
        recommended not to make unpicklable attributes here, as this will
        interfere with restarting the instrument server if that becomes
        necessary.
        add_parameter and add_function calls can go here if you're sure the
        server will not need these objects for any custom methods you define.

    More notes about dealing with instrument servers:

    - If you don't use a server, all three parts of initialization code will
        execute locally, in the order presented above.

    - Once a server is started, attributes will not sync between the local
        and server copies automatically. But you can use the getattr and
        setattr METHODS (not the getattr and setattr functions) to access
        attributes of the server copy from any process. These methods can
        also access pieces of nested dictionaries as well.

    - It's OK for decorated methods to call other decorated methods.
    '''
    connection = None

    def __init__(self, name, server_name='Instruments', server_extras={},
                 **kwargs):
        super().__init__(**kwargs)

        # you can call add_parameter and add_function *before* calling
        # super().__init__(...) from a subclass, since they contain this
        # hasattr check as well. We just put it here too so we're sure these
        # dicts get created even if they are empty.
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        if not hasattr(self, 'functions'):
            self.functions = {}

        self.uuid = uuid4().hex

        self.name = str(name)

        # keep a (weak) record of all instances of this Instrument
        # cls._type is there to make sure we aren't using instances
        # from a superclass that has been instantiated previously
        cls = type(self)
        if getattr(cls, '_type', None) is not cls:
            cls._type = cls
            cls._instances = []
        cls._instances.append(weakref.ref(self))

        # check if read/write/ask have been implemented by a subclass
        # for each set, there are now 4 possible methods that could be
        # overridden, any of which would be sufficient
        for action in ('write', 'read', 'ask'):
            if not self._has_action(action):
                self._set_not_implemented(action)

        if server_name is not None:
            connect_instrument_server(server_name, self, server_extras)
        else:
            self.server_extras = server_extras
            self.on_connect()

    def on_connect(self):
        '''
        This method gets called after connecting the Instrument to an
        InstrumentServer, which happens on init as well as if the
        server gets restarted. It's called locally but should either
        be decorated itself or only call decorated methods, to prepare
        anything that should happen after connection to the server,
        primarily setting up the hardware connection.

        It is recommended to decorate this with @ask_server so it waits
        for a response and thus can catch errors, but you can also
        decorate it with @write_server and it will execute asynchronously,
        in the server process, not blocking the main process to wait
        for a response.
        '''
        pass

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
        if not hasattr(self, 'parameters'):
            self.parameters = {}

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
        if not hasattr(self, 'functions'):
            self.functions = {}

        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        self.functions[name] = Function(name=name, instrument=self, **kwargs)

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
    # write, read, and ask are the interface to hardware                     #
    #                                                                        #
    # at least one (sync or async) of each pair should be overridden by a    #
    # subclass. These defaults simply convert between sync and async if only #
    # one is defined, but raise an error if neither is.                      #
    #                                                                        #
    # Note: no subclasses should set (write|read|ask)[_async] to an instance #
    # variable, as these will not be available to parameters defined before  #
    # super().__init__().                                                    #
    # Intermediate subclasses (such as VisaInstrument, which will be         #
    # subclassed again by an actual instrument) MAY set _write_fn etc to an  #
    # instance variable if they wish to supply write/ask only if neither     #
    # the sync or async forms are overridden by a final subclass             #
    # this way the base methods are all defined at the class level, so are   #
    # available before __init__, but they can still be modified.             #
    ##########################################################################

    def write(self, cmd):
        return self._write_fn(cmd)

    @asyncio.coroutine
    def write_async(self, cmd):
        return (yield from self._write_async_fn(cmd))

    def _write_fn(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .write
        converts to the async version if the subclass supplies one.
        '''
        return wait_for_async(self.write_async, cmd)

    @asyncio.coroutine
    def _write_async_fn(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .write_async
        converts to the sync version if the subclass supplies one.
        '''
        return self.write(cmd)

    def read(self):
        return self._read_fn()

    @asyncio.coroutine
    def read_async(self):
        return (yield from self._read_async_fn())

    def _read_fn(self):
        '''
        The Instrument base class has no hardware connection. This .read
        converts to the async version if the subclass supplies one.
        '''
        return wait_for_async(self.read_async)

    @asyncio.coroutine
    def _read_async_fn(self):
        '''
        The Instrument base class has no hardware connection. This .read_async
        converts to the sync version if the subclass supplies one.
        '''
        return self.read()

    def ask(self, cmd):
        return self._ask_fn(cmd)

    @asyncio.coroutine
    def ask_async(self, cmd):
        return (yield from self._ask_async_fn(cmd))

    def _ask_fn(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .ask
        converts to the async version if the subclass supplies one.
        '''
        return wait_for_async(self.ask_async, cmd)

    @asyncio.coroutine
    def _ask_async_fn(self, cmd):
        '''
        The Instrument base class has no hardware connection. This .ask_async
        converts to the sync version if the subclass supplies one.
        '''
        return self.ask(cmd)

    def _raise_not_implemented(self, method, *args):
        '''
        intended to replace _(write|read|ask)[_async]_fn when no
        subclasses have overridden any of the appropriate methods.
        This way we don't need to check for recursion loops in every call.

        usage examples:
        self._write_fn = functools.partial(
            self._raise_not_implemented, 'write')
        self._write_async_fn = mock_async(
            functools.partial(self._raise_not_implemented, 'write'))
        '''
        msg = 'instrument {0} has no {1} or {1}_async method defined'
        raise NotImplementedError(msg.format(self.name, method))

    def _has_action(self, action):
        for method_form in ('{}', '{}_async', '_{}_fn', '_{}_async_fn'):
            method = method_form.format(action)
            this_func = getattr(self, method).__func__
            base_func = getattr(Instrument, method)
            if (this_func is not base_func):
                return True
        return False

    def _set_not_implemented(self, action):
        setattr(self, '_{}_fn'.format(action),
                functools.partial(self._raise_not_implemented, action))
        setattr(self, '_{}_async_fn'.format(action), mock_async(
                functools.partial(self._raise_not_implemented, action)))

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
