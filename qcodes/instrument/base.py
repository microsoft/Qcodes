import asyncio
import weakref

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import wait_for_async
from qcodes.utils.helpers import DelegateAttributes
from .parameter import StandardParameter
from .function import Function


class Instrument(Metadatable, DelegateAttributes):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.functions = {}
        self.parameters = {}

        self.name = str(name)

        # keep a (weak) record of all instances of this Instrument
        # instancetype is there to make sure we aren't using instances
        # from a superclass that has been instantiated previously
        if getattr(type(self), '_type', None) is not type(self):
            type(self)._type = type(self)
            type(self)._instances = []
        self._instances.append(weakref.ref(self))

        # TODO: need a sync/async, multiprocessing-friendly lock
        # should be based on multiprocessing.Lock (or RLock)
        # but with a non-blocking option for async use
        # anyway threading.Lock is unpicklable on Windows
        # self.lock = threading.Lock()

    def __del__(self):
        wr = weakref.ref(self)
        if wr in self._instances:
            self._instances.remove(wr)

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

        see Function for the list of kwargs
        '''
        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        self.functions[name] = Function(name=name, instrument=self, **kwargs)

    def snapshot_base(self, update=False):
        if update:
            for par in self.parameters:
                self[par].get()
        return {
            'parameters': dict((name, param.snapshot())
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
