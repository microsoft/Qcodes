import threading

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import wait_for_async
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.function import Function


class BaseInstrument(Metadatable):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.functions = {}
        self.parameters = {}

        self.name = str(name)
        # TODO: need an async-friendly non-blocking lock
        self.lock = threading.Lock()

    def add_parameter(self, name, **kwargs):
        '''
        binds one Parameter to this instrument.

        instrument subclasses can call this repeatedly in their __init__
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get

        `name` is how the Parameter will be stored within instrument.parameters
        and also how you  address it using the shortcut methods:
        instrument.set(param_name, value) etc.

        see Parameter for the list of kwargs
        '''
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))
        self.parameters[name] = Parameter(self, name, **kwargs)

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
        self.functions[name] = Function(self, name, **kwargs)

    def snapshot_base(self):
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
        wait_for_async(self.write_async, cmd)

    async def write_async(self, cmd):
        # check if the paired function is still from the base class (so we'd
        # have a recursion loop) notice that we only have to do this in one
        # of the pair, because the other will call this one.
        if self.write.__func__ is BaseInstrument.write:
            raise NotImplementedError(
                'instrument {} has no write method defined'.format(self.name))
        self.write(cmd)

    def read(self):
        return wait_for_async(self.read_async)

    async def read_async(self):
        if self.read.__func__ is BaseInstrument.read:
            raise NotImplementedError(
                'instrument {} has no read method defined'.format(self.name))
        return self.read()

    def ask(self, cmd):
        return wait_for_async(self.ask_async, cmd)

    async def ask_async(self, cmd):
        if self.ask.__func__ is BaseInstrument.ask:
            raise NotImplementedError(
                'instrument {} has no ask method defined'.format(self.name))
        return self.ask(cmd)

    ##########################################################################
    # shortcuts to parameters & setters & getters                            #
    #                                                                        #
    #  instrument['someparam'] === instrument.parameters['someparam']        #
    #  instrument.get('someparam') === instrument['someparam'].get()         #
    #  etc...                                                                #
    ##########################################################################

    def __getitem__(self, key):
        return self.parameters[key]

    def set(self, param_name, value):
        self.parameters[param_name].set(value)

    async def set_async(self, param_name, value):
        await self.parameters[param_name].set_async(value)

    def get(self, param_name):
        return self.parameters[param_name].get()

    async def get_async(self, param_name):
        return await self.parameters[param_name].get_async()

    def call(self, func_name, *args):
        return self.functions[func_name].call(*args)

    async def call_async(self, func_name, *args):
        return await self.functions[func_name].call_async(*args)
