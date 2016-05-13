from qcodes.utils.deferred_operations import DeferredOperations
from qcodes.utils.helpers import DelegateAttributes
from .parameter import Parameter, GetLatest
from .function import Function
from .server import get_instrument_server


class RemoteInstrument(DelegateAttributes):
    '''
    A proxy for an instrument (of any class) running on a server process
    '''
    delegate_attr_dicts = ['_methods', 'parameters', 'functions']

    def __init__(self, *args, instrument_class=None, server_name='',
                 **kwargs):

        if server_name == '':
            server_name = instrument_class.default_server_name(**kwargs)

        shared_kwargs = {}
        for kwname in instrument_class.shared_kwargs:
            if kwname in kwargs:
                shared_kwargs[kwname] = kwargs[kwname]
                del kwargs[kwname]

        manager = get_instrument_server(server_name, shared_kwargs)
        # connect sets self.connection
        manager.connect(self, instrument_class, args, kwargs)

        # bind all the different categories of actions we need
        # to interface with the remote instrument
        # TODO: anything else?

        self.name = self.connection.instrument_name

        self._methods = {
            name: RemoteMethod(name, self, attrs)
            for name, attrs in self.connection.methods.items()
        }

        self.parameters = {
            name: RemoteParameter(name, self, attrs)
            for name, attrs in self.connection.parameters.items()
        }

        self.functions = {
            name: RemoteFunction(name, self, attrs)
            for name, attrs in self.connection.functions.items()
        }

        self._instrument_class = instrument_class
        self._server_name = server_name
        self._shared_attrs = shared_kwargs

        instrument_class.record_instance(self)

    def add_parameter(self, name, **kwargs):
        attrs = self.connection.ask('add_parameter', name, **kwargs)
        self.parameters[name] = RemoteParameter(name, self, attrs)

    def add_function(self, name, **kwargs):
        attrs = self.connection.ask('add_function', name, **kwargs)
        self.functions[name] = RemoteFunction(name, self, attrs)

    def instances(self):
        return self._instrument_class.instances()

    def close(self):
        if self.connection:
            self.connection.close()
        self.connection = None
        self._instrument_class.remove_instance(self)

    def restart(self):
        self.close()
        manager = get_instrument_server(self._server_name, self._shared_attrs)
        manager.restart()

    def __getitem__(self, key):
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]


class RemoteComponent:
    '''
    One piece of a RemoteInstrument, that proxies all of its calls to the
    corresponding object in the server instrument
    '''
    def __init__(self, name, instrument, attrs):
        self.name = name

        # Note that we don't store the connection itself in these objects, we
        # just store the instrument. That's so if the connection gets reset,
        # we don't keep an old copy of it here but instead look it up on the
        # instrument and use the current copy.
        self._instrument = instrument

        for attribute, value in attrs.items():
            if attribute == '__doc__' and value:
                value = '{} {} in RemoteInstrument {}\n---\n\n{}'.format(
                    type(self).__name__, self.name, instrument.name, value)
            setattr(self, attribute, value)


class RemoteMethod(RemoteComponent):
    def __call__(self, *args, **kwargs):
        return self._instrument.connection.ask(self.name, *args, **kwargs)


class RemoteParameter(RemoteComponent, DeferredOperations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_latest = GetLatest(self)

    def __call__(self, *args):
        if len(args) == 0:
            return self.get()
        else:
            self.set(*args)

    def get(self):
        return self._instrument.connection.ask('get', self.name)

    def set(self, value):
        # TODO: sometimes we want set to block (as here) and sometimes
        # we want it async... which would just be changing the 'ask'
        # to 'write' below. how do we decide, and how do we let the user
        # do it?
        self._instrument.connection.ask('set', self.name, value)

    # manually copy over validate and __getitem__ so they execute locally
    # no reason to send these to the server, unless the validators change...
    def validate(self, value):
        return Parameter.validate(self, value)

    def __getitem__(self, keys):
        return Parameter.__getitem__(self, keys)

    def _latest(self):
        return self._instrument.connection.ask('param_call', self.name,
                                               '_latest')

    def snapshot(self):
        return self._instrument.connection.ask('param_call', self.name,
                                               'snapshot')

    def setattr(self, attr, value):
        self._instrument.connection.ask('param_setattr', self.name,
                                        attr, value)

    def getattr(self, attr):
        return self._instrument.connection.ask('param_getattr', self.name,
                                               attr)

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            self.name,
            id(self))
        return s

    # TODO: need set_sweep if it exists, and any methods a subclass defines.


class RemoteFunction(RemoteComponent):
    def __call__(self, *args):
        return self._instrument.connection.ask('call', self.name, *args)

    def call(self, *args):
        return self.__call__(*args)

    def validate(self, *args):
        return Function.validate(self, *args)

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            self.name,
            id(self))
        return s
