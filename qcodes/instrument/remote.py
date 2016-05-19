import multiprocessing as mp

from qcodes.utils.deferred_operations import DeferredOperations
from qcodes.utils.helpers import DelegateAttributes
from .parameter import Parameter, GetLatest
from .function import Function
from .server import get_instrument_server_manager


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

        self._server_name = server_name
        self._shared_kwargs = shared_kwargs
        self._manager = get_instrument_server_manager(self._server_name,
                                                      self._shared_kwargs)

        self._instrument_class = instrument_class
        self._args = args
        self._kwargs = kwargs

        instrument_class.record_instance(self)
        self.connect()

    def connect(self):
        connection_attrs = self._manager.connect(self, self._instrument_class,
                                                 self._args, self._kwargs)

        # bind all the different categories of actions we need
        # to interface with the remote instrument
        # TODO: anything else?

        self.name = connection_attrs['name']
        self._id = connection_attrs['id']

        self._methods = {
            name: RemoteMethod(name, self, attrs)
            for name, attrs in connection_attrs['methods'].items()
        }

        self.parameters = {
            name: RemoteParameter(name, self, attrs)
            for name, attrs in connection_attrs['parameters'].items()
        }

        self.functions = {
            name: RemoteFunction(name, self, attrs)
            for name, attrs in connection_attrs['functions'].items()
        }

    def _ask_server(self, func_name, *args, **kwargs):
        """
        Query the server copy of this instrument, expecting a response
        """
        return self._manager.ask('cmd', self._id, func_name, *args, **kwargs)

    def _write_server(self, func_name, *args, **kwargs):
        """
        Send a command to the server copy of this instrument, without
        waiting for a response
        """
        self._manager.write('cmd', self._id, func_name, *args, **kwargs)

    def add_parameter(self, name, **kwargs):
        attrs = self._ask_server('add_parameter', name, **kwargs)
        self.parameters[name] = RemoteParameter(name, self, attrs)

    def add_function(self, name, **kwargs):
        attrs = self._ask_server('add_function', name, **kwargs)
        self.functions[name] = RemoteFunction(name, self, attrs)

    def instances(self):
        return self._instrument_class.instances()

    def close(self):
        if hasattr(self, '_manager'):
            if self._manager._server in mp.active_children():
                self._manager.delete(self._id)
            del self._manager
        self._instrument_class.remove_instance(self)

    def restart(self):
        self.close()
        self._manager.restart()

    def __getitem__(self, key):
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.name),
            id(self))
        return s


class RemoteComponent:
    '''
    One piece of a RemoteInstrument, that proxies all of its calls to the
    corresponding object in the server instrument
    '''
    def __init__(self, name, instrument, attrs):
        self.name = name
        self._instrument = instrument

        for attribute, value in attrs.items():
            if attribute == '__doc__' and value:
                value = '{} {} in RemoteInstrument {}\n---\n\n{}'.format(
                    type(self).__name__, self.name, instrument.name, value)
            setattr(self, attribute, value)


class RemoteMethod(RemoteComponent):
    def __call__(self, *args, **kwargs):
        return self._instrument._ask_server(self.name, *args, **kwargs)


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
        return self._instrument._ask_server('get', self.name)

    def set(self, value):
        # TODO: sometimes we want set to block (as here) and sometimes
        # we want it async... which would just be changing the '_ask_server'
        # to '_write_server' below. how do we decide, and how do we let the
        # user do it?
        self._instrument._ask_server('set', self.name, value)

    # manually copy over validate and __getitem__ so they execute locally
    # no reason to send these to the server, unless the validators change...
    def validate(self, value):
        return Parameter.validate(self, value)

    def __getitem__(self, keys):
        return Parameter.__getitem__(self, keys)

    def _latest(self):
        return self._instrument._ask_server('param_call', self.name,
                                            '_latest')

    def snapshot(self):
        return self._instrument._ask_server('param_call', self.name,
                                            'snapshot')

    def setattr(self, attr, value):
        self._instrument._ask_server('param_setattr', self.name,
                                     attr, value)

    def getattr(self, attr):
        return self._instrument._ask_server('param_getattr', self.name,
                                            attr)

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.name),
            id(self))
        return s

    # TODO: need set_sweep if it exists, and any methods a subclass defines.


class RemoteFunction(RemoteComponent):
    def __call__(self, *args):
        return self._instrument._ask_server('call', self.name, *args)

    def call(self, *args):
        return self.__call__(*args)

    def validate(self, *args):
        return Function.validate(self, *args)

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.name),
            id(self))
        return s
