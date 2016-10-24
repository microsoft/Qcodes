"""Proxies to interact with server-based instruments from another process."""
import multiprocessing as mp

from qcodes.utils.deferred_operations import DeferredOperations
from qcodes.utils.helpers import DelegateAttributes, named_repr
from .parameter import Parameter, GetLatest
from .server import get_instrument_server_manager


class RemoteInstrument(DelegateAttributes):

    """
    A proxy for an instrument (of any class) running on a server process.

    Creates the server if necessary, then loads this instrument onto it,
    then mirrors the API to that instrument.

    Args:
        *args: Passed along to the real instrument constructor.

        instrument_class (type): The class of the real instrument to make.

        server_name (str): The name of the server to create or use for this
            instrument. If not provided (''), gets a name from
            ``instrument_class.default_server_name(**kwargs)`` using the
            same kwargs passed to the instrument constructor.

        **kwargs: Passed along to the real instrument constructor, also
            to ``default_server_name`` as mentioned.

    Attributes:
        name (str): an identifier for this instrument, particularly for
            attaching it to a Station.

        parameters (Dict[Parameter]): All the parameters supported by this
            instrument. Usually populated via ``add_parameter``

        functions (Dict[Function]): All the functions supported by this
            instrument. Usually populated via ``add_function``
    """

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

        self.connect()

    def connect(self):
        """Create the instrument on the server and replicate its API here."""

        # connection_attrs is created by instrument.connection_attrs(),
        # called by InstrumentServer.handle_new after it creates the instrument
        # on the server.
        connection_attrs = self._manager.connect(self, self._instrument_class,
                                                 self._args, self._kwargs)

        self.name = connection_attrs['name']
        self._id = connection_attrs['id']
        self._methods = {}
        self.parameters = {}
        self.functions = {}

        # bind all the different categories of actions we need
        # to interface with the remote instrument

        self._update_components(connection_attrs)

    def _update_components(self, connection_attrs):
        """
        Update the three component dicts with new or updated connection attrs.

        Args:
            connection_attrs (dict): as returned by
                ``Instrument.connection_attrs``, should contain at least keys
                ``_methods``, ``parameters``, and ``functions``, whose values
                are themselves dicts of {component_name: list of attributes}.
                These get translated into the corresponding dicts eg:
                ``self.parameters = {parameter_name: RemoteParameter}``
        """
        component_types = (('_methods', RemoteMethod),
                           ('parameters', RemoteParameter),
                           ('functions', RemoteFunction))

        for container_name, component_class in component_types:
            container = getattr(self, container_name)
            components_spec = connection_attrs[container_name]

            # first delete components that are gone and update those that
            # have changed
            for name in list(container.keys()):
                if name in components_spec:
                    container[name].update(components_spec[name])
                else:
                    del container[name]

            # then add new components
            for name, attrs in components_spec.items():
                if name not in container:
                    container[name] = component_class(name, self, attrs)

    def update(self):
        """Check with the server for updated components."""
        connection_attrs = self._ask_server('connection_attrs', self._id)
        self._update_components(connection_attrs)

    def _ask_server(self, func_name, *args, **kwargs):
        """Query the server copy of this instrument, expecting a response."""
        return self._manager.ask('cmd', self._id, func_name, *args, **kwargs)

    def _write_server(self, func_name, *args, **kwargs):
        """Send a command to the server, without waiting for a response."""
        self._manager.write('cmd', self._id, func_name, *args, **kwargs)

    def add_parameter(self, name, **kwargs):
        """
        Proxy to add a new parameter to the server instrument.

        This is only for adding parameters remotely to the server copy.
        Normally parameters are added in the instrument constructor, rather
        than via this method. This method is limited in that you can generally
        only use the string form of a command, not the callable form.

        Args:
            name (str): How the parameter will be stored within
                ``instrument.parameters`` and also how you address it using the
                shortcut methods: ``instrument.set(param_name, value)`` etc.

            parameter_class (Optional[type]): You can construct the parameter
                out of any class. Default ``StandardParameter``.

            **kwargs: constructor arguments for ``parameter_class``.
        """
        attrs = self._ask_server('add_parameter', name, **kwargs)
        self.parameters[name] = RemoteParameter(name, self, attrs)

    def add_function(self, name, **kwargs):
        """
        Proxy to add a new Function to the server instrument.

        This is only for adding functions remotely to the server copy.
        Normally functions are added in the instrument constructor, rather
        than via this method. This method is limited in that you can generally
        only use the string form of a command, not the callable form.

        Args:
            name (str): how the function will be stored within
            ``instrument.functions`` and also how you  address it using the
            shortcut methods: ``instrument.call(func_name, *args)`` etc.

            **kwargs: constructor kwargs for ``Function``
        """
        attrs = self._ask_server('add_function', name, **kwargs)
        self.functions[name] = RemoteFunction(name, self, attrs)

    def instances(self):
        """
        A RemoteInstrument shows as an instance of its proxied class.

        Returns:
            List[Union[Instrument, RemoteInstrument]]
        """
        return self._instrument_class.instances()

    def find_instrument(self, name, instrument_class=None):
        """
        Find an existing instrument by name.

        Args:
            name (str)

        Returns:
            Union[Instrument, RemoteInstrument]

        Raises:
            KeyError: if no instrument of that name was found, or if its
                reference is invalid (dead).
        """
        return self._instrument_class.find_instrument(
            name, instrument_class=instrument_class)

    def close(self):
        """Irreversibly close and tear down the server & remote instruments."""
        if hasattr(self, '_manager'):
            if self._manager._server in mp.active_children():
                self._manager.delete(self._id)
            del self._manager
        self._instrument_class.remove_instance(self)

    def restart(self):
        """
        Remove and recreate the server copy of this instrument.

        All instrument state will be returned to the initial conditions,
        including deleting any parameters you've added after initialization,
        or modifications to parameters etc.
        """
        self._manager.delete(self._id)
        self.connect()

    def __getitem__(self, key):
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def __repr__(self):
        """repr including the instrument name."""
        return named_repr(self)


class RemoteComponent:

    """
    An object that lives inside a RemoteInstrument.

    Proxies all of its calls and specific listed attributes to the
    corresponding object in the server instrument.

    Args:
        name (str): The name of this component.

        instrument (RemoteInstrument): the instrument this is part of.

        attrs (List[str]): instance attributes to proxy to the server
            copy of this component.

    Attributes:
        name (str): The name of this component.

        _instrument (RemoteInstrument): the instrument this is part of.

        _attrs (Set[str]): All the attributes we are allowed to proxy.

        _delattrs (Set[str]): Attributes we've deleted from the server,
            a subset of ``_attrs``, but if you set them again, they will
            still be set on the server.

        _local_attrs (Set[str]): (class attribute only) Attributes that we
            shouldn't look for on the server, even if they do not exist
            locally. Mostly present to prevent infinite recursion in the
            accessors.
    """
    _local_attrs = {
        '_attrs',
        'name',
        '_instrument',
        '_local_attrs',
        '__doc__',
        '_delattrs'
    }

    def __init__(self, name, instrument, attrs):
        self.name = name
        self._instrument = instrument
        self.update(attrs)

    def update(self, attrs):
        """
        Update the set of attributes proxied by this component.

        The docstring is not proxied every time it is accessed, but it is
        read and updated during this method.

        Args:
            attrs (Sequence[str]): the new set of attributes to proxy.
        """
        self._attrs = set(attrs)
        self._delattrs = set()
        self._set_doc()

    def __getattr__(self, attr):
        """
        Get an attribute value from the server.

        If there was a local attribute, we don't even get here.
        """
        if attr not in type(self)._local_attrs and attr in self._attrs:
            full_attr = self.name + '.' + attr
            return self._instrument._ask_server('getattr', full_attr)
        else:
            raise AttributeError('RemoteComponent has no local or remote '
                                 'attribute: ' + attr)

    def __setattr__(self, attr, val):
        """
        Set a new attribute value.

        If the attribute is listed as remote, we'll set it on the server,
        otherwise we'll set it locally.
        """
        if attr not in type(self)._local_attrs and attr in self._attrs:
            full_attr = self.name + '.' + attr
            self._instrument._ask_server('setattr', full_attr, val)
            if attr in self._delattrs:
                self._delattrs.remove(attr)
        else:
            object.__setattr__(self, attr, val)

    def __delattr__(self, attr):
        """
        Delete an attribute.

        If the attribute is listed as remote, we'll delete it on the server,
        otherwise we'll delete it locally.
        """

        if attr not in type(self)._local_attrs and attr in self._attrs:
            full_attr = self.name + '.' + attr
            self._instrument._ask_server('delattr', full_attr)
            self._delattrs.add(attr)

        else:
            object.__delattr__(self, attr)

    def __dir__(self):
        """dir listing including both local and server attributes."""
        remote_attrs = self._attrs - self._delattrs
        return sorted(remote_attrs.union(super().__dir__()))

    def _set_doc(self):
        """
        Prepend a note about remoteness to the server docstring.

        If no server docstring is found, we leave the class docstring.

        __doc__, as a magic attribute, is handled differently from
        other attributes so we won't make it dynamic (updating on the
        server when you change it here)
        """
        doc = self._instrument._ask_server('getattr',
                                           self.name + '.__doc__')

        docbase = '{} {} in RemoteInstrument {}'.format(
            type(self).__name__, self.name, self._instrument.name)

        self.__doc__ = docbase + (('\n---\n\n' + doc) if doc else '')

    def __repr__(self):
        """repr including the component name."""
        return named_repr(self)


class RemoteMethod(RemoteComponent):

    """Proxy for a method of the server instrument."""

    def __call__(self, *args, **kwargs):
        """Call the method on the server, passing on any args and kwargs."""
        return self._instrument._ask_server(self.name, *args, **kwargs)


class RemoteParameter(RemoteComponent, DeferredOperations):

    """Proxy for a Parameter of the server instrument."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_latest = GetLatest(self)

    def __call__(self, *args):
        """
        Shortcut to get (with no args) or set (with one arg) the parameter.

        Args:
            *args: If empty, get the parameter. If one arg, set the parameter
                to this value

        Returns:
            any: The parameter value, if called with no args,
                otherwise no return.
        """
        if len(args) == 0:
            return self.get()
        else:
            self.set(*args)

    def get(self):
        """
        Read the value of this parameter.

        Returns:
            any: the current value of the parameter.
        """
        return self._instrument._ask_server('get', self.name)

    def set(self, value):
        """
        Set a new value of this parameter.

        Args:
            value (any): the new value for the parameter.
        """
        # TODO: sometimes we want set to block (as here) and sometimes
        # we want it async... which would just be changing the '_ask_server'
        # to '_write_server' below. how do we decide, and how do we let the
        # user do it?
        self._instrument._ask_server('set', self.name, value)

    def validate(self, value):
        """
        Raise an error if the given value is not allowed for this Parameter.

        Args:
            value (any): the proposed new parameter value.

        Raises:
            TypeError: if ``value`` has the wrong type for this Parameter.
            ValueError: if the type is correct but the value is wrong.
        """
        self._instrument._ask_server('callattr',
                                     self.name + '.validate', value)

    # manually copy over sweep and __getitem__ so they execute locally
    # and are still based off the RemoteParameter
    def __getitem__(self, keys):
        """Create a SweepValues from this parameter with slice notation."""
        return Parameter.__getitem__(self, keys)

    def sweep(self, *args, **kwargs):
        """Create a SweepValues from this parameter. See Parameter.sweep."""
        return Parameter.sweep(self, *args, **kwargs)

    def _latest(self):
        return self._instrument._ask_server('callattr', self.name + '._latest')

    def snapshot(self, update=False):
        """
        State of the parameter as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest value in memory.

        Returns:
            dict: snapshot
        """
        return self._instrument._ask_server('callattr',
                                            self.name + '.snapshot', update)

    def setattr(self, attr, value):
        """
        Set an attribute of the parameter on the server.

        Args:
            attr (str): the attribute name. Can be nested as in
                ``NestedAttrAccess``.
            value: The new value to set.
        """
        self._instrument._ask_server('setattr', self.name + '.' + attr, value)

    def getattr(self, attr):
        """
        Get an attribute of the parameter on the server.

        Args:
            attr (str): the attribute name. Can be nested as in
                ``NestedAttrAccess``.

        Returns:
            any: The attribute value.
        """
        return self._instrument._ask_server('getattr', self.name + '.' + attr)

    def callattr(self, attr, *args, **kwargs):
        """
        Call arbitrary methods of the parameter on the server.

        Args:
            attr (str): the method name. Can be nested as in
                ``NestedAttrAccess``.
            *args: positional args to the method
            **kwargs: keyword args to the method

        Returns:
            any: the return value of the called method.
        """
        return self._instrument._ask_server(
            'callattr', self.name + '.' + attr, *args, **kwargs)


class RemoteFunction(RemoteComponent):

    """Proxy for a Function of the server instrument."""

    def __call__(self, *args):
        """
        Call the Function.

        Args:
            *args: The positional args to this Function. Functions only take
                positional args, not kwargs.

        Returns:
            any: the return value of the function.
        """
        return self._instrument._ask_server('call', self.name, *args)

    def call(self, *args):
        """An alias for __call__."""
        return self.__call__(*args)

    def validate(self, *args):
        """
        Raise an error if the given args are not allowed for this Function.

        Args:
            *args: the proposed arguments with which to call the Function.
        """
        return self._instrument._ask_server(
            'callattr', self.name + '.validate', *args)
