import logging
from typing import Sequence, Any
import numpy as np

from qcodes.utils.helpers import DelegateAttributes, full_class
from qcodes.utils.metadata import Metadatable
from qcodes.config.config import DotDict
from qcodes.instrument.parameter import _BaseParameter, Parameter
from qcodes.instrument.function import Function

logger = logging.getLogger(__name__)


class ParameterNode(Metadatable, DelegateAttributes):
    parameters = {}

    def __init__(self, name: str = None,
                 use_as_attributes: bool = False,
                 **kwargs):
        self.use_as_attributes = use_as_attributes

        self.name = name

        self.parameters = DotDict()
        self.parameter_nodes = DotDict()
        self.submodules = {}
        self.functions = {}

        super().__init__(**kwargs)

        self._meta_attrs = ['name']

    def __repr__(self):
        return 'ParameterNode {} containing {} parameters, {} nodes'.format(
            self.name, len(self.parameters), len(self.parameter_nodes))

    def __call__(self) -> dict:
        return self.parameters

    def __getattr__(self, attr):
        if attr == 'use_as_attributes':
            return super().__getattr__(attr)
        elif attr in self.parameter_nodes:
            return self.parameter_nodes[attr]
        elif attr in self.parameters:
            parameter = self.parameters[attr]
            if self.use_as_attributes:
                # Perform get and return value
                return parameter()
            else:
                # Return parameter instance
                return parameter
        else:
            return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if isinstance(val, _BaseParameter):
            self.parameters[attr] = val
            if val.name == 'None':
                # Parameter has been created without name, update name to attr
                val.name = attr
                if val.label is None:
                    # For label, convert underscores to spaces and capitalize
                    label = attr.replace('_', ' ')
                    label = label[0].capitalize() + label[1:]
                    val.label = label
        elif isinstance(val, ParameterNode):
            self.parameter_nodes[attr] = val
            # Update nested ParameterNode name
            val.name = attr
        elif attr in self.parameters:
            # Set parameter value
            self.parameters[attr](val)
        else:
            super().__setattr__(attr, val)

    # def __dir__(self):
    #     # Add parameters to dir
    #     items = super().__dir__()
    #     items.extend(self.parameters.keys())
    #     return items

    def add_function(self, name: str, **kwargs):
        """ Bind one Function to this parameter node.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real function of the instrument.

        This functionality is meant for simple cases, principally things that
        map to simple commands like '\*RST' (reset) or those with just a few
        arguments. It requires a fixed argument count, and positional args
        only. If your case is more complicated, you're probably better off
        simply making a new method in your ``Instrument`` subclass definition.

        Args:
            name: how the Function will be stored within
            ``instrument.Functions`` and also how you  address it using the
            shortcut methods: ``parameter_node.call(func_name, *args)`` etc.

            **kwargs: constructor kwargs for ``Function``

        Raises:
            KeyError: if this instrument already has a function with this
                name.
        """
        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

    def add_submodule(self, name: str, submodule: Metadatable):
        """ Bind one submodule to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        method for every submodule of the instrument.

        Submodules can effectively be considered as instruments within the main
        instrument, and should at minimum be snapshottable. For example, they
        can be used to either store logical groupings of parameters, which may
        or may not be repeated, or channel lists.

        Args:
            name: how the submodule will be stored within
                `instrument.submodules` and also how it can be addressed.

            submodule: The submodule to be stored.

        Raises:
            KeyError: if this instrument already contains a submodule with this
                name.
            TypeError: if the submodule that we are trying to add is not an
                instance of an Metadatable object.
        """
        if name in self.submodules:
            raise KeyError('Duplicate submodule name {}'.format(name))
        if not isinstance(submodule, Metadatable):
            raise TypeError('Submodules must be metadatable.')
        self.submodules[name] = submodule

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)

        Returns:
            dict: base snapshot
        """

        snap = {
            "functions": {name: func.snapshot(update=update)
                          for name, func in self.functions.items()},
            "submodules": {name: subm.snapshot(update=update)
                           for name, subm in self.submodules.items()},
            "__class__": full_class(self)
        }

        snap['parameters'] = {}
        for name, param in self.parameters.items():
            update = update
            if params_to_skip_update and name in params_to_skip_update:
                update = False
            try:
                snap['parameters'][name] = param.snapshot(update=update)
            except:
                logging.info("Snapshot: Could not update parameter:", name)
                snap['parameters'][name] = param.snapshot(update=False)
        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    def print_snapshot(self,
                       update: bool = False,
                       max_chars: int = 80):
        """ Prints a readable version of the snapshot.

        The readable snapshot includes the name, value and unit of each
        parameter.
        A convenience function to quickly get an overview of the parameter node.

        Args:
            update: If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
                This argument gets passed to the snapshot function.
            max_chars: the maximum number of characters per line. The
                readable snapshot will be cropped if this value is exceeded.
                Defaults to 80 to be consistent with default terminal width.
        """
        floating_types = (float, np.integer, np.floating)
        snapshot = self.snapshot(update=update)

        par_lengths = [len(p) for p in snapshot['parameters']]

        # Min of 50 is to prevent a super long parameter name to break this
        # function
        par_field_len = min(max(par_lengths)+1, 50)

        print(str(self.name)+ ':')
        print('{0:<{1}}'.format('\tparameter ', par_field_len) + 'value')
        print('-'*max_chars)
        for par in sorted(snapshot['parameters']):
            name = snapshot['parameters'][par]['name']
            msg = '{0:<{1}}:'.format(name, par_field_len)

            # in case of e.g. ArrayParameters, that usually have
            # snapshot_value == False, the parameter may not have
            # a value in the snapshot
            val = snapshot['parameters'][par].get('value', 'Not available')

            unit = snapshot['parameters'][par].get('unit', None)
            if unit is None:
                # this may be a multi parameter
                unit = snapshot['parameters'][par].get('units', None)
            if isinstance(val, floating_types):
                msg += '\t{:.5g} '.format(val)
            else:
                msg += '\t{} '.format(val)
            if unit is not '':  # corresponds to no unit
                msg += '({})'.format(unit)
            # Truncate the message if it is longer than max length
            if len(msg) > max_chars and not max_chars == -1:
                msg = msg[0:max_chars-3] + '...'
            print(msg)

        for submodule in self.submodules.values():
            if hasattr(submodule, '_channels'):
                if submodule._snapshotable:
                    for channel in submodule._channels:
                        channel.print_readable_snapshot()
            else:
                submodule.print_readable_snapshot(update, max_chars)

    #
    # shortcuts to parameters & setters & getters                           #
    # instrument.someparam === instrument.parameters['someparam']           #
    # instrument.get('someparam') === instrument['someparam'].get()         #
    # etc...                                                                #
    #
    delegate_attr_dicts = ['parameters', 'parameter_nodes', 'functions',
                           'submodules']

    def __getitem__(self, key):
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            return self.functions[key]

    def call(self, func_name: str, *args, **kwargs):
        """ Shortcut for calling a function from its name.

        Args:
            func_name: The name of a function of this instrument.
            *args: any arguments to the function.
            **kwargs: any keyword arguments to the function.

        Returns:
            any: The return value of the function.
        """
        return self.functions[func_name].call(*args, **kwargs)

    def __getstate__(self):
        """Prevent pickling instruments by raising an error."""
        raise RuntimeError('Instrumentts cannot be pickled')

    def validate_status(self,
                        verbose: bool = False):
        """ Validate the values of all gettable parameters

        The validation is done for all parameters that have both a get and
        set method.

        Arguments:
            verbose: If True, information of checked parameters is printed.

        """
        for k, p in self.parameters.items():
            if hasattr(p, 'get') and hasattr(p, 'set'):
                value = p.get()
                if verbose:
                    print('validate_status: param %s: %s' % (k, value))
                p.validate(value)

    # Deprecated methods
    def set(self, param_name: str, value: Any):
        """ Shortcut for setting a parameter from its name and new value.

        Args:
            param_name: The name of a parameter of this instrument.
            value: The new value to set.
        """
        logger.warning('parameter_node.set is deprecated.')
        self.parameters[param_name].set(value)

    def get(self, param_name: str):
        """ Shortcut for getting a parameter from its name.

        Args:
            param_name: The name of a parameter of this instrument.

        Returns:
            any: The current value of the parameter.
        """
        logger.warning('parameter_node.get is deprecated.')
        return self.parameters[param_name].get()

    def print_readable_snapshot(self, update=False, max_chars=80):
        logger.warning('print_readable_snapshot is replaced with print_snapshot')
        self.print_snapshot(update=update, max_chars=max_chars)

    def add_parameter(self, name, parameter_class=Parameter, **kwargs):
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

        Raises:
            KeyError: if this instrument already has a parameter with this
                name.
        """
        logger.warning('adding parameter should be done by setting the '
                       'attribute instead of via add_parameter')
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))
        param = parameter_class(name=name, instrument=self, **kwargs)
        self.parameters[name] = param