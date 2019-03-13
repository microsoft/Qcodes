import logging
from typing import Sequence, Any, Dict, Callable, List, Union
import numpy as np
from functools import partial, wraps
from copy import copy, deepcopy, _reconstruct
from blinker import Signal

from qcodes.utils.helpers import DelegateAttributes, full_class
from qcodes.utils.metadata import Metadatable
from qcodes.config.config import DotDict
from qcodes.instrument.parameter import _BaseParameter, Parameter
from qcodes.instrument.function import Function

logger = logging.getLogger(__name__)


def parameter(fun):
    def parameter_decorator(self, *args, **kwargs):
        return fun(self, *args, **kwargs)
    return parameter_decorator


parameter_attrs = ['get', 'set', 'vals', 'get_parser', 'set_parser']
class ParameterNodeMetaClass(type):
    def __new__(meta, name, bases, dct):
        dct['_parameter_decorators'] = {}
        # Initialize parameter decorators with those of parent bases
        for base in bases:
            if hasattr(base, '_parameter_decorators'):
                dct['_parameter_decorators'].update(**base._parameter_decorators)

        for attr in list(dct):
            val = dct[attr]
            if getattr(val, '__name__', None) == 'parameter_decorator':
                for parameter_attr in parameter_attrs:
                    if attr.endswith('_'+parameter_attr):
                        parameter_name = attr[:-len(parameter_attr)-1]
                        attr_dict = dct['_parameter_decorators'].get(parameter_name, {})
                        attr_dict[parameter_attr] = val
                        dct['_parameter_decorators'][parameter_name] = attr_dict
                        break
                else:
                    raise SyntaxError('Parameter decorator must end with '
                                      '`_` + {parameter_attr} ')
                dct.pop(attr)
        return super(ParameterNodeMetaClass, meta).__new__(meta, name, bases, dct)


def __deepcopy__(self, memodict={}):
    """Deepcopy method for ParameterNode.

    The ParameterNode needs a custom __deepcopy__ since decorated parameters
    would need to be redecorated, and the optional `parent` ParameterNode should
    not be copied.

    This method is attached to the ParameterNode during instantiation
    """
    restore_attrs = {'__deepcopy__': self.__deepcopy__,
                     'parent': self.parent}
    try:
        for attr in restore_attrs:
            delattr(self, attr)

        self_copy = deepcopy(self)

        # Move deepcopy method to the instance scope, since it will temporarily
        # delete its own method during copying (see ParameterNode.__deepcopy__)
        self_copy.__deepcopy__ = partial(__deepcopy__, self_copy)
        self_copy.parent = None  # No parent by default

        for parameter_name, parameter in self_copy.parameters.items():
            if parameter_name in self._parameter_decorators:
                parameter_decorators = self._parameter_decorators[parameter_name]
                self_copy._attach_parameter_decorators(parameter, parameter_decorators)

        return self_copy
    finally:
        for attr_name, attr in restore_attrs.items():
            setattr(self, attr_name, attr)


class ParameterNode(Metadatable, DelegateAttributes, metaclass=ParameterNodeMetaClass):
    """ Container for parameters

    The ParameterNode is a container for `Parameters`, and is primarily used for
    an `Instrument`.

    Args:
        name: Optional name for parameter node
        use_as_attributes: Treat parameters as attributes (see below)
        log_changes: Log all changes of parameter values as debug messages
        simplify_snapshot: Snapshot contains simplified parameter snapshots

    A parameter can be added to a ParameterNode by settings its attribute:
    ``parameter_node.new_parameter = Parameter()``
    The name of the parameter is set to the attribute name

    Once a parameter has been added, its value can be set/get depending on the
    arg use_as_attributes. If use_as_attributes is False, calling
    ``parameter_node.new_parameter`` returns the Parameter object.
    The parameter is set using ``parameter_node.new_parameter(value)`` and
    retrieved via ``parameter_node.new_parameter()`` (same as you would for a
    parameter that does not belong to a Node).

    If ``use_as_attributes`` is True, its value can be set as such:
    ``parameter_node.new_parameter = 42``
    Note that this doesn't replace the parameter by 42, but instead sets the
    value of the parameter.

     Similarly, its value can be returned by accessing the attribute
     ``parameter_node.new_parameter`` (returns 42)
     Again, this doesn't return the parameter, but its value

    The parameter object can then be accessed via ``parameter_node['new_parameter']``
    """

    parameters = {}
    parameter_nodes = {}    #
    # attributes to delegate from dict attributes, for example:
    # instrument.someparam === instrument.parameters['someparam']
    delegate_attr_dicts = ['parameters', 'parameter_nodes', 'functions',
                           'submodules']

    def __init__(self, name: str = None,
                 use_as_attributes: bool = False,
                 log_changes: bool = True,
                 simplify_snapshot: bool = False,
                 parent: Union['ParameterNode', bool] = None,
                 **kwargs):
        # Move deepcopy method to the instance scope, since it will temporarily
        # delete its own method during copying (see ParameterNode.__deepcopy__)
        self.__deepcopy__ = partial(__deepcopy__, self)

        self.use_as_attributes = use_as_attributes
        self.log_changes = log_changes
        self.simplify_snapshot = simplify_snapshot

        if name is not None:
            self.name = name

        # Parent ParameterNode. Can also be None or False.
        # If set to None, it will be set the next time this node is attached as
        # an attribute of another Node.
        # If False, setting this Node to an attribute of another Node will not
        # set this attribute
        self.parent = parent

        self.parameters = DotDict()
        self.parameter_nodes = DotDict()
        self.functions = {}
        self.submodules = {}

        super().__init__(**kwargs)

        self._meta_attrs = ['name']

    def __str__(self):
        s = ''
        if getattr(self, 'name', None):
            if self.parent:
                s += f'{self.parent}_'
            if isinstance(self.name, _BaseParameter):
                s += f'{self.name()}'
            else:
                s += f'{self.name}'
        return s

    def __repr__(self):
        repr_str = f'ParameterNode {self} containing '
        if self.parameter_nodes:
            repr_str += f'{len(self.parameter_nodes)} nodes, '
        repr_str += f'{len(self.parameters)} parameters'
        return repr_str

    def __getattr__(self, attr):
        if attr == 'use_as_attributes':
            return super().__getattr__(attr)
        elif attr in self.parameters:
            parameter = self.parameters[attr]
            if self.use_as_attributes:
                # Perform get and return value
                return parameter()
            else:
                # Return parameter instance
                return parameter
        elif attr in self.parameter_nodes:
            return self.parameter_nodes[attr]
        else:
            return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if attr == 'parent':
            super().__setattr__(attr, val)
        elif isinstance(val, _BaseParameter):
            # Delete attr if it already exists so we can delegate param attribute
            if attr in self.__dict__:
                delattr(self, attr)

            self.parameters[attr] = val
            if val.parent is None:  # Attach self as parent if not already set
                val.parent = self

            if hasattr(self, 'multiple_senders'):
                val.multiple_senders = self.multiple_senders

            if attr in self._parameter_decorators:
                # Some methods have been defined in the ParameterNode as methods
                # Using the @parameter decorator.
                self._attach_parameter_decorators(
                    parameter=val,
                    decorator_methods=self._parameter_decorators[attr])
            if val.name == 'None':
                # Parameter has been created without name, update name to attr
                val.name = attr
                if val.label is None:
                    # For label, convert underscores to spaces and capitalize
                    label = attr.replace('_', ' ')
                    val.label = label[0].capitalize() + label[1:]
            val.log_changes = self.log_changes
        elif isinstance(val, ParameterNode):
            # Check if attribute is an @property
            if isinstance(getattr(type(self), attr, None), property):
                super().__setattr__(attr, val)
                return

            # Delete attr if it already exists so we can delegate node attribute
            if attr in self.__dict__:
                delattr(self, attr)

            self.parameter_nodes[attr] = val
            if val.parent is None:  # Attach self as parent if not already set
                val.parent = self
            if not hasattr(val, 'name'):
                # Update nested ParameterNode name
                val.name = attr
            val.log_changes = self.log_changes
        elif attr in self.parameters:
            # Set parameter value
            self.parameters[attr](val)
        else:
            super().__setattr__(attr, val)

    def __copy__(self):
        """Copy method for ParameterNode, invoked by copy.copy(parameter_node).
        """
        rv = self.__reduce_ex__(4)
        self_copy = _reconstruct(self, None, *rv)

        self_copy.__deepcopy__ = partial(__deepcopy__, self_copy)
        self_copy.parent = None  # By default no parent for copied nodes

        self_copy.parameters = {}
        for parameter_name, parameter in self.parameters.items():
            parameter_copy = copy(parameter)
            self_copy.parameters[parameter_name] = parameter_copy
            parameter_copy.parent = self_copy

        # Attach parameter decorators, done in a second loop because the
        # decorators may call parameters, and so all parameters must exist
        for parameter_name, parameter_decorators in self._parameter_decorators.items():
            parameter = self_copy.parameters[parameter_name]
            self_copy._attach_parameter_decorators(parameter, parameter_decorators)

        self_copy.parameter_nodes = {}
        for node_name, parameter_node in self_copy.parameter_nodes.items():
            parameter_node_copy = copy(parameter_node)
            self_copy.parameter_nodes[node_name] = parameter_node_copy
            parameter_node_copy.parent = self_copy

        return self_copy

    def __getitem__(self, key):
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            pass
        try:
            return self.functions[key]
        except KeyError:
            pass
        return super().__getitem__(key)

    def __dir__(self):
        # Add parameters to dir
        items = super().__dir__()
        items.extend(self.parameters)
        items.extend(self.parameter_nodes)
        return items

    def __eq__(self, other):
        return self.matches_parameter_node(other)

    def __getstate__(self):
        d = copy(self.__dict__)
        d['signal'] = Signal()
        d['parent'] = None
        return d

    def _attach_parameter_decorators(self,
                                     parameter: _BaseParameter,
                                     decorator_methods: Dict[str, Callable]):
        """Attaches @parameter decorators to a parameter

        Args:
            parameter: Parameter to attach decorators to
            decorator_methods: Decorator methods to attach to parameter
            """
        for param_attr, param_method in decorator_methods.items():
            method_with_args = partial(param_method, self, parameter)
            if param_attr == 'get':
                parameter.get_raw = method_with_args
                if parameter.wrap_get:
                    parameter.get = parameter._wrap_get(parameter.get_raw)
                else:
                    parameter.get = parameter.get_raw
            elif param_attr == 'set':
                parameter.set_raw = method_with_args
                if parameter.wrap_set:
                    parameter.set = parameter._wrap_set(parameter.set_raw)
                else:
                    parameter.set = parameter.set_raw
            else:
                setattr(parameter, param_attr, method_with_args)
        # perform a set without evaluating, which saves the value,
        # ensuring that new modifications such as the set_parser are
        # taken into account
        if hasattr(parameter, 'set') and parameter.raw_value is not None and parameter.wrap_set:
            parameter.set(copy(parameter.get_latest()), evaluate=False)

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
        if self.simplify_snapshot:
            snap = {"__class__": full_class(self)}
            if self.functions:
                snap["functions"] = {name: func.snapshot(update=update)
                                     for name, func in self.functions.items()}
            if self.submodules:
                snap["submodules"] = {name: subm.snapshot(update=update)
                                      for name, subm in self.submodules.items()}
            for parameter_name, parameter in self.parameters.items():
                parameter_snapshot = parameter.snapshot()
                if 'unit' in parameter_snapshot:
                    parameter_name = f'{parameter_name} ({parameter_snapshot["unit"]})'
                if parameter._snapshot_value:
                    snap[parameter_name] = parameter_snapshot['value']
                else:
                    snap[parameter_name] = parameter_snapshot
            for parameter_node_name, parameter_node in self.parameter_nodes.items():
                snap[parameter_node_name] = parameter_node.snapshot()
        else:
            snap = {
                "functions": {name: func.snapshot(update=update)
                              for name, func in self.functions.items()},
                "submodules": {name: subm.snapshot(update=update)
                               for name, subm in self.submodules.items()},
                "__class__": full_class(self),
                "parameters": {},
                "parameter_nodes": {name: node.snapshot()
                                    for name, node in self.parameter_nodes.items()}
            }

            for name, param in self.parameters.items():
                update = update
                if params_to_skip_update and name in params_to_skip_update:
                    update = False
                try:
                    snap['parameters'][name] = param.snapshot(
                        update=update, simplify=self.simplify_snapshot)
                except:
                    logger.debug(f"Snapshot: Could not update parameter: {name}")
                    snap['parameters'][name] = param.snapshot(
                        update=False, simplify=self.simplify_snapshot)
            for attr in set(self._meta_attrs):
                if hasattr(self, attr):
                    snap[attr] = getattr(self, attr)

        return snap

    def matches_parameter_node(self,
                                other: Any,
                                exclude_parameters: List[str] = [],
                                exclude_parameter_nodes: List[str] = []) -> bool:
        """Tests if the parameters of another node matches this one.

        Returns:
            False if any parameter values aren't equal
            False if the compared object is not exactly the same class.
            False if the number of parameters doesn't match
            True if all parameter values match
        """
        if self is other:
            return True

        if other.__class__ != self.__class__:
            return False
        elif len(self.parameters) != len(other.parameters):
            return False
        elif len(self.parameter_nodes) != len(other.parameter_nodes):
            return False

        for parameter_name, parameter in self.parameters.items():
            if parameter_name in exclude_parameters:
                continue
            elif not parameter_name in other.parameters \
                    or parameter.get_latest() != other.parameters[parameter_name].get_latest():
                return False

        for parameter_node_name, parameter_node in self.parameter_nodes.items():
            if parameter_node_name in exclude_parameter_nodes:
                continue
            elif not parameter_node_name in other.parameter_nodes \
                    or not parameter_node == other.parameter_nodes[parameter_node_name]:
                return False
        return True

    def sweep(self, parameter_name: str, start=None, stop=None, step=None,
              num=None, **kwargs):
        """ Sweep a parameter in the parameter node

        The following lines are identical:

        >>> parameter_node.param.sweep(start=0, stop=10, step=1)
        >>> parameter_node['param'].sweep(start=0, stop=10, step=1)
        >>> parameter_node.sweep('param', start=0, stop=10, step=1)

        Using the parameter node's sweep method is the recommended method,
        especially if parameter_node.use_as_attributes == True.

        Args:
            parameter_name: Name of parameter to sweep
            start: Sweep start value. Does not need to be set if window is set
            stop: Sweep stop value. Does not need to be set if window is set
            step: Optional sweep step. Does not need to be set if num or
                step_percentage is set
            num: Optional number of sweep values between start and stop. Does
                not need to be set if step or step_percentage is set
            **kwargs: Additional sweep kwargs, for SilQ QCoDeS these are:
                window: Optional sweep window around current value.
                    If set, start and stop do not need to be set
                step_percentage: Optional step percentage, calculated from silq
                    config (needs work, and does not work in unforked QCoDeS)
        """
        return self.parameters[parameter_name].sweep(start=start, stop=stop,
                                                     step=step, num=num,
                                                     **kwargs)

    def print_snapshot(self,
                       update: bool = False,
                       max_chars: int = 80,
                       level=0):
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

        pretabs = '\t'*level  # Number of tabs at the start of each line

        floating_types = (float, np.integer, np.floating)

        try:
            # Temporarily set simplify_snapshot to False for snapshotting
            simplify_snapshot = self.simplify_snapshot
            self.simplify_snapshot = False

            snapshot = self.snapshot(update=update)
        finally:
            self.simplify_snapshot = simplify_snapshot

        par_lengths = [len(p) for p in snapshot['parameters']]

        # Min of 50 is to prevent a super long parameter name to break this
        # function
        par_field_len = min(max(par_lengths)+1, 50)

        if str(self):
            print(pretabs + f'{self} :')
        print(pretabs + '{0:<{1}}'.format('\tparameter ', par_field_len) + 'value')
        print(pretabs + '-'*max_chars)
        for param_name, param_snapshot in sorted(snapshot['parameters'].items()):
            name = param_snapshot['name']
            msg = '{0:<{1}}:'.format(name, par_field_len)

            # in case of e.g. ArrayParameters, that usually have
            # snapshot_value == False, the parameter may not have
            # a value in the snapshot
            val = param_snapshot.get('value', 'Not available')
            if isinstance(val, floating_types):
                msg += f'\t{val:.5g} '
            else:
                msg += f'\t{val} '

            unit = param_snapshot.get('unit', None)
            if unit is None:
                # this may be a multi parameter
                unit = param_snapshot.get('units', None)
            if unit:
                msg += f'({unit})'

            # Truncate the message if it is longer than max length
            if len(msg) > max_chars and not max_chars == -1:
                msg = msg[0:max_chars-3] + '...'
            print(pretabs + msg)

        for submodule in self.submodules.values():
            if hasattr(submodule, '_channels'):
                if submodule._snapshotable:
                    for channel in submodule._channels:
                        channel.print_snapshot()
            else:
                submodule.print_snapshot(update, max_chars)

        for parameter_node in self.parameter_nodes.values():
            print(f'\n{parameter_node.name}')
            parameter_node.print_snapshot(level=level+1)

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
    def print_readable_snapshot(self, update=False, max_chars=80):
        logger.warning('print_readable_snapshot is replaced with print_snapshot')
        self.print_snapshot(update=update, max_chars=max_chars)

    def add_parameter(self, name, parameter_class=Parameter, parent=None, **kwargs):
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
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))

        if parent is None:
            parent = self

        param = parameter_class(name=name, parent=parent, **kwargs)
        self.parameters[name] = param

    def get(self, parameter):
        return self.parameters[parameter].get()