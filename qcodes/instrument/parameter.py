"""
Measured and/or controlled parameters

Anything that you want to either measure or control within QCoDeS should
satisfy the Parameter interface. Most of the time that is easiest to do
by either using or subclassing one of the classes defined here, but you can
also use any class with the right attributes.

TODO (alexcjohnson) update this with the real duck-typing requirements or
create an ABC for Parameter and MultiParameter - or just remove this statement
if everyone is happy to use these classes.

This file defines four classes of parameters:

``Parameter``, ``ArrayParameter``, and ``MultiParameter`` must be subclassed:

- ``Parameter`` is the base class for scalar-valued parameters, if you have
    custom code to read or write a single value. Provides ``sweep`` and
    ``__getitem__`` (slice notation) methods to use a settable parameter as
    the swept variable in a ``Loop``. To use, fill in ``super().__init__``,
    and provide a ``get`` method, a ``set`` method, or both.

- ``ArrayParameter`` is a base class for array-valued parameters, ie anything
    for which each ``get`` call returns an array of values that all have the
    same type and meaning. Currently not settable, only gettable. Can be used
    in ``Measure``, or in ``Loop`` - in which case these arrays are nested
    inside the loop's setpoint array. To use, provide a ``get`` method that
    returns an array or regularly-shaped sequence, and describe that array in
    ``super().__init__``.

- ``MultiParameter`` is the base class for multi-valued parameters. Currently
    not settable, only gettable, but can return an arbitrary collection of
    scalar and array values and can be used in ``Measure`` or ``Loop`` to
    feed data to a ``DataSet``. To use, provide a ``get`` method
    that returns a sequence of values, and describe those values in
    ``super().__init__``.

``StandardParameter`` and ``ManualParameter`` can be instantiated directly:

- ``StandardParameter`` is the default class for instrument parameters
    (see ``Instrument.add_parameter``). Can be gettable, settable, or both.
    Provides a standardized interface to construct strings to pass to the
    instrument's ``write`` and ``ask`` methods (but can also be given other
    functions to execute on ``get`` or ``set``), to convert the string
    responses to meaningful output, and optionally to ramp a setpoint with
    stepped ``write`` calls from a single ``set``. Does not need to be
    subclassed, just instantiated.

- ``ManualParameter`` is for values you want to keep track of but cannot
    set or get electronically. Holds the last value it was ``set`` to, and
    returns it on ``get``.
"""

from datetime import datetime, timedelta
from copy import copy
import time
import logging
import os
import collections
import warnings

import numpy

from qcodes.utils.deferred_operations import DeferredOperations
from qcodes.utils.helpers import (permissive_range, wait_secs, is_sequence,
                                  is_sequence_of, DelegateAttributes,
                                  full_class, named_repr, warn_units)
from qcodes.utils.metadata import Metadatable
from qcodes.utils.command import Command, NoCommandError
from qcodes.utils.validators import Validator, Numbers, Ints, Enum, Strings
from qcodes.instrument.sweep_values import SweepFixedValues
from qcodes.data.data_array import DataArray


class _BaseParameter(Metadatable, DeferredOperations):
    """
    Shared behavior for simple and multi parameters. Not intended to be used
    directly, normally you should use ``StandardParameter`` or
    ``ManualParameter``, or create your own subclass of ``Parameter`` or
    ``MultiParameter``.

    Args:
        name (str): the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this should match how it will
            be referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        snapshot_get (Optional[bool]): False prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example if it takes too long to update.
            Default True.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """
    def __init__(self, name, instrument, snapshot_get, metadata,
                 snapshot_value=True):
        super().__init__(metadata)
        self._snapshot_get = snapshot_get
        self.name = str(name)
        self._instrument = instrument
        self._snapshot_value = snapshot_value

        self.has_get = hasattr(self, 'get')
        self.has_set = hasattr(self, 'set')

        if not (self.has_get or self.has_set):
            raise AttributeError('A parameter must have either a get or a '
                                 'set method, or both.')

        # record of latest value and when it was set or measured
        # what exactly this means is different for different subclasses
        # but they all use the same attributes so snapshot is consistent.
        self._latest_value = None
        self._latest_ts = None
        self.get_latest = GetLatest(self)

        # subclasses should extend this list with extra attributes they
        # want automatically included in the snapshot
        self._meta_attrs = ['name', 'instrument']

    def __repr__(self):
        return named_repr(self)

    def __call__(self, *args):
        if len(args) == 0:
            if self.has_get:
                return self.get()
            else:
                raise NotImplementedError('no get cmd found in' +
                                          ' Parameter {}'.format(self.name))
        else:
            if self.has_set:
                self.set(*args)
            else:
                raise NotImplementedError('no set cmd found in' +
                                          ' Parameter {}'.format(self.name))

    def _latest(self):
        return {
            'value': self._latest_value,
            'ts': self._latest_ts
        }

    # get_attrs ignores leading underscores, unless they're in this list
    _keep_attrs = ['__doc__', '_vals']

    def get_attrs(self):
        """
        Attributes recreated as properties in the RemoteParameter proxy.

        Grab the names of all attributes that the RemoteParameter needs
        to function like the main one (in loops etc)

        Returns:
            list: All public attribute names, plus docstring and _vals
        """
        out = []

        for attr in dir(self):
            # while we're keeping units as a deprecated attribute in some
            # classes, avoid calling it here so we don't get spurious errors
            if ((attr[0] == '_' and attr not in self._keep_attrs) or
                    (attr != 'units' and callable(getattr(self, attr)))):
                continue
            out.append(attr)

        return out

    def snapshot_base(self, update=False):
        """
        State of the parameter as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by calling
                parameter.get().
                If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
        """

        if self.has_get and self._snapshot_get and self._snapshot_value and \
                update:
            self.get()

        state = self._latest()
        state['__class__'] = full_class(self)

        if not self._snapshot_value:
            state.pop('value')

        if isinstance(state['ts'], datetime):
            state['ts'] = state['ts'].strftime('%Y-%m-%d %H:%M:%S')

        for attr in set(self._meta_attrs):
            if attr == 'instrument' and self._instrument:
                state.update({
                    'instrument': full_class(self._instrument),
                    'instrument_name': self._instrument.name
                })

            elif hasattr(self, attr):
                val = getattr(self, attr)
                attr_strip = attr.lstrip('_')  # eg _vals - do not include _
                if isinstance(val, Validator):
                    state[attr_strip] = repr(val)
                else:
                    state[attr_strip] = val

        return state

    def _save_val(self, value):
        self._latest_value = value
        self._latest_ts = datetime.now()

    @property
    def full_name(self):
        """Include the instrument name with the Parameter name if possible."""
        try:
            inst_name = self._instrument.name
            if inst_name:
                return inst_name + '_' + self.name
        except AttributeError:
            pass

        return self.name


class Parameter(_BaseParameter):
    """
    A parameter that represents a single degree of freedom.
    Not necessarily part of an instrument.

    Subclasses should define either a ``set`` method, a ``get`` method, or
    both.

    Parameters have a ``.get_latest`` method that simply returns the most
    recent set or measured value. This can be called ( ``param.get_latest()`` )
    or used in a ``Loop`` as if it were a (gettable-only) parameter itself:

        ``Loop(...).each(param.get_latest)``

    Note: If you want ``.get`` or ``.set`` to save the measurement for
    ``.get_latest``, you must explicitly call ``self._save_val(value)``
    inside ``.get`` and ``.set``.

    Args:
        name (str): the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        label (Optional[str]): Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit (Optional[str]): The unit of measure. Use ``''`` for unitless.

        units (Optional[str]): DEPRECATED, redirects to ``unit``.

        vals (Optional[Validator]): Allowed values for setting this parameter.
            Only relevant if settable. Defaults to ``Numbers()``

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (Optional[bool]): False prevents any update to the
            parameter during a snapshot, even if the snapshot was called with
            ``update=True``, for example if it takes too long to update.
            Default True.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """
    def __init__(self, name, instrument=None, label=None,
                 unit=None, units=None, vals=None, docstring=None,
                 snapshot_get=True, snapshot_value=True, metadata=None):
        super().__init__(name, instrument, snapshot_get, metadata,
                         snapshot_value=snapshot_value)

        self._meta_attrs.extend(['label', 'unit', '_vals'])

        self.label = name if label is None else label

        if units is not None:
            warn_units('Parameter', self)
            if unit is None:
                unit = units
        self.unit = unit if unit is not None else ''

        self.set_validator(vals)

        # generate default docstring
        self.__doc__ = os.linesep.join((
            'Parameter class:',
            '',
            '* `name` %s' % self.name,
            '* `label` %s' % self.label,
            '* `unit` %s' % self.unit,
            '* `vals` %s' % repr(self._vals)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

    def set_validator(self, vals):
        """
        Set a validator `vals` for this parameter.

        Args:
            vals (Validator):  validator to set
        """
        if vals is None:
            self._vals = Numbers()
        elif isinstance(vals, Validator):
            self._vals = vals
        else:
            raise TypeError('vals must be a Validator')

    def validate(self, value):
        """
        Validate value

        Args:
            value (any): value to validate

        """
        if self._instrument:
            context = (getattr(self._instrument, 'name', '') or
                       str(self._instrument.__class__)) + '.' + self.name
        else:
            context = self.name

        self._vals.validate(value, 'Parameter: ' + context)

    def sweep(self, start, stop, step=None, num=None):
        """
        Create a collection of parameter values to be iterated over.
        Requires `start` and `stop` and (`step` or `num`)
        The sign of `step` is not relevant.

        Args:
            start (Union[int, float]): The starting value of the sequence.
            stop (Union[int, float]): The end value of the sequence.
            step (Optional[Union[int, float]]):  Spacing between values.
            num (Optional[int]): Number of values to generate.

        Returns:
            SweepFixedValues: collection of parameter values to be
                iterated over

        Examples:
            >>> sweep(0, 10, num=5)
             [0.0, 2.5, 5.0, 7.5, 10.0]
            >>> sweep(5, 10, step=1)
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            >>> sweep(15, 10.5, step=1.5)
            >[15.0, 13.5, 12.0, 10.5]
        """
        return SweepFixedValues(self, start=start, stop=stop,
                                step=step, num=num)

    def __getitem__(self, keys):
        """
        Slice a Parameter to get a SweepValues object
        to iterate over during a sweep
        """
        return SweepFixedValues(self, keys)

    @property
    def units(self):
        warn_units('Parameter', self)
        return self.unit


class ArrayParameter(_BaseParameter):
    """
    A gettable parameter that returns an array of values.
    Not necessarily part of an instrument.

    Subclasses should define a ``.get`` method, which returns an array.
    When used in a ``Loop`` or ``Measure`` operation, this will be entered
    into a single ``DataArray``, with extra dimensions added by the ``Loop``.
    The constructor args describe the array we expect from each ``.get`` call
    and how it should be handled.

    For now you must specify upfront the array shape, and this cannot change
    from one call to the next. Later we intend to require only that you specify
    the dimension, and the size of each dimension can vary from call to call.

    Note: If you want ``.get`` to save the measurement for ``.get_latest``,
    you must explicitly call ``self._save_val(items)`` inside ``.get``.

    Args:
        name (str): the local name of the parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        shape (Tuple[int]): The shape (as used in numpy arrays) of the array
            to expect. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        label (Optional[str]): Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit (Optional[str]): The unit of measure. Use ``''`` for unitless.

        units (Optional[str]): DEPRECATED, redirects to ``unit``.

        setpoints (Optional[Tuple[setpoint_array]]):
            ``setpoint_array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each dimension of the returned array. An
            N-dimension item should have N setpoint arrays, where the first is
            1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            Note: if the setpoints will be different each measurement, leave
            this out and return the setpoints (with extra names) in ``.get``.

        setpoint_names (Optional[Tuple[str]]): one identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels (Optional[Tuple[str]]): one label (like ``labels``)
            per setpoint array. Ignored if a setpoint is a DataArray, which
            already has a label.

        setpoint_units (Optional[Tuple[str]]): one label (like ``v``)
            per setpoint array. Ignored if a setpoint is a DataArray, which
            already has a unit.

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (bool): Prevent any update to the parameter, for example
            if it takes too long to update. Default True.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """
    def __init__(self, name, shape, instrument=None,
                 label=None, unit=None, units=None,
                 setpoints=None, setpoint_names=None, setpoint_labels=None,
                 setpoint_units=None, docstring=None,
                 snapshot_get=True, snapshot_value=True, metadata=None):
        super().__init__(name, instrument, snapshot_get, metadata,
                         snapshot_value=snapshot_value)

        if self.has_set:  # TODO (alexcjohnson): can we support, ala Combine?
            raise AttributeError('ArrayParameters do not support set '
                                 'at this time.')

        self._meta_attrs.extend(['setpoint_names', 'setpoint_labels', 'setpoint_units',
                                 'label', 'unit'])

        self.label = name if label is None else label

        if units is not None:
            warn_units('ArrayParameter', self)
            if unit is None:
                unit = units
        self.unit = unit if unit is not None else ''

        nt = type(None)

        if not is_sequence_of(shape, int):
            raise ValueError('shapes must be a tuple of ints, not ' +
                             repr(shape))
        self.shape = shape

        # require one setpoint per dimension of shape
        sp_shape = (len(shape),)

        sp_types = (nt, DataArray, collections.Sequence,
                    collections.Iterator)
        if (setpoints is not None and
                not is_sequence_of(setpoints, sp_types, shape=sp_shape)):
            raise ValueError('setpoints must be a tuple of arrays')
        if (setpoint_names is not None and
                not is_sequence_of(setpoint_names, (nt, str), shape=sp_shape)):
            raise ValueError('setpoint_names must be a tuple of strings')
        if (setpoint_labels is not None and
                not is_sequence_of(setpoint_labels, (nt, str),
                                   shape=sp_shape)):
            raise ValueError('setpoint_labels must be a tuple of strings')
        if (setpoint_units is not None and
                not is_sequence_of(setpoint_units, (nt, str),
                                   shape=sp_shape)):
            raise ValueError('setpoint_units must be a tuple of strings')

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join((
            'Parameter class:',
            '',
            '* `name` %s' % self.name,
            '* `label` %s' % self.label,
            '* `unit` %s' % self.unit,
            '* `shape` %s' % repr(self.shape)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

    @property
    def units(self):
        warn_units('ArrayParameter', self)
        return self.unit


def _is_nested_sequence_or_none(obj, types, shapes):
    """Validator for MultiParameter setpoints/names/labels"""
    if obj is None:
        return True

    if not is_sequence_of(obj, tuple, shape=(len(shapes),)):
        return False

    for obji, shapei in zip(obj, shapes):
        if not is_sequence_of(obji, types, shape=(len(shapei),)):
            return False

    return True


class MultiParameter(_BaseParameter):
    """
    A gettable parameter that returns multiple values with separate names,
    each of arbitrary shape.
    Not necessarily part of an instrument.

    Subclasses should define a ``.get`` method, which returns a sequence of
    values. When used in a ``Loop`` or ``Measure`` operation, each of these
    values will be entered into a different ``DataArray``. The constructor
    args describe what data we expect from each ``.get`` call and how it
    should be handled. ``.get`` should always return the same number of items,
    and most of the constructor arguments should be tuples of that same length.

    For now you must specify upfront the array shape of each item returned by
    ``.get``, and this cannot change from one call to the next. Later we intend
    to require only that you specify the dimension of each item returned, and
    the size of each dimension can vary from call to call.

    Note: If you want ``.get`` to save the measurement for ``.get_latest``,
    you must explicitly call ``self._save_val(items)`` inside ``.get``.

    Args:
        name (str): the local name of the whole parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``

        names (Tuple[str]): A name for each item returned by a ``.get``
            call. Will be used as the basis of the ``DataArray`` names
            when this parameter is used to create a ``DataSet``.

        shapes (Tuple[Tuple[int]]): The shape (as used in numpy arrays) of
            each item. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        labels (Optional[Tuple[str]]): A label for each item. Normally used
            as the axis label when a component is graphed, along with the
            matching entry from ``units``.

        units (Optional[Tuple[str]]): The unit of measure for each item.
            Use ``''`` or ``None`` for unitless values.

        setpoints (Optional[Tuple[Tuple[setpoint_array]]]):
            ``setpoint_array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each returned array. An N-dimension item should
            have N setpoint arrays, where the first is 1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            Note: if the setpoints will be different each measurement, leave
            this out and return the setpoints (with extra names) in ``.get``.

        setpoint_names (Optional[Tuple[Tuple[str]]]): one identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels (Optional[Tuple[Tuple[str]]]): one label (like
            ``labels``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a label.

        setpoint_units (Optional[Tuple[Tuple[str]]]): one unit (like
            ``V``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a unit.

        docstring (Optional[str]): documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all

        snapshot_get (bool): Prevent any update to the parameter, for example
            if it takes too long to update. Default True.

        metadata (Optional[dict]): extra information to include with the
            JSON snapshot of the parameter
    """
    def __init__(self, name, names, shapes, instrument=None,
                 labels=None, units=None,
                 setpoints=None, setpoint_names=None, setpoint_labels=None,
                 setpoint_units=None, docstring=None,
                 snapshot_get=True, snapshot_value=True, metadata=None):
        super().__init__(name, instrument, snapshot_get, metadata,
                         snapshot_value=snapshot_value)

        if self.has_set:  # TODO (alexcjohnson): can we support, ala Combine?
            warnings.warn('MultiParameters do not fully support set '
                          'at this time.')

        self._meta_attrs.extend(['setpoint_names', 'setpoint_labels', 'setpoint_units',
                                 'names', 'labels', 'units'])

        if not is_sequence_of(names, str):
            raise ValueError('names must be a tuple of strings, not' +
                             repr(names))

        self.names = names
        self.labels = labels if labels is not None else names
        self.units = units if units is not None else [''] * len(names)

        nt = type(None)

        if (not is_sequence_of(shapes, int, depth=2) or
                len(shapes) != len(names)):
            raise ValueError('shapes must be a tuple of tuples '
                             'of ints, not ' + repr(shapes))
        self.shapes = shapes

        sp_types = (nt, DataArray, collections.Sequence,
                    collections.Iterator)
        if not _is_nested_sequence_or_none(setpoints, sp_types, shapes):
            raise ValueError('setpoints must be a tuple of tuples of arrays')

        if not _is_nested_sequence_or_none(setpoint_names, (nt, str), shapes):
            raise ValueError(
                'setpoint_names must be a tuple of tuples of strings')

        if not _is_nested_sequence_or_none(setpoint_labels, (nt, str), shapes):
            raise ValueError(
                'setpoint_labels must be a tuple of tuples of strings')

        if not _is_nested_sequence_or_none(setpoint_units, (nt, str), shapes):
            raise ValueError(
                'setpoint_units must be a tuple of tuples of strings')

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join((
            'MultiParameter class:',
            '',
            '* `name` %s' % self.name,
            '* `names` %s' % ', '.join(self.names),
            '* `labels` %s' % ', '.join(self.labels),
            '* `units` %s' % ', '.join(self.units)))

        if docstring is not None:
            self.__doc__ = os.linesep.join((
                docstring,
                '',
                self.__doc__))

    @property
    def full_names(self):
        """Include the instrument name with the Parameter names if possible."""
        try:
            inst_name = self._instrument.name
            if inst_name:
                return [inst_name + '_' + name for name in self.names]
        except AttributeError:
            pass

        return self.names


def no_setter(*args, **kwargs):
    raise NotImplementedError('This Parameter has no setter defined.')


def no_getter(*args, **kwargs):
    raise NotImplementedError(
        'This Parameter has no getter, use .get_latest to get the most recent '
        'set value.')


class StandardParameter(Parameter):
    """
    Define one measurement parameter.

    Args:
        name (str): the local name of this parameter

        instrument (Optional[Instrument]): the instrument this parameter
            belongs to, if any

        get_cmd (Optional[Union[str, function]]): a string or function to
            get this parameter. You can only use a string if an instrument is
            provided, then this string will be passed to instrument.ask

        get_parser ( Optional[function]): function to transform the response
            from get to the final output value.
            See also val_mapping

        set_cmd (Optional[Union[str, function]]): command to set this
            parameter, either:

            - a string (containing one field to .format, like "{}" etc)
              you can only use a string if an instrument is provided,
              this string will be passed to instrument.write
            - a function (of one parameter)

        set_parser (Optional[function]): function to transform the input set
            value to an encoded value sent to the instrument.
            See also val_mapping

        val_mapping (Optional[dict]): a bidirectional map data/readable values
            to instrument codes, expressed as a dict:
            ``{data_val: instrument_code}``
            For example, if the instrument uses '0' to mean 1V and '1' to mean
            10V, set val_mapping={1: '0', 10: '1'} and on the user side you
            only see 1 and 10, never the coded '0' and '1'

            If vals is omitted, will also construct a matching Enum validator.
            NOTE: only applies to get if get_cmd is a string, and to set if
            set_cmd is a string.

            You can use ``val_mapping`` with ``get_parser``, in which case
            ``get_parser`` acts on the return value from the instrument first,
            then ``val_mapping`` is applied (in reverse).

            You CANNOT use ``val_mapping`` and ``set_parser`` together - that
            would just provide too many ways to do the same thing.

        vals (Optional[Validator]): a Validator object for this parameter

        delay (Optional[Union[int, float]]): time (in seconds) to wait after
            the *start* of each set, whether part of a sweep or not. Can be
            set to 0 to go maximum speed with no errors.

        max_delay (Optional[Union[int, float]]): If > delay, we don't emit a
            warning unless the time taken during a single set is greater than
            this, even though we aim for delay.

        step (Optional[Union[int, float]]): max increment of parameter value.
            Larger changes are broken into multiple steps this size.

        max_val_age (Optional[Union[int, float]]): max time (in seconds) to
            trust a saved value from this parameter as the starting point of
            a sweep.

        **kwargs: Passed to Parameter parent class

    Raises:
        NoCommandError: if get and set are not found
    """
    def __init__(self, name, instrument=None,
                 get_cmd=None, get_parser=None,
                 set_cmd=None, set_parser=None,
                 delay=None, max_delay=None, step=None, max_val_age=3600,
                 vals=None, val_mapping=None, **kwargs):
        # handle val_mapping before super init because it impacts
        # vals / validation in the base class
        if val_mapping:
            if vals is None:
                vals = Enum(*val_mapping.keys())

            self._get_mapping = {v: k for k, v in val_mapping.items()}

            if get_parser is None:
                get_parser = self._valmapping_get_parser
            else:
                # First run get_parser, then run the result through
                # val_mapping
                self._get_preparser = get_parser
                get_parser = self._valmapping_with_preparser

            if set_parser is None:
                self._set_mapping = val_mapping
                set_parser = self._set_mapping.__getitem__
            else:
                raise TypeError(
                    'You cannot use set_parser and val_mapping together.')

        super().__init__(name=name, instrument=instrument, vals=vals, **kwargs)

        self._meta_attrs.extend(['sweep_step', 'sweep_delay',
                                 'max_sweep_delay'])

        # stored value from last .set() or .get()
        # normally only used by set with a sweep, to avoid
        # having to call .get() for every .set()
        self._max_val_age = 0

        self._set_get(get_cmd, get_parser)
        self._set_set(set_cmd, set_parser)
        self.set_delay(delay, max_delay)
        self.set_step(step, max_val_age)

        if not (self.has_get or self.has_set):
            raise NoCommandError('neither set nor get cmd found in' +
                                 ' Parameter {}'.format(self.name))

    def get(self):
        try:
            value = self._get()
            self._save_val(value)
            return value
        except Exception as e:
            e.args = e.args + ('getting {}'.format(self.full_name),)
            raise e

    def _valmapping_get_parser(self, val):
        """
        Get parser to be used in the case that a val_mapping is defined
        and a separate get_parser is not defined.

        Tries to match against defined strings in the mapping dictionary. If
        there are no matches, we try to convert the val into an integer.
        """

        # Try and match the raw value from the instrument directly
        try:
            return self._get_mapping[val]
        except KeyError:
            pass

        # If there is no match, we can try to convert the parameter into a
        # numeric value
        try:
            val = int(val)
            return self._get_mapping[val]
        except (ValueError, KeyError):
            raise KeyError('Unmapped value from instrument: {!r}'.format(val))

    def _valmapping_with_preparser(self, val):
        return self._valmapping_get_parser(self._get_preparser(val))

    def _set_get(self, get_cmd, get_parser):
        exec_str = self._instrument.ask if self._instrument else None
        self._get = Command(arg_count=0, cmd=get_cmd, exec_str=exec_str,
                            output_parser=get_parser,
                            no_cmd_function=no_getter)

        self.has_get = (get_cmd is not None)

    def _set_set(self, set_cmd, set_parser):
        # note: this does not set the final setter functions. that's handled
        # in self.set_sweep, when we choose a swept or non-swept setter.
        # TODO(giulioungaretti) lies! that method does not exis.
        # probably alexj left it out :(
        exec_str = self._instrument.write if self._instrument else None
        self._set = Command(arg_count=1, cmd=set_cmd, exec_str=exec_str,
                            input_parser=set_parser, no_cmd_function=no_setter)

        self.has_set = set_cmd is not None

    def _validate_and_set(self, value):
        try:
            clock = time.perf_counter()
            self.validate(value)
            self._set(value)
            self._save_val(value)
            if self._delay is not None:
                clock, remainder = self._update_set_ts(clock)
                time.sleep(remainder)
        except Exception as e:
            e.args = e.args + (
                'setting {} to {}'.format(self.full_name, repr(value)),)
            raise e

    def _sweep_steps(self, value):
        oldest_ok_val = datetime.now() - timedelta(seconds=self._max_val_age)
        state = self._latest()
        if state['ts'] is None or state['ts'] < oldest_ok_val:
            start_value = self.get()
        else:
            start_value = state['value']

        self.validate(start_value)

        if not (isinstance(start_value, (int, float)) and
                isinstance(value, (int, float))):
            # something weird... parameter is numeric but one of the ends
            # isn't, even though it's valid.
            # probably a MultiType with a mix of numeric and non-numeric types
            # just set the endpoint and move on
            logging.warning('cannot sweep {} from {} to {} - jumping.'.format(
                self.name, start_value, value))
            return []

        # drop the initial value, we're already there
        return permissive_range(start_value, value, self._step)[1:]

    def _update_set_ts(self, step_clock):
        # calculate the delay time to the *max* delay,
        # then take off up to the tolerance
        tolerance = self._delay_tolerance
        step_clock += self._delay
        remainder = wait_secs(step_clock + tolerance)
        if remainder <= tolerance:
            # don't allow extra delays to compound
            step_clock = time.perf_counter()
            remainder = 0
        else:
            remainder -= tolerance
        return step_clock, remainder

    def _validate_and_sweep(self, value):
        try:
            self.validate(value)
            step_clock = time.perf_counter()

            for step_val in self._sweep_steps(value):
                self._set(step_val)
                self._save_val(step_val)
                if self._delay is not None:
                    step_clock, remainder = self._update_set_ts(step_clock)
                    time.sleep(remainder)

            self._set(value)
            self._save_val(value)

            if self._delay is not None:
                step_clock, remainder = self._update_set_ts(step_clock)
                time.sleep(remainder)
        except Exception as e:
            e.args = e.args + (
                'setting {} to {}'.format(self.full_name, repr(value)),)
            raise e

    def set_step(self, step, max_val_age=None):
        """
        Configure whether this Parameter uses steps during set operations.
        If step is a positive number, this is the maximum value change
        allowed in one hardware call, so a single set can result in many
        calls to the hardware if the starting value is far from the target.

        Args:
            step (Union[int, float]): A positive number, the largest change
                allowed in one call. All but the final change will attempt to
                change by +/- step exactly

            max_val_age (Optional[int]): Only used with stepping, the max time
                (in seconds) to trust a saved value. If this parameter has not
                been set or measured more recently than this, it will be
                measured before starting to step, so we're confident in the
                value we're starting from.

        Raises:
            TypeError: if step is not numeric
            ValueError: if step is negative
            TypeError:  if step is not integer for an integer parameter
            TypeError: if step is not a number
            TypeError: if max_val_age is not numeric
            ValueError: if max_val_age is negative
        """
        if not step:
            # single-command setting
            self.set = self._validate_and_set

        elif not self._vals.is_numeric:
            raise TypeError('you can only step numeric parameters')
        elif step <= 0:
            raise ValueError('step must be positive')
        elif (isinstance(self._vals, Ints) and
                not isinstance(step, int)):
            raise TypeError(
                'step must be a positive int for an Ints parameter')
        elif not isinstance(step, (int, float)):
            raise TypeError('step must be a number')

        else:
            # stepped setting
            if max_val_age is not None:
                if not isinstance(max_val_age, (int, float)):
                    raise TypeError(
                        'max_val_age must be a number')
                if max_val_age < 0:
                    raise ValueError('max_val_age must be non-negative')
                self._max_val_age = max_val_age

            self._step = step
            self.set = self._validate_and_sweep

    def get_delay(self):
        """Return the delay time of this parameter. Also see `set_delay` """
        return self._delay

    def set_delay(self, delay, max_delay=None):
        """
        Configure this parameter with a delay between set operations.

        Typically used in conjunction with set_step to create an effective
        ramp rate, but can also be used without a step to enforce a delay
        after every set.
        If delay and max_delay are both None or 0, we never emit warnings
        no matter how long the set takes.

        Args:
            delay(Union[int, float]): the target time between set calls. The
                actual time will not be shorter than this, but may be longer
                if the underlying set call takes longer.

            max_delay(Optional[Union[int, float]]): if given, the longest time
                allowed for the underlying set call before we emit a warning.

        Raises:
            TypeError: If delay is not int nor float
            TypeError: If max_delay is not int nor float
            ValueError: If delay is negative
            ValueError: If max_delay is smaller than delay
        """
        if delay is None:
            delay = 0
        if not isinstance(delay, (int, float)):
            raise TypeError('delay must be a number')
        if delay < 0:
            raise ValueError('delay must not be negative')
        self._delay = delay

        if max_delay is not None:
            if not isinstance(max_delay, (int, float)):
                raise TypeError(
                    'max_delay must be a either  int or a float')
            if max_delay < delay:
                raise ValueError('max_delay must be no shorter than delay')
            self._delay_tolerance = max_delay - delay
        else:
            self._delay_tolerance = 0

        if not (self._delay or self._delay_tolerance):
            # denotes that we shouldn't follow the wait code or
            # emit any warnings
            self._delay = None


class ManualParameter(Parameter):
    """
    Define one parameter that reflects a manual setting / configuration.

    Args:
        name (str): the local name of this parameter

        instrument (Optional[Instrument]): the instrument this applies to,
            if any.

        initial_value (Optional[str]): starting value, may be None even if
            None does not pass the validator. None is only allowed as an
            initial value and cannot be set after initiation.

        **kwargs: Passed to Parameter parent class
    """
    def __init__(self, name, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, instrument=instrument, **kwargs)
        self._meta_attrs.extend(['initial_value'])

        if initial_value is not None:
            self.validate(initial_value)
            self._save_val(initial_value)

    def set(self, value):
        """
        Validate and saves value

        Args:
            value (any): value to validate and save
        """
        self.validate(value)
        self._save_val(value)

    def get(self):
        """ Return latest value"""
        return self._latest()['value']


class GetLatest(DelegateAttributes, DeferredOperations):
    """
    Wrapper for a Parameter that just returns the last set or measured value
    stored in the Parameter itself.

    Examples:
        >>> # Can be called:
        >>> param.get_latest()
        >>> # Or used as if it were a gettable-only parameter itself:
        >>> Loop(...).each(param.get_latest)

    Args:
        parameter (Parameter): Parameter to be wrapped
    """
    def __init__(self, parameter):
        self.parameter = parameter

    delegate_attr_objects = ['parameter']
    omit_delegate_attrs = ['set']

    def get(self):
        """ Return latest value"""
        return self.parameter._latest()['value']

    def __call__(self):
        return self.get()


def combine(*parameters, name, label=None, unit=None, units=None,
            aggregator=None):
    """
    Combine parameters into one sweepable parameter

    Args:
        *paramters (qcodes.Parameter): the parameters to combine
        name (str): the name of the paramter
        label (Optional[str]): the label of the combined parameter
        unit (Optional[str]): the unit of the combined parameter
        aggregator (Optional[Callable[list[any]]]): a function to aggregate
            the set values into one

    A combined parameter sets all the combined parameters at every point of the
    sweep.
    The sets are called in the same order the parameters are, and
    sequantially.
    """
    parameters = list(parameters)
    multi_par = CombinedParameter(parameters, name, label, unit, units,
                                  aggregator)
    return multi_par


class CombinedParameter(Metadatable):
    """ A combined parameter

    Args:
        *parameters (qcodes.Parameter): the parameters to combine
        name (str): the name of the parameter
        label (Optional[str]): the label of the combined parameter
        unit (Optional[str]): the unit of the combined parameter
        aggregator (Optional[Callable[list[any]]]): a function to aggregate
            the set values into one

    A combined parameter sets all the combined parameters at every point of the
    sweep.
    The sets are called in the same order the parameters are, and
    sequentially.
    """

    def __init__(self, parameters, name, label=None,
                 unit=None, units=None, aggregator=None):
        super().__init__()
        # TODO(giulioungaretti)temporary hack
        # starthack
        # this is a dummy parameter
        # that mimicks the api that a normal parameter has
        self.parameter = lambda: None
        self.parameter.full_name = name
        self.parameter.name = name
        self.parameter.label = label

        if units is not None:
            warn_units('CombinedParameter', self)
            if unit is None:
                unit = units
        self.parameter.unit = unit
        # endhack
        self.parameters = parameters
        self.sets = [parameter.set for parameter in self.parameters]
        self.dimensionality = len(self.sets)

        if aggregator:
            self.f = aggregator
            setattr(self, 'aggregate', self._aggregate)

    def set(self, index: int):
        """
        Set multiple parameters.

        Args:
            index (int): the index of the setpoints one wants to set

        Returns:
            list: values that where actually set
        """
        values = self.setpoints[index]
        for setFunction, value in zip(self.sets, values):
            setFunction(value)
        return values

    def sweep(self, *array: numpy.ndarray):
        """
        Creates a new combined parameter to be iterated over.
        One can sweep over either:

         - n array of lenght m
         - one nxm array

        where n is the number of combined parameters
        and m is the number of setpoints

        Args:
            *array(numpy.ndarray): array(s) of setopoints

        Returns:
            MultiPar: combined parameter
        """
        # if it's a list of arrays, convert to one array
        if len(array) > 1:
            dim = set([len(a) for a in array])
            if len(dim) != 1:
                raise ValueError('Arrays have different number of setpoints')
            array = numpy.array(array).transpose()
        else:
            # cast to array in case users
            # decide to not read docstring
            # and pass a 2d list
            array = numpy.array(array[0])
        new = copy(self)
        _error_msg = """ Dimensionality of array does not match\
                        the number of parameter combined. Expected a \
                        {} dimensional array, got a {} dimensional array. \
                        """
        try:
            if array.shape[1] != self.dimensionality:
                raise ValueError(_error_msg.format(self.dimensionality,
                                                   array.shape[1]))
        except KeyError:
            # this means the array is 1d
            raise ValueError(_error_msg.format(self.dimensionality, 1))

        new.setpoints = array.tolist()
        return new

    def _aggregate(self, *vals):
        # check f args
        return self.f(*vals)

    def __iter__(self):
        return iter(range(len(self.setpoints)))

    def __len__(self):
        # dimension of the sweep_values
        # i.e. how many setpoint
        return numpy.shape(self.setpoints)[0]

    def snapshot_base(self, update=False):
        """
        State of the combined parameter as a JSON-compatible dict.

        Args:
            update (bool):

        Returns:
            dict: base snapshot
        """
        meta_data = collections.OrderedDict()
        meta_data['__class__'] = full_class(self)
        meta_data['unit'] = self.parameter.unit
        meta_data['label'] = self.parameter.label
        meta_data['full_name'] = self.parameter.full_name
        meta_data['aggreagator'] = repr(getattr(self, 'f', None))
        for param in self.parameters:
            meta_data[param.full_name] = param.snapshot()

        return meta_data


class InstrumentRefParameter(ManualParameter):
    """
    An InstrumentRefParameter

    Args:
        name (string): the name of the parameter that one wants to add.

        instrument (Optional[Instrument]): the "parent" instrument this
            parameter is attached to, if any.

        initial_value (Optional[str]): starting value, may be None even if
            None does not pass the validator. None is only allowed as an
            initial value and cannot be set after initiation.

        **kwargs: Passed to InstrumentRefParameter parent class

    This parameter is useful when one needs a reference to another instrument
    from within an instrument, e.g., when creating a meta instrument that
    sets parameters on instruments it contains.
    """

    def get_instr(self):
        """
        Returns the instance of the instrument with the name equal to the
        value of this parameter.
        """
        ref_instrument_name = self.get()
        # note that _instrument refers to the instrument this parameter belongs
        # to, while the ref_instrument_name is the instrument that is the value
        # of this parameter.
        return self._instrument.find_instrument(ref_instrument_name)

    def set_validator(self, vals):
        """
        Set a validator `vals` for this parameter.

        Args:
            vals (Validator):  validator to set
        """
        if vals is None:
            self._vals = Strings()
        elif isinstance(vals, Validator):
            self._vals = vals
        else:
            raise TypeError('vals must be a Validator')
