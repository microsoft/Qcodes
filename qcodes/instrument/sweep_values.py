from copy import deepcopy
from typing import Any, TYPE_CHECKING, List, Optional, Union, Iterator, Dict, \
    Sequence

from qcodes.utils.helpers import (is_sequence, permissive_range, make_sweep,
                                  named_repr)
from qcodes.utils.metadata import Metadatable
if TYPE_CHECKING:
    from qcodes.instrument.parameter import _BaseParameter


class SweepValues(Metadatable):
    """
    Base class for sweeping a parameter.

    Must be subclassed to provide the sweep values
    Intended use is to iterate over in a sweep, so it must support:

    >>> .__iter__ # (and .__next__ if necessary).
    >>> .set # is provided by the base class

    Optionally, it can have a feedback method that allows the sweep to pass
    measurements back to this object for adaptive sampling:

    >>> .feedback(set_values, measured_values)

    Todo:
        - Link to adawptive sweep

    Args:
        parameter: the target of the sweep, an object with
         set, and optionally validate methods

        **kwargs: Passed on to Metadatable parent

    Raises:
        TypeError: when parameter is not settable

    See AdaptiveSweep for an example

    example usage:

    >>> for i, value in eumerate(sv):
            sv.set(value)
            sleep(delay)
            vals = measure()
            sv.feedback((i, ), vals) # optional - sweep should not assume
                                     # .feedback exists

    note though that sweeps should only require set and __iter__ - ie
    "for val in sv", so any class that implements these may be used in sweeps.

    That allows things like adaptive sampling, where you don't know ahead of
    time what the values will be or even how many there are.
    """
    def __init__(self, parameter: '_BaseParameter', **kwargs: Any):
        super().__init__(**kwargs)
        self.parameter = parameter
        self.name = parameter.name
        self._values: List[Any] = []

        # allow has_set=False to override the existence of a set method,
        # but don't require it to be present (and truthy) otherwise
        if not (getattr(parameter, 'set', None) and
                getattr(parameter, 'has_set', True)):
            raise TypeError(f'parameter {parameter} is not settable')

        self.set = parameter.set

    def validate(self, values: Sequence[Any]) -> None:
        """
        Check that all values are allowed for this Parameter.

        Args:
            values: values to be validated.
        """
        if hasattr(self.parameter, 'validate'):
            for value in values:
                self.parameter.validate(value)

    def __iter__(self) -> Iterator:
        """
        must be overridden (along with __next__ if this returns self)
        by a subclass to tell how to iterate over these values
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return named_repr(self)


class SweepFixedValues(SweepValues):
    """
    A fixed collection of parameter values to be iterated over during a sweep.

    Args:
        parameter: the target of the sweep, an object with set and
            optionally validate methods

        keys: one or a sequence of items, each of which can be:
            - a single parameter value
            - a sequence of parameter values
            - a slice object, which MUST include all three args

        start: The starting value of the sequence.
        stop: The end value of the sequence.
        step:  Spacing between values.
        num: Number of values to generate.


    A SweepFixedValues object is normally created by slicing a Parameter p:

    >>>  sv = p[1.2:2:0.01]  # slice notation
    sv = p[1, 1.1, 1.3, 1.6]  # explicit individual values
    sv = p[1.2:2:0.01, 2:3:0.02]  # sequence of slices
    sv = p[logrange(1,10,.01)]  # some function that returns a sequence

    You can also use list operations to modify these:

    >>> sv += p[2:3:.01] # (another SweepFixedValues of the same parameter)
    sv += [4, 5, 6] # (a bare sequence)
    sv.extend(p[2:3:.01])
    sv.append(3.2)
    sv.reverse()
    sv2 = reversed(sv)
    sv3 = sv + sv2
    sv4 = sv.copy()

    note though that sweeps should only require set and __iter__ - ie
    "for val in sv", so any class that implements these may be used in sweeps.
    That allows things like adaptive sampling, where you don't know ahead of
    time what the values will be or even how many there are.
    """
    def __init__(self, parameter: '_BaseParameter',
                 keys: Optional[Any] = None,
                 start: Optional[float] = None,
                 stop: Optional[float] = None,
                 step: Optional[float] = None,
                 num: Optional[int] = None):
        super().__init__(parameter)
        self._snapshot: Dict[str, Any] = {}
        self._value_snapshot: List[Dict[str, Any]] = []

        if keys is None:
            if start is None:
                raise ValueError('If keys is None, start needs to be not None.')
            if stop is None:
                raise ValueError('If keys is None, stop needs to be not None.')
            keys = make_sweep(start=start, stop=stop,
                              step=step, num=num)
            self._values = keys
            self._add_linear_snapshot(self._values)

        elif isinstance(keys, slice):
            self._add_slice(keys)
            self._add_linear_snapshot(self._values)

        elif is_sequence(keys):
            for key in keys:
                if isinstance(key, slice):
                    self._add_slice(key)
                elif is_sequence(key):
                    # not sure if we really need to support this (and I'm not
                    # going to recurse any more!) but we will get nested lists
                    # if for example someone does `p[list1, list2]`
                    self._values.extend(key)
                else:
                    # assume a single value
                    self._values.append(key)
            # we dont want the snapshot to go crazy on big data
            if self._values:
                self._add_sequence_snapshot(self._values)

        else:
            # assume a single value
            self._values.append(keys)
            self._value_snapshot.append({'item': keys})

        self.validate(self._values)

    def _add_linear_snapshot(self, vals: List[Any]) -> None:
        self._value_snapshot.append({'first': vals[0],
                                     'last': vals[-1],
                                     'num': len(vals),
                                     'type': 'linear'})

    def _add_sequence_snapshot(self, vals: Sequence[Any]) -> None:
        self._value_snapshot.append({'min': min(vals),
                                     'max': max(vals),
                                     'first': vals[0],
                                     'last': vals[-1],
                                     'num': len(vals),
                                     'type': 'sequence'})

    def _add_slice(self, slice_: slice) -> None:
        if slice_.start is None or slice_.stop is None or slice_.step is None:
            raise TypeError('all 3 slice parameters are required, ' +
                            f'{slice_} is missing some')
        p_range = permissive_range(slice_.start, slice_.stop, slice_.step)
        self._values.extend(p_range)

    def append(self, value: Any) -> None:
        """
        Append a value.

        Args:
            value: new value to append
        """
        self.validate((value,))
        self._values.append(value)
        self._value_snapshot.append({'item': value})

    def extend(self, new_values: Union[Sequence, 'SweepFixedValues']) -> None:
        """
        Extend sweep with new_values

        Args:
            new_values: new values to append

        Raises:
            TypeError: if new_values is not Sequence, nor SweepFixedValues
        """
        if isinstance(new_values, SweepFixedValues):
            if new_values.parameter is not self.parameter:
                raise TypeError(
                    'can only extend SweepFixedValues of the same parameters')
            # these values are already validated
            self._values.extend(new_values._values)
            self._value_snapshot.extend(new_values._value_snapshot)
        elif is_sequence(new_values):
            self.validate(new_values)
            self._values.extend(new_values)
            self._add_sequence_snapshot(new_values)
        else:
            raise TypeError(
                f'cannot extend SweepFixedValues with {new_values}')

    def copy(self) -> 'SweepFixedValues':
        """
        Copy this SweepFixedValues.

        Returns:
            SweepFixedValues of copied values
        """
        new_sv = SweepFixedValues(self.parameter, [])
        # skip validation by adding values and snapshot separately
        # instead of on init
        new_sv._values = self._values[:]
        new_sv._value_snapshot = deepcopy(self._value_snapshot)
        return new_sv

    def reverse(self) -> None:
        """ Reverse SweepFixedValues in place. """
        self._values.reverse()
        self._value_snapshot.reverse()
        for snap in self._value_snapshot:
            if 'first' in snap and 'last' in snap:
                snap['last'], snap['first'] = snap['first'], snap['last']

    def snapshot_base(self, update: Optional[bool] = False,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict:
        """
        Snapshot state of SweepValues.

        Args:
            update: Place holder for API compatibility.
            params_to_skip_update: Place holder for API compatibility.

        Returns:
            dict: base snapshot
        """
        self._snapshot['parameter'] = self.parameter.snapshot(update=update)
        self._snapshot['values'] = self._value_snapshot
        return self._snapshot

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __getitem__(self, key: slice) -> Any:
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values)

    def __add__(self, other: Union[Sequence, 'SweepFixedValues']
                ) -> 'SweepFixedValues':
        new_sv = self.copy()
        new_sv.extend(other)
        return new_sv

    def __iadd__(self, values: Union[Sequence, 'SweepFixedValues']
                 ) -> 'SweepFixedValues':
        self.extend(values)
        return self

    def __contains__(self, value: float) -> bool:
        return value in self._values

    def __reversed__(self) -> 'SweepFixedValues':
        new_sv = self.copy()
        new_sv.reverse()
        return new_sv
