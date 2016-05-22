from qcodes.utils.helpers import is_sequence, permissive_range
from qcodes.utils.sync_async import mock_async, mock_sync


class SweepValues:
    '''
    base class for sweeping a parameter
    must be subclassed to provide the sweep values

    inputs:
        parameter: the target of the sweep, an object with
            set (and/or set_async), and optionally validate methods

    intended use is to iterate over in a sweep, so it must support:
        .__iter__ (and .__next__ if necessary).
        .set and .set_async are provided by the base class

    optionally, it can have a feedback method that allows the sweep to pass
    measurements back to this object for adaptive sampling:
        .feedback(set_values, measured_values)
    See AdaptiveSweep for an example

    example usage:
        for i, value in eumerate(sv):
            sv.set(value)  # or (await / yield from) sv.set_async(value)
                           # set(_async) just shortcuts sv.parameter.set
            sleep(delay)
            vals = measure()
            sv.feedback((i, ), vals) # optional - sweep should not assume
                                     # .feedback exists

    note though that sweeps should only require set, set_async, and
    __iter__ - ie "for val in sv", so any class that implements these
    may be used in sweeps. That allows things like adaptive sampling,
    where you don't know ahead of time what the values will be or even
    how many there are.
    '''
    def __init__(self, parameter):
        self.parameter = parameter
        self.name = parameter.name
        self._values = []

        # create the set and set_async shortcuts
        if hasattr(parameter, 'set'):
            self.set = parameter.set
        else:
            self.set = mock_sync(parameter.set_async)

        if hasattr(parameter, 'set_async'):
            self.set_async = parameter.set_async
        else:
            self.set_async = mock_async(parameter.set)

    def validate(self, values):
        '''
        check that all values are allowed for this Parameter
        '''
        if hasattr(self.parameter, 'validate'):
            for value in values:
                self.parameter.validate(value)

    def __iter__(self):
        '''
        must be overridden (along with __next__ if this returns self)
        by a subclass to tell how to iterate over these values
        '''
        raise NotImplementedError

    def __repr__(self):
        s = '<{}.{}: {} at {}>'.format(
            self.__module__,
            self.__class__.__name__,
            str(self.name),
            id(self))
        return s


class SweepFixedValues(SweepValues):
    '''
    a fixed collection of parameter values to be iterated over during a sweep.

    inputs:
        parameter: the target of the sweep, an object with
            set (and/or set_async), and optionally validate methods
        keys: one or a sequence of items, each of which can be:
            - a single parameter value
            - a sequence of parameter values
            - a slice object, which MUST include all three args

    a SweepFixedValues object is normally created by slicing a Parameter p:

        sv = p[1.2:2:0.01]  # slice notation
        sv = p[1, 1.1, 1.3, 1.6]  # explicit individual values
        sv = p[1.2:2:0.01, 2:3:0.02]  # sequence of slices
        sv = p[logrange(1,10,.01)]  # some function that returns a sequence

    you can also use list operations to modify these:

    sv += p[2:3:.01] (another SweepFixedValues of the same parameter)
    sv += [4, 5, 6] (a bare sequence)
    sv.extend(p[2:3:.01])
    sv.append(3.2)
    sv.reverse()
    sv2 = reversed(sv)
    sv3 = sv + sv2
    sv4 = sv.copy()

    note though that sweeps should only require set, set_async, and
    __iter__ - ie "for val in sv", so any class that implements these
    may be used in sweeps. That allows things like adaptive sampling,
    where you don't know ahead of time what the values will be or even
    how many there are.
    '''
    def __init__(self, parameter, keys):
        super().__init__(parameter)
        keyset = keys if is_sequence(keys) else (keys,)

        for key in keyset:
            if is_sequence(key):
                self._values.extend(key)
            elif isinstance(key, slice):
                if key.start is None or key.stop is None or key.step is None:
                    raise TypeError('all 3 slice parameters are required, ' +
                                    '{} is missing some'.format(key))
                self._values.extend(permissive_range(key.start, key.stop,
                                                     key.step))
            else:
                # assume a single value
                self._values.append(key)

        self.validate(self._values)

    def append(self, value):
        self.validate((value,))
        self._values.append(value)

    def extend(self, new_values):
        if isinstance(new_values, SweepFixedValues):
            if new_values.parameter is not self.parameter:
                raise TypeError(
                    'can only extend SweepFixedValues of the same parameters')
            # these values are already validated
            self._values.extend(new_values._values)
        elif is_sequence(new_values):
            self.validate(new_values)
            self._values.extend(new_values)
        else:
            raise TypeError(
                'cannot extend SweepFixedValues with {}'.format(new_values))

    def copy(self):
        new_sv = SweepFixedValues(self.parameter, [])
        # skip validation by adding values separately instead of on init
        new_sv._values = self._values[:]
        return new_sv

    def reverse(self):
        self._values.reverse()

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __len__(self):
        return len(self._values)

    def __add__(self, other):
        new_sv = self.copy()
        new_sv.extend(other)
        return new_sv

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __contains__(self, value):
        return value in self._values

    def __reversed__(self):
        new_sv = self.copy()
        new_sv.reverse()
        return new_sv
