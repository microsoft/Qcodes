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


class AdaptiveSweep(SweepValues):
    '''
    an example class to show how adaptive sampling might be implemented

    usage:
    Loop(AdaptiveSweep(param, start, end, target_delta), delay).run()

    inputs:
        start: initial parameter value
        end: final parameter value
        target_delta: change in the measured val to target
        max_step: biggest change in parameter value allowed
        min_step: smallest change allowed, so we don't sweep forever
        measurement_index: which measurement parameter are we feeding back on?
    '''
    def __init__(self, parameter, start, end, target_delta, max_step=None,
                 min_step=None, measurement_index=0):
        super().__init__(parameter)

        self._start = start
        self._end = end
        self._direction = 1 if end > start else -1

        self._target_delta = target_delta

        self._max_step = max_step or abs(end - start) / 100
        self._min_step = min_step or self._max_step / 100

        self._measurement_index = measurement_index

    def __iter__(self):
        '''
        start or restart the adaptive algorithm
        called at the beginning of "for ... in ..."

        in principle, each iteration could base its outputs
        on the previous iteration, for example to follow peaks
        that move slowly as a function of the outer loop parameter.
        but in this simple example we're just basing each point on the
        previous two
        '''
        self._setting = None
        self._step = None
        self._measured = None
        self._delta = None
        self._new_val_count = 0

        # return self so iteration will call our own __next__
        return self

    def feedback(self, set_values, measured_values):
        '''
        the sweep routine will look for a .feedback method
        to pass new measurements into the SweepValues object

        it provides:
            set_values: sequence of the current sweep parameters
            measured_values: sequence of the measured values at this setting
        '''
        self._new_val_count += 1
        if self._new_val_count > 1:
            # more than one measurement per iteration means we're
            # not in the inner loop. in principle one could adaptively
            # sample an outer loop too, using the whole line of inner loop
            # measurements, but the algorithm here only applies to the inner.
            raise RuntimeError(
                'AdaptiveSweep can only be used on the inner loop')

        new_measured = measured_values[self._measurement_index]

        if self._measured is not None:
            self._delta = new_measured - self._measured

        self._measured = new_measured

    def __next__(self):
        self._new_val_count = 0

        if self._setting == self._end:
            # terminate the iteration if we've already set the endpoint
            raise StopIteration

        # target the step so the next delta is target_delta, if data is linear
        if self._delta is None:
            step = self._min_step  # start off slow
        else:
            step = abs(self._step * self._target_delta / self._delta)
            # don't increase too much at once
            step = max(step, self._step * 3)

        # constrain it to provide min and max
        step = min(max(self._min_step, step), self._max_step)
        self._setting += self._direction * step

        # stop at the end
        if self._setting * self._direction > self._end * self._direction:
            self._setting = self._end

        return self._setting
