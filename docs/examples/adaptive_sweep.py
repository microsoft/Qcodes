import qcodes as qc


class AdaptiveSweep(qc.SweepValues):
    '''
    an example class to show how adaptive sampling might be implemented
    this code has not been tested

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
