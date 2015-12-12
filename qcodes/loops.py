'''
Data acquisition loops

The general scheme is:

1. create a (potentially nested) Loop, which defines the sweep
    setpoints and delays
2. activate the loop (which changes it to an ActiveLoop object),
    or omit this step to use the default measurement as given by the
    Loop.set_measurement class method.
3. run it with the .run method, which creates a DataSet to hold the data,
    and defines how and where to save the data.

Some examples:

    # set default measurements for later Loop's to use
    Loop.set_measurement(param1, param2, param3)

    # 1D sweep, using the default measurement set
    Loop(sweep_values, delay).run()

    # 2D sweep, using the default measurement set
    # sv1 is the outer loop, sv2 is the inner.
    Loop(sv1, delay1).loop(sv2, delay2).run()

    # 1D sweep with specific measurements to take at each point
    Loop(sv, delay).each(param4, param5).run()

    # Multidimensional sweep: 1D measurement of param6 on the outer loop,
    # and the default measurements in a 2D loop
    Loop(sv1, delay).each(param6, Loop(sv2, delay)).run()

Supported actions (args to .set_measurement or .each) are:
    Parameter: anything with a .get method and .name or .names
        see parameter.py for options
    ActiveLoop (or Loop, will be activated with default measurement)
    Task: any callable that does not generate data
    Wait: a delay
'''
from datetime import datetime, timedelta
import multiprocessing as mp
import time
import numpy as np

from qcodes.station import Station
from qcodes.data.data_set import DataSet, DataMode
from qcodes.data.data_array import DataArray
from qcodes.utils.helpers import wait_secs, PrintableProcess
from qcodes.utils.sync_async import mock_sync


def get_bg():
    processes = mp.active_children()
    loops = [p for p in processes if isinstance(p, MeasurementProcess)]

    if len(loops) == 1:
        return loops[0]

    if len(loops):
        raise RuntimeError('Oops, multiple loops are running???')

    return None


def halt_bg(self, timeout=5):
    loop = get_bg()
    if not loop:
        print('No loop running')
        return

    loop.signal_queue.put(ActiveLoop.HALT)
    loop.join(timeout)

    if loop.is_alive():
        loop.terminate()
        print('Background loop did not respond to halt signal, terminated')


# def measure(*actions):
#     # measure has been moved into Station
#     # TODO - for all-at-once parameters we want to be able to
#     # store the output into a DataSet without making a Loop.
#     pass


class Loop(object):
    def __init__(self, sweep_values, delay):
        self.sweep_values = sweep_values
        self.delay = delay
        self.nested_loop = None

    def loop(self, sweep_values, delay):
        '''
        Nest another loop inside this one

        Loop(sv1, d1).loop(sv2, d2).each(*a) is equivalent to:
        Loop(sv1, d1).each(Loop(sv2, d2).each(*a))

        returns a new Loop object - the original is untouched
        '''
        out = Loop(self.sweep_values, self.delay)

        if self.nested_loop:
            # nest this new loop inside the deepest level
            out.nested_loop = self.nested_loop.loop(sweep_values, delay)
        else:
            out.nested_loop = Loop(sweep_values, delay)

        return out

    def each(self, *actions):
        '''
        Perform a set of actions at each setting of this loop

        Each action can be:
        - a Parameter to measure
        - a Task to execute
        - a Wait
        - another Loop or ActiveLoop
        '''
        actions = list(actions)

        # check for nested Loops, and activate them with default measurement
        for i, action in enumerate(actions):
            if isinstance(action, Loop):
                default = Station.default.default_measurement
                actions[i] = action.each(*default)

        if self.nested_loop:
            # recurse into the innermost loop and apply these actions there
            actions = [self.nested_loop.each(*actions)]

        return ActiveLoop(self.sweep_values, self.delay, *actions)

    def run(self, *args, **kwargs):
        '''
        shortcut to run a loop with the default measurement set
        stored by Station.set_measurement
        '''
        default = Station.default.default_measurement
        return self.each(*default).run(*args, **kwargs)


class ActiveLoop(object):
    HALT = 'HALT LOOP'

    def __init__(self, sweep_values, delay, *actions):
        self.sweep_values = sweep_values
        self.delay = delay
        self.actions = actions

        # compile now, but don't save the results
        # just used for preemptive error checking
        # if we saved the results, we wouldn't capture nesting
        # nor would we be able to reuse an ActiveLoop multiple times
        # within one outer Loop.
        # TODO: this doesn't work, because _Measure needs the data_set,
        # which doesn't exist yet - do we want to make a special "dry run"
        # mode, or is it sufficient to let errors wait until .run()?
        # self._compile_actions(actions)

        # if the first action is another loop, it changes how delays
        # happen - the outer delay happens *after* the inner var gets
        # set to its initial value
        self._nest_first = hasattr(actions[0], 'containers')

        self._monitor = None  # TODO: how to specify this?

    def containers(self):
        loop_size = len(self.sweep_values)
        loop_array = DataArray(parameter=self.sweep_values.parameter)
        loop_array.nest(size=loop_size)

        data_arrays = [loop_array]

        for i, action in enumerate(self.actions):
            if hasattr(action, 'containers'):
                action_arrays = action.containers()

            elif hasattr(action, 'get'):
                # this action is a parameter to measure
                # note that this supports lists (separate output arrays)
                # and arrays (nested in one/each output array) of return values
                action_arrays = self._parameter_arrays(action)

            else:
                continue

            for array in action_arrays:
                array.nest(size=loop_size, action_index=i,
                           set_array=loop_array)
            data_arrays.extend(action_arrays)

        return data_arrays

    def _parameter_arrays(self, action):
        out = []

        size = getattr(action, 'size', ())
        if isinstance(size, int):
            size = (size,)
            sp_blank = None
        else:
            size = tuple(size)
            sp_blank = (None,) * len(size)

        ndim = len(size)
        if ndim:
            for dim in self.size:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError('size must consist of positive '
                                     'integers, not ' + repr(dim))
            sp_vals = getattr(action, 'setpoints', sp_blank)
            sp_names = getattr(action, 'sp_names', sp_blank)
            sp_labels = getattr(action, 'sp_labels', sp_blank)

            if sp_blank is None:
                sp_vals = (sp_vals,)
                sp_names = (sp_names,)
                sp_labels = (sp_labels,)

            if (len(sp_vals) != ndim or len(sp_names) != ndim or
                    len(sp_labels) != ndim):
                raise ValueError('Wrong number of setpoint, setpoint names, '
                                 'or setpoint labels provided')

            for i, sp in enumerate(zip(sp_vals, sp_names, sp_labels)):
                out.append(self._make_setpoint_array(size, i, tuple(out), *sp))
        else:
            setpoints = ()

        if hasattr(action, 'names'):
            names = action.names
            labels = getattr(action, 'labels', names)
            for i, (name, label) in enumerate(zip(names, labels)):
                out.append(DataArray(name=name, label=label, size=size,
                           action_indices=(i,), set_arrays=setpoints))

        elif hasattr(action, 'name'):
            name = action.name
            label = getattr(action, 'label', name)
            out.append(DataArray(name=name, label=label, size=size,
                                 action_indices=(), set_arrays=setpoints))

        else:
            raise ValueError('a gettable parameter must have .name or .names')

        return out

    def _make_setpoint_array(self, size, i, prev_setpoints, vals, name, label):
        this_size = size[: i + 1]

        if vals is None:
            vals = self._default_setpoints(this_size)
        elif isinstance(vals, DataArray):
            # can't simply use the DataArray, even though that's
            # what we're going to return here, because it will
            # get nested (don't want to alter the original)
            # DataArrays do have the advantage though of already including
            # name and label, so take these if they exist
            if vals.name is not None:
                name = vals.name
            if vals.label is not None:
                label = vals.label

            # extract a copy of the numpy array
            vals = np.array(vals.data)
        else:
            # turn any sequence into a (new) numpy array
            vals = np.array(vals)

        if vals.shape != this_size:
            raise ValueError('nth setpoint array should have size matching '
                             'the first n dimensions of size.')

        if name is None:
            if len(size) > 1:
                name = 'index{}'.format(i + 1)
            else:
                name = 'index'

        return DataArray(name=name, label=label, set_arrays=prev_setpoints,
                         size=this_size, preset_data=vals)

    def _default_setpoints(self, size):
        if len(size) == 1:
            return np.arange(0, size[0], 1)

        sp = np.ndarray(size)
        sp_inner = self._default_setpoints(self, size[1:])
        for i in range(len(sp)):
            sp[i] = sp_inner

        return sp

    def set_common_attrs(self, data_set, signal_queue):
        '''
        set a couple of common attributes that the main and nested loops
        all need to have:
        - the DataSet collecting all our measurements
        - a queue for communicating with the main process
        '''
        self.data_set = data_set
        self.signal_queue = signal_queue
        for action in self.actions:
            if hasattr(action, 'set_common_attrs'):
                action.set_common_attrs(data_set, signal_queue)

    def _check_signal(self):
        while not self.signal_queue.empty():
            signal = self.signal_queue.get()
            if signal == self.HALT:
                raise KeyboardInterrupt('sweep was halted')

    def run(self, location=None, formatter=None, io=None, data_manager=None,
            background=True, use_async=False, enqueue=False, quiet=False):
        '''
        execute this loop

        location: the location of the DataSet, a string whose meaning
            depends on formatter and io
        formatter: knows how to read and write the file format
        io: knows how to connect to the storage (disk vs cloud etc)
        data_manager: a DataManager instance (omit to use default)
        background: (default True) run this sweep in a separate process
            so we can have live plotting and other analysis in the main process
        use_async: (default True): execute the sweep asynchronously as much
            as possible
        enqueue: (default False): wait for a previous background sweep to
            finish? If false, will raise an error if another sweep is running

        returns:
            a DataSet object that we can use to plot
        '''

        prev_loop = get_bg()
        if prev_loop:
            if enqueue:
                prev_loop.join()  # wait until previous loop finishes
            else:
                raise RuntimeError(
                    'a loop is already running in the background')

        data_set = DataSet(arrays=self.containers(),
                           mode=DataMode.PUSH_TO_SERVER,
                           data_manager=data_manager, location=location,
                           formatter=formatter, io=io)
        signal_queue = mp.Queue()
        self.set_common_attrs(data_set=data_set, signal_queue=signal_queue)

        if use_async:
            raise NotImplementedError  # TODO
        loop_fn = mock_sync(self._async_loop) if use_async else self._run_loop

        if background:
            # TODO: in notebooks, errors in a background sweep will just appear
            # the next time a command is run. Do something better?
            # (like log them somewhere, show in monitoring window)?
            p = MeasurementProcess(target=loop_fn, daemon=True)
            p.is_sweep = True
            p.signal_queue = self.signal_queue
            p.start()
            self.data_set.sync()
            self.data_set.mode = DataMode.PULL_FROM_SERVER
        else:
            loop_fn()
            self.data_set.read()

        if not quiet:
            print(repr(self.data_set))
            print(datetime.now().strftime('started at %Y-%m-%d %H:%M:%S'))
        return self.data_set

    def _compile_actions(self, actions, action_indices=()):
        callables = []
        measurement_group = []
        for i, action in enumerate(actions):
            new_action_indices = action_indices + (i,)
            if hasattr(action, 'get'):
                measurement_group.append((action, new_action_indices))
                continue
            elif measurement_group:
                callables.append(_Measure(measurement_group, self.data_set))
                measurement_group[:] = []

            callables.append(self._compile_one(action, new_action_indices))

        if measurement_group:
            callables.append(_Measure(measurement_group, self.data_set))
            measurement_group[:] = []

        return callables

    def _compile_one(self, action, new_action_indices):
        if isinstance(action, Wait):
            return Task(self._wait, action.delay)
        elif isinstance(action, ActiveLoop):
            return _Nest(action, new_action_indices)
        elif callable(action):
            return action
        else:
            raise TypeError('unrecognized action', action)

    def _run_loop(self, first_delay=0, action_indices=(),
                  loop_indices=(), current_values=(),
                  **ignore_kwargs):
        '''
        the routine that actually executes the loop, and can be called
        from one loop to execute a nested loop

        first_delay: any delay carried over from an outer loop
        action_indices: where we are in any outer loop action arrays
        loop_indices: setpoint indices in any outer loops
        current_values: setpoint values in any outer loops
        signal_queue: queue to communicate with main process directly
        ignore_kwargs: for compatibility with other loop tasks
        '''

        # at the beginning of the loop, the time to wait after setting
        # the loop parameter may be increased if an outer loop requested longer
        delay = max(self.delay, first_delay)

        callables = self._compile_actions(self.actions, action_indices)

        for i, value in enumerate(self.sweep_values):
            self.sweep_values.set(value)
            new_indices = loop_indices + (i,)
            new_values = current_values + (value,)
            set_name = self.data_set.action_id_map[action_indices]
            self.data_set.store(new_indices, {set_name: value})

            if not self._nest_first:
                # only wait the delay time if an inner loop will not inherit it
                self._wait(delay)

            for f in callables:
                f(first_delay=delay,
                  loop_indices=new_indices,
                  values=new_values)

                # after the first action, no delay is inherited
                delay = 0

            # after the first setpoint, delay reverts to the loop delay
            delay = self.delay

        if not current_values:
            self.data_set.close()

    def _wait(self, delay):
        finish_datetime = datetime.now() + timedelta(seconds=delay)

        if self._monitor:
            self._monitor.call(finish_by=finish_datetime)

        self._check_signal()
        time.sleep(wait_secs(finish_datetime))


class MeasurementProcess(PrintableProcess):
    name = 'MeasurementLoop'


class Task(object):
    '''
    A predefined task to be executed within a measurement Loop
    This form is for a simple task that does not measure any data,
    and does not depend on the state of the loop when it is called.

    The first argument should be a callable, to which any subsequent
    args and kwargs (which are evaluated before the loop starts) are passed.

    kwargs passed when the Task is called are ignored,
    but are accepted for compatibility with other things happening in a Loop.
    '''
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, **ignore_kwargs):
        self.func(*self.args, **self.kwargs)


class Wait(object):
    '''
    A simple class to tell a Loop to wait <delay> seconds

    This is transformed into a Task within the Loop, such that
    it can do other things (monitor, check for halt) during the delay.

    But for use outside of a Loop, it is also callable (then it just sleeps)
    '''
    def __init__(self, delay):
        self.delay = delay

    def __call__(self):
        time.sleep(self.delay)


class _Measure(object):
    '''
    A callable collection of parameters to measure.
    This should not be constructed manually, only by an ActiveLoop.
    '''
    def __init__(self, params_indices, data_set):
        # the applicable DataSet.store function
        self.store = data_set.store

        # for performance, pre-calculate which params return data for
        # multiple arrays, pre-create the dict to pass these to store fn
        # and pre-calculate the name mappings
        self.dict = {}
        self.params_ids = []
        for param, action_indices in params_indices:
            if hasattr(param, 'names'):
                part_ids = []
                for i in range(len(param.names)):
                    param_id = data_set.action_id_map[action_indices + (i,)]
                    part_ids.append(param_id)
                    self.dict[param_id] = None
                self.params_ids.append((param, None, part_ids))
            else:
                param_id = data_set.action_id_map[action_indices]
                self.params_ids.append((param, param_id, False))
                self.dict[param_id] = None

    def __call__(self, loop_indices, **ignore_kwargs):
        for param, param_id, composite in self.params_ids:
            if composite:
                for val, part_id in zip(param.get(), composite):
                    self.dict[part_id] = val
            else:
                self.dict[param_id] = param.get()

        self.store(loop_indices, self.dict)


class _Nest(object):
    '''
    wrapper to make a callable nested ActiveLoop
    This should not be constructed manually, only by an ActiveLoop.
    '''
    def __init__(self, inner_loop, action_indices):
        self.inner_loop = inner_loop
        self.action_indices = action_indices

    def __call__(self, **kwargs):
        self.inner_loop._run_loop(action_indices=self.action_indices, **kwargs)
