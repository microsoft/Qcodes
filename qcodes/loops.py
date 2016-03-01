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
from qcodes.data.data_set import new_data, DataMode
from qcodes.data.data_array import DataArray
from qcodes.utils.helpers import wait_secs
from qcodes.utils.multiprocessing import QcodesProcess
from qcodes.utils.sync_async import mock_sync


MP_NAME = 'Measurement'


def get_bg(return_first=False):
    '''
    find the active background measurement process, if any
    returns None otherwise

    return_first: if there are multiple loops running return the first anyway.
        If false, multiple loops is a RuntimeError.
        default False
    '''
    processes = mp.active_children()
    loops = [p for p in processes if getattr(p, 'name', '') == MP_NAME]

    if len(loops) > 1 and not return_first:
        raise RuntimeError('Oops, multiple loops are running???')

    if loops:
        return loops[0]

    return None


def halt_bg(timeout=5):
    '''
    Stop the active background measurement process, if any
    '''
    loop = get_bg(return_first=True)
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
    '''
    The entry point for creating measurement loops

    sweep_values - a SweepValues or compatible object describing what
        parameter to set in the loop and over what values
    delay - a number of seconds to wait after setting a value before
        continuing.

    After creating a Loop, you attach `action`s to it, making an `ActiveLoop`
    that you can `.run()`, or you can `.run()` a `Loop` directly, in which
    case it takes the default `action`s from the default `Station`

    `actions` are a sequence of things to do at each `Loop` step: they can be
    `Parameter`s to measure, `Task`s to do (any callable that does not yield
    data), `Wait` times, or other `ActiveLoop`s or `Loop`s to nest inside
    this one.
    '''
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
    '''
    Created by attaching actions to a `Loop`, this is the object that actually
    runs a measurement loop. An `ActiveLoop` can no longer be nested, only run,
    or used as an action inside another `Loop` which will run the whole thing.

    The `ActiveLoop` determines what `DataArray`s it will need to hold the data
    it collects, and it creates a `DataSet` holding these `DataArray`s
    '''
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
        '''
        Finds the data arrays that will be created by the actions in this
        loop, and nests them inside this level of the loop.

        Recursively calls `.containers` on any enclosed actions.
        '''
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

        # first massage all the input parameters to the general multi-name form
        if hasattr(action, 'names'):
            names = action.names
            labels = getattr(action, 'labels', names)
            if len(labels) != len(names):
                raise ValueError('must have equal number of names and labels')
            action_indices = tuple((i,) for i in range(len(names)))
        elif hasattr(action, 'name'):
            names = (action.name,)
            labels = (getattr(action, 'label', action.name),)
            action_indices = ((),)
        else:
            raise ValueError('a gettable parameter must have .name or .names')

        num_arrays = len(names)
        sizes = getattr(action, 'sizes', None)
        sp_vals = getattr(action, 'setpoints', None)
        sp_names = getattr(action, 'setpoint_names', None)
        sp_labels = getattr(action, 'setpoint_labels', None)

        if sizes is None:
            sizes = (getattr(action, 'size', ()),) * num_arrays
            sp_vals = (sp_vals,) * num_arrays
            sp_names = (sp_names,) * num_arrays
            sp_labels = (sp_labels,) * num_arrays
        else:
            sp_blank = (None,) * num_arrays
            # _fill_blank both supplies defaults and tests length
            # if values are supplied (for sizes it ONLY tests length)
            sizes = self._fill_blank(sizes, sp_blank)
            sp_vals = self._fill_blank(sp_vals, sp_blank)
            sp_names = self._fill_blank(sp_names, sp_blank)
            sp_labels = self._fill_blank(sp_labels, sp_blank)

        # now loop through these all, to make the DataArrays
        # record which setpoint arrays we've made, so we don't duplicate
        all_setpoints = {}
        for name, label, size, i, sp_vi, sp_ni, sp_li in zip(
                names, labels, sizes, action_indices,
                sp_vals, sp_names, sp_labels):

            # convert the integer form of each size etc. to the tuple form
            if isinstance(size, int):
                size = (size,)
                sp_vi = (sp_vi,)
                sp_ni = (sp_ni,)
                sp_li = (sp_li,)
            elif size is None or size == ():
                size, sp_vi, sp_ni, sp_li = (), (), (), ()
            else:
                sp_blank = (None,) * len(size)
                sp_vi = self._fill_blank(sp_vi, sp_blank)
                sp_ni = self._fill_blank(sp_ni, sp_blank)
                sp_li = self._fill_blank(sp_li, sp_blank)

            setpoints = ()
            # loop through dimensions of size to make the setpoint arrays
            for j, (vij, nij, lij) in enumerate(zip(sp_vi, sp_ni, sp_li)):
                sp_def = (size[: 1 + j], j, setpoints, vij, nij, lij)
                if sp_def not in all_setpoints:
                    all_setpoints[sp_def] = self._make_setpoint_array(*sp_def)
                    out.append(all_setpoints[sp_def])
                setpoints = setpoints + (all_setpoints[sp_def],)

            # finally, make the output data array with these setpoints
            out.append(DataArray(name=name, label=label, size=size,
                       action_indices=i, set_arrays=setpoints))

        return out

    def _fill_blank(self, inputs, blanks):
        if inputs is None:
            return blanks
        elif len(inputs) == len(blanks):
            return inputs
        else:
            raise ValueError('Wrong number of inputs supplied')

    def _make_setpoint_array(self, size, i, prev_setpoints, vals, name, label):
        if vals is None:
            vals = self._default_setpoints(size)
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
            vals = np.array(vals.ndarray)
        else:
            # turn any sequence into a (new) numpy array
            vals = np.array(vals)

        if vals.shape != size:
            raise ValueError('nth setpoint array should have size matching '
                             'the first n dimensions of size.')

        if name is None:
            name = 'index{}'.format(i)

        return DataArray(name=name, label=label, set_arrays=prev_setpoints,
                         size=size, preset_data=vals)

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

    def run(self, background=True, use_async=False, enqueue=False, quiet=False,
            data_manager=None, **kwargs):
        '''
        execute this loop

        background: (default True) run this sweep in a separate process
            so we can have live plotting and other analysis in the main process
        use_async: (default True): execute the sweep asynchronously as much
            as possible
        enqueue: (default False): wait for a previous background sweep to
            finish? If false, will raise an error if another sweep is running
        data_manager: a DataManager instance (omit to use default,
            False to store locally and not write to disk)

        kwargs are passed along to data_set.new_data. The key ones are:
        location: the location of the DataSet, a string whose meaning
            depends on formatter and io, or False to only keep in memory.
            May be a callable to provide automatic locations. If omitted, will
            use the default DataSet.location_provider
        name: if location is default or another provider function, name is
            a string to add to location to make it more readable/meaningful
            to users
        formatter: knows how to read and write the file format
            default can be set in DataSet.default_formatter
        io: knows how to connect to the storage (disk vs cloud etc)


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
        if data_manager is False:
            data_mode = DataMode.LOCAL
        else:
            data_mode = DataMode.PUSH_TO_SERVER

        data_set = new_data(arrays=self.containers(), mode=data_mode,
                            data_manager=data_manager, **kwargs)
        signal_queue = mp.Queue()
        self.set_common_attrs(data_set=data_set, signal_queue=signal_queue)

        if use_async:
            raise NotImplementedError  # TODO
            loop_fn = mock_sync(self._async_loop)
        else:
            loop_fn = self._run_wrapper

        if background:
            # TODO: in notebooks, errors in a background sweep will just appear
            # the next time a command is run. Do something better?
            # (like log them somewhere, show in monitoring window)?
            p = QcodesProcess(target=loop_fn, name=MP_NAME)
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

    def _run_wrapper(self, *args, **kwargs):
        try:
            self._run_loop(*args, **kwargs)
        finally:
            if(hasattr(self, 'data_set') and hasattr(self.data_set, 'close')):
                self.data_set.close()

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
                  current_values=new_values)

                # after the first action, no delay is inherited
                delay = 0

            # after the first setpoint, delay reverts to the loop delay
            delay = self.delay

    def _wait(self, delay):
        finish_clock = time.perf_counter() + delay

        if self._monitor:
            self._monitor.call(finish_by=finish_clock)

        self._check_signal()
        time.sleep(wait_secs(finish_clock))


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
