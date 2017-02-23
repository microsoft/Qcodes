"""
Data acquisition loops.

The general scheme is:

1. create a (potentially nested) Loop, which defines the sweep setpoints and
delays

2. activate the loop (which changes it to an ActiveLoop object),
or omit this step to use the default measurement as given by the
Loop.set_measurement class method.

3. run it with the .run method, which creates a DataSet to hold the data,
and defines how and where to save the data.

Some examples:

- set default measurements for later Loop's to use

>>> Loop.set_measurement(param1, param2, param3)

- 1D sweep, using the default measurement set

>>> Loop(sweep_values, delay).run()

- 2D sweep, using the default measurement set sv1 is the outer loop, sv2 is the
  inner.

>>> Loop(sv1, delay1).loop(sv2, delay2).run()

- 1D sweep with specific measurements to take at each point

>>> Loop(sv, delay).each(param4, param5).run()

- Multidimensional sweep: 1D measurement of param6 on the outer loop, and the
  default measurements in a 2D loop

>>> Loop(sv1, delay).each(param6, Loop(sv2, delay)).run()

Supported commands to .set_measurement or .each are:

    - Parameter: anything with a .get method and .name or .names see
      parameter.py for options
    - ActiveLoop (or Loop, will be activated with default measurement)
    - Task: any callable that does not generate data
    - Wait: a delay
"""

from datetime import datetime
import logging
import multiprocessing as mp
import time
import numpy as np
import warnings

from qcodes import config
from qcodes.station import Station
from qcodes.data.data_set import new_data, DataMode
from qcodes.data.data_array import DataArray
from qcodes.data.manager import get_data_manager
from qcodes.utils.helpers import wait_secs, full_class, tprint
from qcodes.process.qcodes_process import QcodesProcess
from qcodes.utils.metadata import Metadatable

from .actions import (_actions_snapshot, Task, Wait, _Measure, _Nest,
                      BreakIf, _QcodesBreak)


log = logging.getLogger(__name__)
# Switches off multiprocessing by default, cant' be altered after module
USE_MP = config.core.legacy_mp
MP_NAME = 'Measurement'


def get_bg(return_first=False):
    """
    Find the active background measurement process, if any
    returns None otherwise.

    Todo:
        RuntimeError message is really hard to understand.
    Args:
        return_first(bool): if there are multiple loops running return the
                            first anyway.
    Raises:
        RuntimeError: if multiple loops are active and return_first is False.
    Returns:
        Union[loop, None]: active loop or none if no loops are active
    """
    processes = mp.active_children()
    loops = [p for p in processes if getattr(p, 'name', '') == MP_NAME]

    if len(loops) > 1 and not return_first:
        raise RuntimeError('Oops, multiple loops are running???')

    if loops:
        return loops[0]

    # if we got here, there shouldn't be a loop running. Make sure the
    # data manager, if there is one, agrees!
    _clear_data_manager()
    return None


def halt_bg(timeout=5, traceback=True):
    """
    Stop the active background measurement process, if any.

    Args:
        timeout (int): seconds to wait for a clean exit before forcibly
         terminating.

        traceback (bool):  whether to print a traceback at the point of
         interrupt, for debugging purposes.
    """
    loop = get_bg(return_first=True)
    if not loop:
        print('No loop running')
        return

    if traceback:
        signal_ = ActiveLoop.HALT_DEBUG
    else:
        signal_ = ActiveLoop.HALT

    loop.signal_queue.put(signal_)
    loop.join(timeout)

    if loop.is_alive():
        loop.terminate()
        loop.join(timeout/2)
        print('Background loop did not respond to halt signal, terminated')

    _clear_data_manager()


def _clear_data_manager():
    dm = get_data_manager(only_existing=True)
    if dm and dm.ask('get_measuring'):
        dm.ask('finalize_data')

# TODO(giulioungaretti) remove dead code
# def measure(*actions):
#     # measure has been moved into Station
#     # TODO - for all-at-once parameters we want to be able to
#     # store the output into a DataSet without making a Loop.
#     pass


class Loop(Metadatable):
    """
    The entry point for creating measurement loops

    Args:
        sweep_values: a SweepValues or compatible object describing what
            parameter to set in the loop and over what values
        delay: a number of seconds to wait after setting a value before
            continuing. 0 (default) means no waiting and no warnings. > 0
            means to wait, potentially filling the delay time with monitoring,
            and give an error if you wait longer than expected.
        progress_interval: should progress of the loop every x seconds. Default
            is None (no output)

    After creating a Loop, you attach ``action``\s to it, making an ``ActiveLoop``

    TODO:
        how? Maybe obvious but not specified! that you can ``.run()``,
        or you can ``.run()`` a ``Loop`` directly, in which
        case it takes the default ``action``\s from the default ``Station``

    ``actions`` are a sequence of things to do at each ``Loop`` step: they can be
    ``Parameter``\s to measure, ``Task``\s to do (any callable that does not yield
    data), ``Wait`` times, or other ``ActiveLoop``\s or ``Loop``\s to nest inside
    this one.
    """
    def __init__(self, sweep_values, delay=0, station=None,
                 progress_interval=None):
        super().__init__()
        if delay < 0:
            raise ValueError('delay must be > 0, not {}'.format(repr(delay)))

        self.sweep_values = sweep_values
        self.delay = delay
        self.station = station
        self.nested_loop = None
        self.actions = None
        self.then_actions = ()
        self.bg_task = None
        self.bg_final_task = None
        self.bg_min_delay = None
        self.progress_interval = progress_interval

    def loop(self, sweep_values, delay=0):
        """
        Nest another loop inside this one.

        Args:
            sweep_values ():
            delay (int):

        Examples:
            >>> Loop(sv1, d1).loop(sv2, d2).each(*a)

            is equivalent to:

            >>> Loop(sv1, d1).each(Loop(sv2, d2).each(*a))

        Returns: a new Loop object - the original is untouched
        """
        out = self._copy()

        if out.nested_loop:
            # nest this new loop inside the deepest level
            out.nested_loop = out.nested_loop.loop(sweep_values, delay)
        else:
            out.nested_loop = Loop(sweep_values, delay)

        return out

    def _copy(self):
        out = Loop(self.sweep_values, self.delay,
                   progress_interval=self.progress_interval)
        out.nested_loop = self.nested_loop
        out.then_actions = self.then_actions
        out.station = self.station
        return out

    def each(self, *actions):
        """
        Perform a set of actions at each setting of this loop.
        TODO(setting vs setpoints) ? better be verbose.

        Args:
            *actions (Any): actions to perform at each setting of the loop

        Each action can be:

        - a Parameter to measure
        - a Task to execute
        - a Wait
        - another Loop or ActiveLoop

        """
        actions = list(actions)

        # check for nested Loops, and activate them with default measurement
        for i, action in enumerate(actions):
            if isinstance(action, Loop):
                default = Station.default.default_measurement
                actions[i] = action.each(*default)

        self.validate_actions(*actions)

        if self.nested_loop:
            # recurse into the innermost loop and apply these actions there
            actions = [self.nested_loop.each(*actions)]

        return ActiveLoop(self.sweep_values, self.delay, *actions,
                          then_actions=self.then_actions, station=self.station,
                          progress_interval=self.progress_interval,
                          bg_task=self.bg_task, bg_final_task=self.bg_final_task, bg_min_delay=self.bg_min_delay)

    def with_bg_task(self, task, bg_final_task=None, min_delay=0.01):
        """
        Attaches a background task to this loop.

        Args:
            task: A callable object with no parameters. This object will be
                invoked periodically during the measurement loop.

            bg_final_task: A callable object with no parameters. This object will be
                invoked to clean up after or otherwise finish the background
                task work.

            min_delay (default 0.01): The minimum number of seconds to wait
                between task invocations.
                Note that if a task is doing a lot of processing it is recommended
                to increase min_delay.
                Note that the actual time between task invocations may be much
                longer than this, as the task is only run between passes
                through the loop.
        """
        return _attach_bg_task(self, task, bg_final_task, min_delay)

    @staticmethod
    def validate_actions(*actions):
        """
        Whitelist acceptable actions, so we can give nice error messages
        if an action is not recognized
        """
        for action in actions:
            if isinstance(action, (Task, Wait, BreakIf, ActiveLoop)):
                continue
            if hasattr(action, 'get') and (hasattr(action, 'name') or
                                           hasattr(action, 'names')):
                continue
            raise TypeError('Unrecognized action:', action,
                            'Allowed actions are: objects (parameters) with '
                            'a `get` method and `name` or `names` attribute, '
                            'and `Task`, `Wait`, `BreakIf`, and `ActiveLoop` '
                            'objects. `Loop` objects are OK too, except in '
                            'Station default measurements.')

    def run(self, *args, **kwargs):
        """
        shortcut to run a loop with the default measurement set
        stored by Station.set_measurement
        """
        default = Station.default.default_measurement
        return self.each(*default).run(*args, **kwargs)

    def run_temp(self, *args, **kwargs):
        """
        shortcut to run a loop in the foreground as a temporary dataset
        using the default measurement set
        """
        return self.run(*args, background=False, quiet=True,
                        data_manager=False, location=False, **kwargs)

    def then(self, *actions, overwrite=False):
        """
        Attach actions to be performed after the loop completes.

        These can only be *Task* and *Wait* actions, as they may not generate
        any data.

        returns a new Loop object - the original is untouched

        This is more naturally done to an ActiveLoop (ie after .each())
        and can also be done there, but it's allowed at this stage too so that
        you can define final actions and share them among several *Loop*\s that
        have different loop actions, or attach final actions to a Loop run

        TODO:
            examples of this ? with default actions.

        Args:
            \*actions: *Task* and *Wait* objects to execute in order

            overwrite: (default False) whether subsequent .then() calls (including
                calls in an ActiveLoop after .then() has already been called on
                the Loop) will add to each other or overwrite the earlier ones.
        Returns:
            a new Loop object - the original is untouched
        """
        return _attach_then_actions(self._copy(), actions, overwrite)

    def snapshot_base(self, update=False):
        """
        State of the loop as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the underlying
             sweep_values and actions. If False, just use the latest values in
             memory.

        Returns:
            dict: base snapshot
        """
        return {
            '__class__': full_class(self),
            'sweep_values': self.sweep_values.snapshot(update=update),
            'delay': self.delay,
            'then_actions': _actions_snapshot(self.then_actions, update)
        }


def _attach_then_actions(loop, actions, overwrite):
    """Inner code for both Loop.then and ActiveLoop.then."""
    for action in actions:
        if not isinstance(action, (Task, Wait)):
            raise TypeError('Unrecognized action:', action,
                            '.then() allows only `Task` and `Wait` '
                            'actions.')

    if overwrite:
        loop.then_actions = actions
    else:
        loop.then_actions = loop.then_actions + actions

    return loop


def _attach_bg_task(loop, task, bg_final_task, min_delay):
    """Inner code for both Loop and ActiveLoop.bg_task"""
    if loop.bg_task is None:
        loop.bg_task = task
        loop.bg_min_delay = min_delay
    else:
        raise RuntimeError('Only one background task is allowed per loop')

    if bg_final_task:
        loop.bg_final_task = bg_final_task

    return loop


class ActiveLoop(Metadatable):
    """
    Created by attaching actions to a *Loop*, this is the object that actually
    runs a measurement loop. An *ActiveLoop* can no longer be nested, only run,
    or used as an action inside another `Loop` which will run the whole thing.

    The *ActiveLoop* determines what *DataArray*\s it will need to hold the data
    it collects, and it creates a *DataSet* holding these *DataArray*\s
    """
    # constants for signal_queue
    HALT = 'HALT LOOP'
    HALT_DEBUG = 'HALT AND DEBUG'

    # maximum sleep time (secs) between checking the signal_queue for a HALT
    signal_period = 1

    def __init__(self, sweep_values, delay, *actions, then_actions=(),
                 station=None, progress_interval=None, bg_task=None,
                 bg_final_task=None, bg_min_delay=None):
        super().__init__()
        self.sweep_values = sweep_values
        self.delay = delay
        self.actions = list(actions)
        self.progress_interval = progress_interval
        self.then_actions = then_actions
        self.station = station
        self.bg_task = bg_task
        self.bg_final_task = bg_final_task
        self.bg_min_delay = bg_min_delay
        self.data_set = None

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

        # for sending halt signals to the loop
        self.signal_queue = mp.Queue()

        self._monitor = None  # TODO: how to specify this?

    def then(self, *actions, overwrite=False):
        """
        Attach actions to be performed after the loop completes.

        These can only be `Task` and `Wait` actions, as they may not generate
        any data.

        returns a new ActiveLoop object - the original is untouched

        \*actions: `Task` and `Wait` objects to execute in order

        Args:
            overwrite: (default False) whether subsequent .then() calls (including
                calls in an ActiveLoop after .then() has already been called on
                the Loop) will add to each other or overwrite the earlier ones.
        """
        loop = ActiveLoop(self.sweep_values, self.delay, *self.actions,
                          then_actions=self.then_actions, station=self.station)
        return _attach_then_actions(loop, actions, overwrite)

    def with_bg_task(self, task, bg_final_task=None, min_delay=0.01):
        """
        Attaches a background task to this loop.

        Args:
            task: A callable object with no parameters. This object will be
                invoked periodically during the measurement loop.

            bg_final_task: A callable object with no parameters. This object will be
                invoked to clean up after or otherwise finish the background
                task work.

            min_delay (default 1): The minimum number of seconds to wait
                between task invocations. Note that the actual time between
                task invocations may be much longer than this, as the task is
                only run between passes through the loop.
        """
        return _attach_bg_task(self, task, bg_final_task, min_delay)

    def snapshot_base(self, update=False):
        """Snapshot of this ActiveLoop's definition."""
        return {
            '__class__': full_class(self),
            'sweep_values': self.sweep_values.snapshot(update=update),
            'delay': self.delay,
            'actions': _actions_snapshot(self.actions, update),
            'then_actions': _actions_snapshot(self.then_actions, update)
        }

    def containers(self):
        """
        Finds the data arrays that will be created by the actions in this
        loop, and nests them inside this level of the loop.

        Recursively calls `.containers` on any enclosed actions.
        """
        loop_size = len(self.sweep_values)
        data_arrays = []
        loop_array = DataArray(parameter=self.sweep_values.parameter,
                               is_setpoint=True)
        loop_array.nest(size=loop_size)

        data_arrays = [loop_array]
        # hack set_data into actions
        new_actions = self.actions[:]
        if hasattr(self.sweep_values, "parameters"):
            for parameter in self.sweep_values.parameters:
                new_actions.append(parameter)

        for i, action in enumerate(new_actions):
            if hasattr(action, 'containers'):
                action_arrays = action.containers()

            elif hasattr(action, 'get'):
                # this action is a parameter to measure
                # note that this supports lists (separate output arrays)
                # and arrays (nested in one/each output array) of return values
                action_arrays = self._parameter_arrays(action)

            else:
                # this *is* covered but the report misses it because Python
                # optimizes it away. See:
                # https://bitbucket.org/ned/coveragepy/issues/198
                continue  # pragma: no cover

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
            full_names = action.full_names
            labels = getattr(action, 'labels', names)
            if len(labels) != len(names):
                raise ValueError('must have equal number of names and labels')
            action_indices = tuple((i,) for i in range(len(names)))
        elif hasattr(action, 'name'):
            names = (action.name,)
            full_names = (action.full_name,)
            labels = (getattr(action, 'label', action.name),)
            action_indices = ((),)
        else:
            raise ValueError('a gettable parameter must have .name or .names')

        num_arrays = len(names)
        shapes = getattr(action, 'shapes', None)
        sp_vals = getattr(action, 'setpoints', None)
        sp_names = getattr(action, 'setpoint_names', None)
        sp_labels = getattr(action, 'setpoint_labels', None)

        if shapes is None:
            shapes = (getattr(action, 'shape', ()),) * num_arrays
            sp_vals = (sp_vals,) * num_arrays
            sp_names = (sp_names,) * num_arrays
            sp_labels = (sp_labels,) * num_arrays
        else:
            sp_blank = (None,) * num_arrays
            # _fill_blank both supplies defaults and tests length
            # if values are supplied (for shapes it ONLY tests length)
            shapes = self._fill_blank(shapes, sp_blank)
            sp_vals = self._fill_blank(sp_vals, sp_blank)
            sp_names = self._fill_blank(sp_names, sp_blank)
            sp_labels = self._fill_blank(sp_labels, sp_blank)

        # now loop through these all, to make the DataArrays
        # record which setpoint arrays we've made, so we don't duplicate
        all_setpoints = {}
        for name, full_name, label, shape, i, sp_vi, sp_ni, sp_li in zip(
                names, full_names, labels, shapes, action_indices,
                sp_vals, sp_names, sp_labels):

            if shape is None or shape == ():
                shape, sp_vi, sp_ni, sp_li = (), (), (), ()
            else:
                sp_blank = (None,) * len(shape)
                sp_vi = self._fill_blank(sp_vi, sp_blank)
                sp_ni = self._fill_blank(sp_ni, sp_blank)
                sp_li = self._fill_blank(sp_li, sp_blank)

            setpoints = ()
            # loop through dimensions of shape to make the setpoint arrays
            for j, (vij, nij, lij) in enumerate(zip(sp_vi, sp_ni, sp_li)):
                sp_def = (shape[: 1 + j], j, setpoints, vij, nij, lij)
                if sp_def not in all_setpoints:
                    all_setpoints[sp_def] = self._make_setpoint_array(*sp_def)
                    out.append(all_setpoints[sp_def])
                setpoints = setpoints + (all_setpoints[sp_def],)

            # finally, make the output data array with these setpoints
            out.append(DataArray(name=name, full_name=full_name, label=label,
                                 shape=shape, action_indices=i,
                                 set_arrays=setpoints, parameter=action))

        return out

    def _fill_blank(self, inputs, blanks):
        if inputs is None:
            return blanks
        elif len(inputs) == len(blanks):
            return inputs
        else:
            raise ValueError('Wrong number of inputs supplied')

    def _make_setpoint_array(self, shape, i, prev_setpoints, vals, name,
                             label):
        if vals is None:
            vals = self._default_setpoints(shape)
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

        if vals.shape != shape:
            raise ValueError('nth setpoint array should have shape matching '
                             'the first n dimensions of shape.')

        if name is None:
            name = 'index{}'.format(i)

        return DataArray(name=name, label=label, set_arrays=prev_setpoints,
                         shape=shape, preset_data=vals)

    def _default_setpoints(self, shape):
        if len(shape) == 1:
            return np.arange(0, shape[0], 1)

        sp = np.ndarray(shape)
        sp_inner = self._default_setpoints(shape[1:])
        for i in range(len(sp)):
            sp[i] = sp_inner

        return sp

    def set_common_attrs(self, data_set, use_threads, signal_queue):
        """
        set a couple of common attributes that the main and nested loops
        all need to have:
        - the DataSet collecting all our measurements
        - a queue for communicating with the main process
        """
        self.data_set = data_set
        self.signal_queue = signal_queue
        self.use_threads = use_threads
        for action in self.actions:
            if hasattr(action, 'set_common_attrs'):
                action.set_common_attrs(data_set, use_threads, signal_queue)

    def _check_signal(self):
        while not self.signal_queue.empty():
            signal_ = self.signal_queue.get()
            if signal_ == self.HALT:
                raise _QuietInterrupt('sweep was halted')
            elif signal_ == self.HALT_DEBUG:
                raise _DebugInterrupt('sweep was halted')
            else:
                raise ValueError('unknown signal', signal_)

    def get_data_set(self, data_manager=USE_MP, *args, **kwargs):
        """
        Return the data set for this loop.

        If no data set has been created yet, a new one will be created and
        returned. Note that all arguments can only be provided when the
        `DataSet` is first created; giving these during `run` when
        `get_data_set` has already been called on its own is an error.

        Args:
            data_manager: a DataManager instance (omit to use default,
                False to store locally)

        kwargs are passed along to data_set.new_data. The key ones are:

        Args:
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
            write_period: how often to save to storage during the loop.
                default 5 sec, use None to write only at the end

        returns:
            a DataSet object that we can use to plot
        """
        if self.data_set is None:
            if data_manager is False:
                data_mode = DataMode.LOCAL
            else:
                warnings.warn("Multiprocessing is in beta, use at own risk",
                              UserWarning)
                data_mode = DataMode.PUSH_TO_SERVER

            data_set = new_data(arrays=self.containers(), mode=data_mode,
                                data_manager=data_manager, *args, **kwargs)

            self.data_set = data_set

        else:
            has_args = len(kwargs) or len(args)
            uses_data_manager = (self.data_set.mode != DataMode.LOCAL)
            if has_args or (uses_data_manager != data_manager):
                raise RuntimeError(
                    'The DataSet for this loop already exists. '
                    'You can only provide DataSet attributes, such as '
                    'data_manager, location, name, formatter, io, '
                    'write_period, when the DataSet is first created.')

        return self.data_set

    def run_temp(self, **kwargs):
        """
        wrapper to run this loop in the foreground as a temporary data set,
        especially for use in composite parameters that need to run a Loop
        as part of their get method
        """
        return self.run(background=False, quiet=True,
                        data_manager=False, location=False, **kwargs)

    def run(self, background=USE_MP, use_threads=False, quiet=False,
            data_manager=USE_MP, station=None, progress_interval=False,
            *args, **kwargs):
        """
        Execute this loop.

        Args:
            background: (default False) run this sweep in a separate process
                so we can have live plotting and other analysis in the main process
            use_threads: (default False): whenever there are multiple `get` calls
                back-to-back, execute them in separate threads so they run in
                parallel (as long as they don't block each other)
            quiet: (default False): set True to not print anything except errors
            data_manager: set to True to use a DataManager. Default to False.
            station: a Station instance for snapshots (omit to use a previously
                provided Station, or the default Station)
            progress_interval (default None): show progress of the loop every x
                seconds. If provided here, will override any interval provided
                with the Loop definition

        kwargs are passed along to data_set.new_data. These can only be
        provided when the `DataSet` is first created; giving these during `run`
        when `get_data_set` has already been called on its own is an error.
        The key ones are:

        Args:
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
                write_period: how often to save to storage during the loop.
                default 5 sec, use None to write only at the end


        returns:
            a DataSet object that we can use to plot
        """
        if progress_interval is not False:
            self.progress_interval = progress_interval

        prev_loop = get_bg()
        if prev_loop:
            if not quiet:
                print('Waiting for the previous background Loop to finish...',
                      flush=True)
            prev_loop.join()

        data_set = self.get_data_set(data_manager, *args, **kwargs)

        if background and not getattr(data_set, 'data_manager', None):
            warnings.warn(
                'With background=True you must also set data_manager=True '
                'or you will not be able to sync your DataSet.',
                UserWarning)

        self.set_common_attrs(data_set=data_set, use_threads=use_threads,
                              signal_queue=self.signal_queue)

        station = station or self.station or Station.default
        if station:
            data_set.add_metadata({'station': station.snapshot()})

        # information about the loop definition is in its snapshot
        data_set.add_metadata({'loop': self.snapshot()})
        # then add information about how and when it was run
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_set.add_metadata({'loop': {
            'ts_start': ts,
            'background': background,
            'use_threads': use_threads,
            'use_data_manager': (data_manager is not False)
        }})

        data_set.save_metadata()

        if prev_loop and not quiet:
            print('...done. Starting ' + (data_set.location or 'new loop'),
                  flush=True)

        try:
            if background:
                warnings.warn("Multiprocessing is in beta, use at own risk",
                              UserWarning)
                p = QcodesProcess(target=self._run_wrapper, name=MP_NAME)
                p.is_sweep = True
                p.signal_queue = self.signal_queue
                p.start()
                self.process = p

                # now that the data_set we created has been put in the loop
                # process, this copy turns into a reader
                # if you're not using a DataManager, it just stays local
                # and sync() reads from disk
                if self.data_set.mode == DataMode.PUSH_TO_SERVER:
                    self.data_set.mode = DataMode.PULL_FROM_SERVER
                self.data_set.sync()
            else:
                if hasattr(self, 'process'):
                    # in case this ActiveLoop was run before in the background
                    del self.process

                self._run_wrapper()

                if self.data_set.mode != DataMode.LOCAL:
                    self.data_set.sync()

            ds = self.data_set

        finally:
            if not quiet:
                print(repr(self.data_set))
                print(datetime.now().strftime('started at %Y-%m-%d %H:%M:%S'))

            # After normal loop execution we clear the data_set so we can run
            # again. But also if something went wrong during the loop execution
            # we want to clear the data_set attribute so we don't try to reuse
            # this one later.
            self.data_set = None


        return ds

    def _compile_actions(self, actions, action_indices=()):
        callables = []
        measurement_group = []
        for i, action in enumerate(actions):
            new_action_indices = action_indices + (i,)
            if hasattr(action, 'get'):
                measurement_group.append((action, new_action_indices))
                continue
            elif measurement_group:
                callables.append(_Measure(measurement_group, self.data_set,
                                          self.use_threads))
                measurement_group[:] = []

            callables.append(self._compile_one(action, new_action_indices))

        if measurement_group:
            callables.append(_Measure(measurement_group, self.data_set,
                                      self.use_threads))
            measurement_group[:] = []

        return callables

    def _compile_one(self, action, new_action_indices):
        if isinstance(action, Wait):
            return Task(self._wait, action.delay)
        elif isinstance(action, ActiveLoop):
            return _Nest(action, new_action_indices)
        else:
            return action

    def _run_wrapper(self, *args, **kwargs):
        try:
            self._run_loop(*args, **kwargs)
        except _QuietInterrupt:
            pass
        finally:
            if hasattr(self, 'data_set'):
                # somehow this does not show up in the data_set returned by
                # run(), but it is saved to the metadata
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.data_set.add_metadata({'loop': {'ts_end': ts}})
                self.data_set.finalize()

    def _run_loop(self, first_delay=0, action_indices=(),
                  loop_indices=(), current_values=(),
                  **ignore_kwargs):
        """
        the routine that actually executes the loop, and can be called
        from one loop to execute a nested loop

        first_delay: any delay carried over from an outer loop
        action_indices: where we are in any outer loop action arrays
        loop_indices: setpoint indices in any outer loops
        current_values: setpoint values in any outer loops
        signal_queue: queue to communicate with main process directly
        ignore_kwargs: for compatibility with other loop tasks
        """

        # at the beginning of the loop, the time to wait after setting
        # the loop parameter may be increased if an outer loop requested longer
        delay = max(self.delay, first_delay)

        callables = self._compile_actions(self.actions, action_indices)

        t0 = time.time()
        last_task = t0
        imax = len(self.sweep_values)
        for i, value in enumerate(self.sweep_values):
            if self.progress_interval is not None:
                tprint('loop %s: %d/%d (%.1f [s])' % (
                    self.sweep_values.name, i, imax, time.time() - t0),
                    dt=self.progress_interval, tag='outerloop')

            set_val = self.sweep_values.set(value)

            new_indices = loop_indices + (i,)
            new_values = current_values + (value,)
            data_to_store = {}

            if hasattr(self.sweep_values, "parameters"):
                set_name = self.data_set.action_id_map[action_indices]
                if hasattr(self.sweep_values, 'aggregate'):
                    value = self.sweep_values.aggregate(*set_val)
                self.data_set.store(new_indices, {set_name: value})
                for j, val in enumerate(set_val):
                    set_index = action_indices + (j+1, )
                    set_name = (self.data_set.action_id_map[set_index])
                    data_to_store[set_name] = val
            else:
                set_name = self.data_set.action_id_map[action_indices]
                data_to_store[set_name] = value

            self.data_set.store(new_indices, data_to_store)

            if not self._nest_first:
                # only wait the delay time if an inner loop will not inherit it
                self._wait(delay)

            try:
                for f in callables:
                    f(first_delay=delay,
                      loop_indices=new_indices,
                      current_values=new_values)

                    # after the first action, no delay is inherited
                    delay = 0
            except _QcodesBreak:
                break

            # after the first setpoint, delay reverts to the loop delay
            delay = self.delay

            # now check for a background task and execute it if it's
            # been long enough since the last time
            # don't let exceptions in the background task interrupt
            # the loop
            # if the background task fails twice consecutively, stop
            # executing it
            if self.bg_task is not None:
                t = time.time()
                if t - last_task >= self.bg_min_delay:
                    try:
                        self.bg_task()
                    except Exception:
                        log.exception("Failed to execute bg task")
                        self.bg_task = None
                        return
                    last_task = t

        if self.progress_interval is not None:
            # final progress note: set dt=-1 so it *always* prints
            tprint('loop %s DONE: %d/%d (%.1f [s])' % (
                   self.sweep_values.name, i + 1, imax, time.time() - t0),
                   dt=-1, tag='outerloop')

        # run the background task one last time to catch the last setpoint(s)
        if self.bg_task is not None:
            self.bg_task()

        # the loop is finished - run the .then actions
        for f in self._compile_actions(self.then_actions, ()):
            f()

        # run the bg_final_task from the bg_task:
        if self.bg_final_task is not None:
            self.bg_final_task()


    def _wait(self, delay):
        if delay:
            finish_clock = time.perf_counter() + delay

            if self._monitor:
                # TODO - perhpas pass self._check_signal in here
                # so that we can halt within monitor.call if it
                # lasts a very long time?
                self._monitor.call(finish_by=finish_clock)

            while True:
                self._check_signal()
                t = wait_secs(finish_clock)
                time.sleep(min(t, self.signal_period))
                if t <= self.signal_period:
                    break
        else:
            self._check_signal()


class _QuietInterrupt(Exception):
    pass


class _DebugInterrupt(Exception):
    pass
