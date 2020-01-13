from typing import Optional, Sequence
from datetime import datetime

from qcodes.instrument.parameter import Parameter
from qcodes.loops import Loop
from qcodes.actions import _actions_snapshot
from qcodes.utils.helpers import full_class
from qcodes.utils.metadata import Metadatable


class Measure(Metadatable):
    """
    Create a DataSet from a single (non-looped) set of actions.

    Args:
        *actions (Any): sequence of actions to perform. Any action that is
            valid in a ``Loop`` can be used here. If an action is a gettable
            ``Parameter``, its output will be included in the DataSet.
            Scalars returned by an action will be saved as length-1 arrays,
            with a dummy setpoint for consistency with other DataSets.
    """
    dummy_parameter = Parameter(name='single',
                                label='Single Measurement',
                                set_cmd=None, get_cmd=None)

    def __init__(self, *actions):
        super().__init__()
        self._dummyLoop = Loop(self.dummy_parameter[0]).each(*actions)

    def run_temp(self, **kwargs):
        """
        Wrapper to run this measurement as a temporary data set
        """
        return self.run(quiet=True, location=False, **kwargs)

    def get_data_set(self, *args, **kwargs):
        return self._dummyLoop.get_data_set(*args, **kwargs)

    def run(self, use_threads=False, quiet=False, station=None, **kwargs):
        """
        Run the actions in this measurement and return their data as a DataSet

        Args:
            quiet (Optional[bool]): Set True to not print anything except
                errors. Default False.

            station (Optional[Station]): the ``Station`` this measurement
                pertains to. Defaults to ``Station.default`` if one is defined.
                Only used to supply metadata.

            use_threads (Optional[bool]): whether to parallelize ``get``
                operations using threads. Default False.

            location (Optional[Union[str, bool]]): the location of the
                DataSet, a string whose meaning depends on formatter and io,
                or False to only keep in memory. May be a callable to provide
                automatic locations. If omitted, will use the default
                DataSet.location_provider

            name (Optional[str]): if location is default or another provider
                function, name is a string to add to location to make it more
                readable/meaningful to users

            formatter (Optional[Formatter]): knows how to read and write the
                file format. Default can be set in DataSet.default_formatter

            io (Optional[io_manager]): knows how to connect to the storage
                (disk vs cloud etc)

        location, name formatter and io are passed to ``data_set.new_data``
        along with any other optional keyword arguments.

        returns:
            a DataSet object containing the results of the measurement
        """

        data_set = self._dummyLoop.get_data_set(**kwargs)

        # set the DataSet to local for now so we don't save it, since
        # we're going to massage it afterward
        original_location = data_set.location
        data_set.location = False

        # run the measurement as if it were a Loop
        self._dummyLoop.run(use_threads=use_threads,
                            station=station, quiet=True)

        # look for arrays that are unnecessarily nested, and un-nest them
        all_unnested = True
        for array in data_set.arrays.values():
            if array.ndim == 1:
                if array.is_setpoint:
                    dummy_setpoint = array
                else:
                    # we've found a scalar - so keep the dummy setpoint
                    all_unnested = False
            else:
                # The original return was an array, so take off the extra dim.
                # (This ensures the outer dim length was 1, otherwise this
                # will raise a ValueError.)
                array.ndarray.shape = array.ndarray.shape[1:]

                # TODO: DataArray.shape masks ndarray.shape, and a user *could*
                # change it, thinking they were reshaping the underlying array,
                # but this would a) not actually reach the ndarray right now,
                # and b) if it *did* and the array was reshaped, this array
                # would be out of sync with its setpoint arrays, so bad things
                # would happen. So we probably want some safeguards in place
                # against this
                array.shape = array.ndarray.shape

                array.set_arrays = array.set_arrays[1:]

                array.init_data()

        # Do we still need the dummy setpoint array at all?
        if all_unnested:
            del data_set.arrays[dummy_setpoint.array_id]
            if hasattr(data_set, 'action_id_map'):
                del data_set.action_id_map[dummy_setpoint.action_indices]

        # now put back in the DataSet location and save it
        data_set.location = original_location
        data_set.write()

        # metadata: ActiveLoop already provides station snapshot, but also
        # puts in a 'loop' section that we need to replace with 'measurement'
        # but we use the info from 'loop' to ensure consistency and avoid
        # duplication.
        LOOP_SNAPSHOT_KEYS = ['ts_start', 'ts_end', 'use_threads']
        data_set.add_metadata({'measurement': {
            k: data_set.metadata['loop'][k] for k in LOOP_SNAPSHOT_KEYS
        }})
        del data_set.metadata['loop']

        # actions are included in self.snapshot() rather than in
        # LOOP_SNAPSHOT_KEYS because they are useful if someone just
        # wants a local snapshot of the Measure object
        data_set.add_metadata({'measurement': self.snapshot()})

        data_set.save_metadata()

        if not quiet:
            print(repr(data_set))
            print(datetime.now().strftime('acquired at %Y-%m-%d %H:%M:%S'))

        return data_set

    def snapshot_base(self, update: bool = False,
                      params_to_skip_update: Optional[Sequence[str]] = None):
        return {
            '__class__': full_class(self),
            'actions': _actions_snapshot(self._dummyLoop.actions, update)
        }
