"""DataSet class and factory functions."""

from enum import Enum
import time
import logging
from copy import deepcopy

from .manager import get_data_manager, NoData
from .gnuplot_format import GNUPlotFormat
from .io import DiskIO
from .location import FormatLocation
from qcodes.utils.helpers import DelegateAttributes, full_class, deep_update


class DataMode(Enum):

    """Server connection modes supported by a DataSet."""

    LOCAL = 1
    PUSH_TO_SERVER = 2
    PULL_FROM_SERVER = 3


SERVER_MODES = set((DataMode.PULL_FROM_SERVER, DataMode.PUSH_TO_SERVER))


def new_data(location=None, loc_record=None, name=None, overwrite=False,
             io=None, data_manager=None, mode=DataMode.LOCAL, **kwargs):
    """
    Create a new DataSet.

    Args:
        location (str or callable or False, optional): If you provide a string,
            it must be an unused location in the io manager. Can also be:
            - a callable ``location provider`` with one required parameter
              (the io manager), and one optional (``record`` dict),
              which returns a location string when called
            - ``False`` - denotes an only-in-memory temporary DataSet.
            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.
            Default ``DataSet.location_provider`` which is initially
            ``FormatLocation()``

        loc_record (dict, optional): If location is a callable, this will be
            passed to it as ``record``

        name (str, optional): overrides the ``name`` key in the ``loc_record``.

        overwrite (bool): Are we allowed to overwrite an existing location?
            Default False.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

        data_manager (DataManager or False, optional): manager for the
            ``DataServer`` that offloads storage and syncing of this
            ``DataSet``. Usually omitted (default None) to use the default
            from ``get_data_manager()``. If ``False``, this ``DataSet`` will
            store itself.

        mode (DataMode, optional): connection type to the ``DataServer``.
            ``DataMode.LOCAL``: this DataSet doesn't communicate across
                processes.
            ``DataMode.PUSH_TO_SERVER``: no local copy of data, just pushes
                each measurement to a ``DataServer``.
            ``DataMode.PULL_FROM_SERVER``: pulls changes from the
                ``DataServer`` on calling ``self.sync()``. Reverts to local if
                and when it stops being the live measurement.
            Default ``DataMode.LOCAL``.

        arrays (dict, optional): dict of ``array_id: DataArray``, can also be
            added later with ``self.add_array(array)``.

        formatter (Formatter, optional): sets the file format/structure to
            write (and read) with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        write_period (float or None, optional): Only if ``mode=LOCAL``, seconds
            between saves to disk. If not ``LOCAL``, the ``DataServer`` handles
            this and generally writes more often. Use None to disable writing
            from calls to ``self.store``. Default 5.

    Returns:
        A new ``DataSet`` object ready for storing new data in.
    """
    if io is None:
        io = DataSet.default_io

    if name is not None:
        if not loc_record:
            loc_record = {}
        loc_record['name'] = name

    if location is None:
        location = DataSet.location_provider

    if callable(location):
        location = location(io, record=loc_record)

    if location and (not overwrite) and io.list(location):
        raise FileExistsError('"' + location + '" already has data')

    if data_manager is False:
        if mode != DataMode.LOCAL:
            raise ValueError('DataSets without a data_manager must be local')
    elif data_manager is None:
        data_manager = get_data_manager()

    return DataSet(location=location, io=io, data_manager=data_manager,
                   mode=mode, **kwargs)


def load_data(location=None, data_manager=None, formatter=None, io=None):
    """
    Load an existing DataSet.

    The resulting ``DataSet.mode`` is determined automatically from location:
    PULL_FROM_SERVER if this is the live DataSet, otherwise LOCAL

    Args:
        location (str, optional): the location to load from. Default is the
            current live DataSet.
            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.

        data_manager (DataManager or False, optional): manager for the
            ``DataServer`` that offloads storage and syncing of this
            ``DataSet``. Usually omitted (default None) to use the default
            from ``get_data_manager()``. If ``False``, this ``DataSet`` will
            store itself. ``load_data`` will not start a DataManager but may
            query an existing one to determine (and pull) the live data.

        formatter (Formatter, optional): sets the file format/structure to
            read with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

    Returns:
        A new ``DataSet`` object loaded with pre-existing data.
    """
    if data_manager is None:
        data_manager = get_data_manager(only_existing=True)

    if location is None:
        if not data_manager:
            raise RuntimeError('Live data requested but DataManager does '
                               'not exist or was requested not to be used')

        return _get_live_data(data_manager)

    elif location is False:
        raise ValueError('location=False means a temporary DataSet, '
                         'which is incompatible with load_data')

    elif (data_manager and
            location == data_manager.ask('get_data', 'location')):
        return _get_live_data(data_manager)

    else:
        data = DataSet(location=location, formatter=formatter, io=io,
                       mode=DataMode.LOCAL)
        data.read_metadata()
        data.read()
        return data


def _get_live_data(data_manager):
    live_data = data_manager.ask('get_data')
    if live_data is None or isinstance(live_data, NoData):
        raise RuntimeError('DataManager has no live data')

    live_data.mode = DataMode.PULL_FROM_SERVER
    return live_data


class DataSet(DelegateAttributes):

    """
    A container for one complete measurement loop.

    May contain many individual arrays with potentially different
    sizes and dimensionalities.

    Normally a DataSet should not be instantiated directly, but through
    ``new_data`` or ``load_data``.

    Args:
        location (str or False): A location in the io manager, or ``False`` for
            an only-in-memory temporary DataSet.
            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

        data_manager (DataManager or False, optional): manager for the
            ``DataServer`` that offloads storage and syncing of this
            ``DataSet``. Usually omitted (default None) to use the default
            from ``get_data_manager()``. If ``False``, this ``DataSet`` will
            store itself.

        mode (DataMode, optional): connection type to the ``DataServer``.
            ``DataMode.LOCAL``: this DataSet doesn't communicate across
                processes.
            ``DataMode.PUSH_TO_SERVER``: no local copy of data, just pushes
                each measurement to a ``DataServer``.
            ``DataMode.PULL_FROM_SERVER``: pulls changes from the
                ``DataServer`` on calling ``self.sync()``. Reverts to local if
                and when it stops being the live measurement.
            Default ``DataMode.LOCAL``.

        arrays (dict, optional): dict of ``array_id: DataArray``, can also be
            added later with ``self.add_array(array)``.

        formatter (Formatter, optional): sets the file format/structure to
            write (and read) with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        write_period (float or None, optional): Only if ``mode=LOCAL``, seconds
            between saves to disk. If not ``LOCAL``, the ``DataServer`` handles
            this and generally writes more often. Use None to disable writing
            from calls to ``self.store``. Default 5.
    """

    # ie data_set.arrays['vsd'] === data_set.vsd
    delegate_attr_dicts = ['arrays']

    default_io = DiskIO('.')
    default_formatter = GNUPlotFormat()
    location_provider = FormatLocation()

    def __init__(self, location=None, mode=DataMode.LOCAL, arrays=None,
                 data_manager=None, formatter=None, io=None, write_period=5):
        if location is False or isinstance(location, str):
            self.location = location
        else:
            raise ValueError('unrecognized location ' + repr(location))

        # TODO: when you change formatter or io (and there's data present)
        # make it all look unsaved
        self.formatter = formatter or self.default_formatter
        self.io = io or self.default_io

        self.write_period = write_period
        self.last_write = 0

        self.metadata = {}

        self.arrays = {}
        if arrays:
            self.action_id_map = self._clean_array_ids(arrays)
            for array in arrays:
                self.add_array(array)

        if data_manager is None and mode in SERVER_MODES:
            data_manager = get_data_manager()

        if mode == DataMode.LOCAL:
            self._init_local()
        elif mode == DataMode.PUSH_TO_SERVER:
            self._init_push_to_server(data_manager)
        elif mode == DataMode.PULL_FROM_SERVER:
            self._init_live(data_manager)
        else:
            raise ValueError('unrecognized DataSet mode', mode)

    def _init_local(self):
        self.mode = DataMode.LOCAL

        if self.arrays:
            for array in self.arrays.values():
                array.init_data()

    def _init_push_to_server(self, data_manager):
        self.mode = DataMode.PUSH_TO_SERVER

        # If some code was not available when data_manager was started,
        # we can't unpickle it on the other end.
        # So we'll try, then restart if this error occurs, then try again.
        #
        # This still has a pitfall, if code has been *changed* since
        # starting the server, it will still have the old version and
        # everything will look fine but it won't have the new behavior.
        # If the user does that, they need to manually restart the server,
        # using:
        #     data_manager.restart()
        try:
            data_manager.ask('new_data', self)
        except AttributeError:
            data_manager.restart()
            data_manager.ask('new_data', self)

        # need to set data_manager *after* sending to data_manager because
        # we can't (and shouldn't) send data_manager itself through a queue
        self.data_manager = data_manager

    def init_on_server(self):
        """
        Configure this DataSet as the DataServer copy
        Should be run only by the DataServer itself.
        """
        if not self.arrays:
            raise RuntimeError('A server-side DataSet needs DataArrays.')

        self._init_local()

    def _init_live(self, data_manager):
        self.mode = DataMode.PULL_FROM_SERVER
        self.data_manager = data_manager
        with data_manager.query_lock:
            if self.is_on_server:
                live_obj = data_manager.ask('get_data')
                self.arrays = live_obj.arrays
            else:
                self._init_local()

    @property
    def is_live_mode(self):
        """
        indicate whether this DataSet thinks it is live in the DataServer
        without actually talking to the DataServer or syncing with it
        """
        return self.mode in SERVER_MODES and self.data_manager and True

    @property
    def is_on_server(self):
        """
        Check whether this DataSet is being mirrored in the DataServer
        If it thought it was but isn't, convert it to mode=LOCAL
        """
        if not self.is_live_mode or self.location is False:
            return False

        with self.data_manager.query_lock:
            live_location = self.data_manager.ask('get_data', 'location')
            return self.location == live_location

    def sync(self):
        """
        synchronize this data set with a possibly newer version either
        in storage or on the DataServer, depending on its mode

        returns: boolean, is this DataSet live on the server
        """
        # TODO: sync implies bidirectional... and it could be!
        # we should keep track of last sync timestamp and last modification
        # so we can tell whether this one, the other one, or both copies have
        # changed (and I guess throw an error if both did? Would be cool if we
        # could find a robust and intuitive way to make modifications to the
        # version on the DataServer from the main copy)
        if not self.is_live_mode:
            # LOCAL DataSet - just read it in
            # TODO: compare timestamps to know if we need to read?
            try:
                self.read()
            except IOError:
                # if no files exist, they probably haven't been created yet.
                pass
            return False
            # TODO - for remote live plotting, maybe set some timestamp
            # threshold and call it static after it's been dormant a long time?
            # I'm thinking like a minute, or ten? Maybe it's configurable?

        with self.data_manager.query_lock:
            if self.is_on_server:
                synced_indices = {
                    array_id: array.get_synced_index()
                    for array_id, array in self.arrays.items()
                }

                changes = self.data_manager.ask('get_changes', synced_indices)

                for array_id, array_changes in changes.items():
                    self.arrays[array_id].apply_changes(**array_changes)

                measuring = self.data_manager.ask('get_measuring')
                if not measuring:
                    # we must have *just* stopped measuring
                    # but the DataSet is still on the server,
                    # so we got the data, and don't need to read.
                    self.mode = DataMode.LOCAL
                    return False
                return True
            else:
                # this DataSet *thought* it was on the server, but it wasn't,
                # so we haven't synced yet and need to read from storage
                self.mode = DataMode.LOCAL
                self.read()
                return False

    def get_changes(self, synced_index):
        changes = {}

        for array_id, synced_index in synced_index.items():
            array_changes = self.arrays[array_id].get_changes(synced_index)
            if array_changes:
                changes[array_id] = array_changes

        return changes

    def add_array(self, data_array):
        """
        add one DataArray to this DataSet

        note: DO NOT just set data_set.arrays[id] = data_array
        because this will not check for overriding, nor set the
        reference back to this DataSet. It would also allow you to
        load the array in with different id than it holds itself.

        """
        # TODO: mask self.arrays so you *can't* set it directly

        if data_array.array_id in self.arrays:
            raise ValueError('array_id {} already exists in this '
                             'DataSet'.format(data_array.array_id))
        self.arrays[data_array.array_id] = data_array

        # back-reference to the DataSet
        data_array.data_set = self

    def _clean_array_ids(self, arrays):
        """
        replace action_indices tuple with compact string array_ids
        stripping off as much extraneous info as possible
        """

        action_indices = [array.action_indices for array in arrays]
        for array in arrays:
            name = array.full_name
            if array.is_setpoint:
                if name:
                    name += '_set'
            array.array_id = name
        array_ids = set([array.array_id for array in arrays])
        for name in array_ids:
            param_arrays = [array for array in arrays if array.array_id == name]
            self._clean_param_ids(param_arrays, name)

        array_ids = [array.array_id for array in arrays]

        return dict(zip(action_indices, array_ids))

    def _clean_param_ids(self, arrays, name):
        # strip off as many leading equal indices as possible
        # and append the rest to the back of the name with underscores
        param_action_indices = [list(array.action_indices) for array in arrays]
        while all(len(ai) for ai in param_action_indices):
            if len(set(ai[0] for ai in param_action_indices)) == 1:
                for ai in param_action_indices:
                    ai[:1] = []
            else:
                break
        for array, ai in zip(arrays, param_action_indices):
            array.array_id = name + ''.join('_' + str(i) for i in ai)

    def store(self, loop_indices, ids_values):
        """
        Set some collection of data points

        loop_indices: the indices within whatever loops we are inside
        values: a dict of action_index:value or array_id:value
            where value may be an arbitrarily nested list, to record
            many values at once into one array
        """
        if self.mode == DataMode.PUSH_TO_SERVER:
            self.data_manager.write('store_data', loop_indices, ids_values)
        else:
            for array_id, value in ids_values.items():
                self.arrays[array_id][loop_indices] = value
            if (self.write_period is not None and
                    time.time() > self.last_write + self.write_period):
                self.write()
                self.last_write = time.time()

    def read(self):
        """Read the whole DataSet from storage, overwriting the local data."""
        if self.location is False:
            return
        self.formatter.read(self)

    def read_metadata(self):
        """Read the metadata from storage, overwriting the local data."""
        if self.location is False:
            return
        self.formatter.read_metadata(self)

    def write(self):
        """Write updates to the DataSet to storage."""
        if self.mode != DataMode.LOCAL:
            raise RuntimeError('This object is connected to a DataServer, '
                               'which handles writing automatically.')

        if self.location is False:
            return
        self.formatter.write(self, self.io, self.location)

    def write_copy(self, path=None, io_manager=None, location=None):
        """
        Write a new complete copy of this DataSet to storage.

        Args:
            path (str, optional): An absolute path on this system to write to.
                If you specify this, you may not include either ``io_manager``
                or ``location``.

            io_manager (io_manager, optional): A new ``io_manager`` to use with
                either the ``DataSet``'s same or a new ``location``.

            location (str, optional): A new ``location`` to write to, using
                either this ``DataSet``'s same or a new ``io_manager``.
        """
        if io_manager is not None or location is not None:
            if path is not None:
                raise TypeError('If you provide io_manager or location '
                                'to write_copy, you may not provide path.')
            if io_manager is None:
                io_manager = self.io
            elif location is None:
                location = self.location
        elif path is not None:
            io_manager = DiskIO(None)
            location = path
        else:
            raise TypeError('You must provide at least one argument '
                            'to write_copy')

        if location is False:
            raise ValueError('write_copy needs a location, not False')

        lsi_cache = {}
        mr_cache = {}
        for array_id, array in self.arrays.items():
            lsi_cache[array_id] = array.last_saved_index
            mr_cache[array_id] = array.modified_range
            # array.clear_save() is not enough, we _need_ to set modified_range
            # TODO - identify *when* clear_save is not enough, and fix it
            # so we *can* use it. That said, maybe we will *still* want to
            # use the full array here no matter what, or strip trailing NaNs
            # separately, either here or in formatter.write?
            array.last_saved_index = None
            array.modified_range = (0, array.ndarray.size - 1)

        try:
            self.formatter.write(self, io_manager, location)
            self.snapshot()
            self.formatter.write_metadata(self, io_manager, location,
                                          read_first=False)
        finally:
            for array_id, array in self.arrays.items():
                array.last_saved_index = lsi_cache[array_id]
                array.modified_range = mr_cache[array_id]

    def add_metadata(self, new_metadata):
        """Update DataSet.metadata with additional data."""
        deep_update(self.metadata, new_metadata)

    def save_metadata(self):
        """Evaluate and save the DataSet's metadata."""
        if self.location is not False:
            self.snapshot()
            self.formatter.write_metadata(self, self.io, self.location)

    def finalize(self):
        """Mark the DataSet as complete."""
        if self.mode == DataMode.PUSH_TO_SERVER:
            self.data_manager.ask('end_data')
        elif self.mode == DataMode.LOCAL:
            self.write()
        else:
            raise RuntimeError('This mode does not allow finalizing',
                               self.mode)
        self.save_metadata()

    def snapshot(self, update=False):
        """JSON state of the DataSet."""
        array_snaps = {}
        for array_id, array in self.arrays.items():
            array_snaps[array_id] = array.snapshot(update=update)

        self.metadata.update({
            '__class__': full_class(self),
            'location': self.location,
            'arrays': array_snaps,
            'formatter': full_class(self.formatter),
            'io': repr(self.io)
        })
        return deepcopy(self.metadata)

    def get_array_metadata(self, array_id):
        try:
            return self.metadata['arrays'][array_id]
        except:
            return None

    def __repr__(self):
        out = type(self).__name__ + ':'

        attrs = [['mode', self.mode],
                 ['location', repr(self.location)]]
        attr_template = '\n   {:8} = {}'
        for var, val in attrs:
            out += attr_template.format(var, val)

        arr_info = [['<Type>', '<array_id>', '<array.name>', '<array.shape>']]

        if hasattr(self, 'action_id_map'):
            id_items = [item for index, item in sorted(self.action_id_map.items())]
        else:
            id_items = self.arrays.keys()

        for array_id in id_items:
            array = self.arrays[array_id]
            setp = 'Setpoint' if array.is_setpoint else 'Measured'
            name = array.name or 'None'
            array_id = array_id or 'None'
            arr_info.append([setp, array_id, name, repr(array.shape)])

        column_lengths = [max(len(row[i]) for row in arr_info)
                          for i in range(len(arr_info[0]))]
        out_template = ('\n   '
                        '{info[0]:{lens[0]}} | {info[1]:{lens[1]}} | '
                        '{info[2]:{lens[2]}} | {info[3]}')

        for arr_info_i in arr_info:
            out += out_template.format(info=arr_info_i, lens=column_lengths)

        return out
