from enum import Enum
from datetime import datetime
import time

from .manager import get_data_manager, NoData
from .gnuplot_format import GNUPlotFormat
from .io import DiskIO
from qcodes.utils.helpers import DelegateAttributes


class DataMode(Enum):
    LOCAL = 1
    PUSH_TO_SERVER = 2
    PULL_FROM_SERVER = 3


SERVER_MODES = set((DataMode.PULL_FROM_SERVER, DataMode.PUSH_TO_SERVER))


def new_data(location=None, name=None, overwrite=False, io=None,
             data_manager=None, mode=DataMode.LOCAL, **kwargs):
    '''
    Create a new DataSet. Arguments are the same as DataSet constructor, plus:

    overwrite: Are we allowed to overwrite an existing location? default False

    location: can be a location string, but can also be a callable (a function
        of one required parameter, the io manager, and an optional name) to
        generate an automatic location, or False to denote an
        only-in-memory temporary DataSet.
        Note that the full path to or physical location of the data is a
        combination of io + location. the default DiskIO sets the base
        directory, which this location sits inside.
        defaults to DataSet.location_provider

    name: an optional string to be passed to location_provider to augment
        the automatic location with something meaningful
    '''
    if io is None:
        io = DataSet.default_io

    if location is None:
        location = DataSet.location_provider(io, name)
    elif callable(location):
        location = location(io)

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
    '''
    Load an existing DataSet. Arguments are a subset of the DataSet
    constructor:

    location: a string for the location to load from
        if omitted (None) defaults to the current live DataSet.
        `mode` is determined automatically from location: PULL_FROM_SERVER if
        this is the live DataSet, otherwise LOCAL
        Note that the full path to or physical location of the data is a
        combination of io + location. the default DiskIO sets the base
        directory, which this location sits inside.

    data_manager: usually omitted (default None) to get the default
        DataManager. load_data will not start a DataManager but may
        query an existing one to determine (and pull) the live data

    formatter: as in DataSet
    io: as in DataSet
    '''
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
        return DataSet(location=location, formatter=formatter, io=io,
                       mode=DataMode.LOCAL)


def _get_live_data(data_manager):
    live_data = data_manager.ask('get_data')
    if live_data is None or isinstance(live_data, NoData):
        raise RuntimeError('DataManager has no live data')

    live_data.mode = DataMode.PULL_FROM_SERVER
    return live_data


class TimestampLocation:
    '''
    This is the default DataSet Location provider.
    It provides a callable of one parameter (the io manager) that
    returns a new location string, which is currently unused.
    Uses `io.list(location)` to search for existing data at this location

    Constructed with one parameter, a datetime.strftime format string,
    which can include slashes (forward and backward are equivalent)
    to create folder structure.
    Default format string is '%Y-%m-%d/%H-%M-%S'
    '''
    def __init__(self, fmt='%Y-%m-%d/%H-%M-%S'):
        self.fmt = fmt

    def __call__(self, io, name=None):
        location = base_location = datetime.now().strftime(self.fmt)

        if name:
            location += '_' + name

        for char in map(chr, range(ord('a'), ord('z') + 2)):
            if not io.list(location):
                break
            location = base_location + '_' + char
        else:
            raise FileExistsError('Too many files with this timestamp')

        return location


class DataSet(DelegateAttributes):
    '''
    A container for one complete measurement loop
    May contain many individual arrays with potentially different
    sizes and dimensionalities.

    Normally a DataSet should not be instantiated directly, but through
    new_data or load_data

    location: where this data set is stored, also the DataSet's identifier.
        location=False or None means this is a temporary DataSet and
        cannot be stored or read.
        Note that the full path to or physical location of the data is a
        combination of io + location. the default DiskIO sets the base
        directory, which this location sits inside.

    arrays: a dict of array_id: DataArray's contained in this DataSet

    mode: sets whether and how this instance connects to a DataServer
        DataMode.LOCAL: this DataSet doesn't communicate across processes,
            ie it lives entirely either in the main proc, or in the DataServer
        DataMode.PUSH_TO_SERVER: no local copy of data, just pushes each
            measurement to a DataServer
        DataMode.PULL_FROM_SERVER: pulls changes from the DataServer
            on calling sync(). Reverts to local if and when this
            DataSet stops being the live measurement

    data_manager: usually omitted (default None) to get the default
        DataManager. But False is different: that means do NOT connect
        to any DataManager (implies mode=LOCAL)

    formatter: knows how to read and write the file format

    io: knows how to connect to the storage (disk vs cloud etc)
        The default (stored in class attribute DataSet.default_io) is
        DiskIO('.') which says the root data storage directory is the
        current working directory, ie where you started the notebook or python.

    write_period: seconds (default 5) between saves to disk. This only applies
        if mode=LOCAL, otherwise the DataManager handles this (and generally
        writes more often because it's not tying up the main process to do so).
        use None to disable writing from calls to self.store
    '''

    # ie data_array.arrays['vsd'] === data_array.vsd
    delegate_attr_dicts = ['arrays']

    default_io = DiskIO('.')
    default_formatter = GNUPlotFormat()
    location_provider = TimestampLocation()

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
        else:
            self.read()

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
        '''
        Configure this DataSet as the DataServer copy
        Should be run only by the DataServer itself.
        '''
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
        '''
        indicate whether this DataSet thinks it is live in the DataServer
        without actually talking to the DataServer or syncing with it
        '''
        return self.mode in SERVER_MODES and self.data_manager and True

    @property
    def is_on_server(self):
        '''
        Check whether this DataSet is being mirrored in the DataServer
        If it thought it was but isn't, convert it to mode=LOCAL
        '''
        if not self.is_live_mode or self.location is False:
            return False

        with self.data_manager.query_lock:
            live_location = self.data_manager.ask('get_data', 'location')
            return self.location == live_location

    def sync(self):
        '''
        synchronize this data set with a possibly newer version either
        in storage or on the DataServer, depending on its mode

        returns: boolean, is this DataSet live on the server
        '''
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
                # TODO: can we reduce the amount of data to send?
                # seems like in the most general case this would need to
                # remember each client DataSet on the server, and what has
                # changed since that particular client last synced
                # (at least first and last pt)
                live_data = self.data_manager.ask('get_data').arrays
                for array_id in self.arrays:
                    self.arrays[array_id].ndarray = live_data[array_id].ndarray

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

    def add_array(self, data_array):
        '''
        add one DataArray to this DataSet

        note: DO NOT just set data_set.arrays[id] = data_array
        because this will not check for overriding, nor set the
        reference back to this DataSet. It would also allow you to
        load the array in with different id than it holds itself.

        '''
        # TODO: mask self.arrays so you *can't* set it directly

        if data_array.array_id in self.arrays:
            raise ValueError('array_id {} already exists in this '
                             'DataSet'.format(data_array.array_id))
        self.arrays[data_array.array_id] = data_array

        # back-reference to the DataSet
        data_array.data_set = self

    def _clean_array_ids(self, arrays):
        '''
        replace action_indices tuple with compact string array_ids
        stripping off as much extraneous info as possible
        '''
        action_indices = [array.action_indices for array in arrays]
        array_names = set(array.name for array in arrays)
        for name in array_names:
            param_arrays = [array for array in arrays
                            if array.name == name]
            if len(param_arrays) == 1:
                # simple case, only one param with this name, id = name
                param_arrays[0].array_id = name
                continue

            # partition into set and measured arrays (weird use case, but
            # it'll happen, if perhaps only in testing)
            set_param_arrays = [pa for pa in param_arrays
                                if pa.set_arrays[-1] == pa]
            meas_param_arrays = [pa for pa in param_arrays
                                 if pa.set_arrays[-1] != pa]
            if len(set_param_arrays) and len(meas_param_arrays):
                # if the same param is in both set and measured,
                # suffix the set with '_set'
                self._clean_param_ids(set_param_arrays, name + '_set')
                self._clean_param_ids(meas_param_arrays, name)
            else:
                # if either only set or only measured, no suffix
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
        '''
        Set some collection of data points

        loop_indices: the indices within whatever loops we are inside
        values: a dict of action_index:value or array_id:value
            where value may be an arbitrarily nested list, to record
            many values at once into one array
        '''
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
        '''
        Read the whole DataSet from storage, overwriting the local data
        '''
        if self.location is False:
            return
        self.formatter.read(self)

    def write(self):
        '''
        Write the whole (or only changed parts) DataSet to storage,
        overwriting the existing storage if any.
        '''
        if self.mode != DataMode.LOCAL:
            raise RuntimeError('This object is connected to a DataServer, '
                               'which handles writing automatically.')

        if self.location is False:
            return
        self.formatter.write(self)

    def finalize(self):
        '''
        Mark the DataSet as complete
        '''
        if self.mode == DataMode.PUSH_TO_SERVER:
            self.data_manager.ask('end_data')
        elif self.mode == DataMode.LOCAL:
            self.write()
        else:
            raise RuntimeError('This mode does not allow finalizing',
                               self.mode)

    def plot(self, cut=None):
        pass  # TODO

    def __repr__(self):
        out = '{}: {}, location={}'.format(
            self.__class__.__name__, self.mode, repr(self.location))
        for array_id, array in self.arrays.items():
            out += '\n   {}: {}'.format(array_id, array.name)

        return out
