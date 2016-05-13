from enum import Enum
from datetime import datetime
import time
import re
import string

from .manager import get_data_manager, NoData
from .gnuplot_format import GNUPlotFormat
from .io import DiskIO
from qcodes.utils.helpers import DelegateAttributes


class DataMode(Enum):
    LOCAL = 1
    PUSH_TO_SERVER = 2
    PULL_FROM_SERVER = 3


SERVER_MODES = set((DataMode.PULL_FROM_SERVER, DataMode.PUSH_TO_SERVER))


def new_data(location=None, loc_record=None, name=None, overwrite=False,
             io=None, data_manager=None, mode=DataMode.LOCAL, **kwargs):
    """
    Create a new DataSet. Arguments are the same as DataSet constructor, plus:

    overwrite: Are we allowed to overwrite an existing location? default False

    location: (default `DataSet.location_provider`) can be:
        - a location string
        - a callable with one required parameter, the io manager, and an
          optional `record` dict), to generate an automatic location
        - False - denotes an only-in-memory temporary DataSet.
        Note that the full path to or physical location of the data is a
        combination of io + location. the default DiskIO sets the base
        directory, which this location is a relative path inside.

    loc_record: an optional dict to use in formatting the location. If
        location is a callable, this will be passed to it as `record`

    name: an optional string to be passed to location_provider to augment
        the automatic location with something meaningful.
        If provided, name overrides the `name` key in the `loc_record`.

    overwrite: (default False) overwrite any data already at this location
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
        data.read()
        return data


def _get_live_data(data_manager):
    live_data = data_manager.ask('get_data')
    if live_data is None or isinstance(live_data, NoData):
        raise RuntimeError('DataManager has no live data')

    live_data.mode = DataMode.PULL_FROM_SERVER
    return live_data


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        '''Overrides string.Formatter.get_value'''
        if isinstance(key, (int)):
            return args[key]
        else:
            return kwargs.get(key, '{{{0}}}'.format(key))


class FormatLocation:
    """
    This is the default DataSet Location provider. It provides a callable that
    returns a new (not used by another DataSet) location string, based on a
    format string `fmt` and a

    The location string is formatted with the `fmt` string provided in
    `__init__` or `__call__`. And a dict provided through the record arguments.

    Default record items are `{date}`, `{time}`, and `{counter}`
    Record item priority from lowest to highest (double items will be
    overwritten)
    - `{counter}`, `{date}`, `{time}`
    - records dict from `__init__`
    - records dict from `__call__`
    Thus if any record dict contains a `date` keyword, it will no longer be
    auto-generated.

    Uses `io.list` to search for existing data at a matching location.

    `{counter}` is special and must NOT be provided in the record.
    If the format string contains `{counter}`, we look for existing files
    matching everything before the counter, then find the highest counter
    (integer) among those files and use the next value. That means the counter
    only increments as long as fields before it do not change, and files with
    incrementing counters will always group together and sort correctly in
    directory listings

    If the format string does not contain `{counter}` but the location we would
    return is occupied, we will add '_{counter}' to the end and do the same.

    Usage:
    ```
        loc_provider = FormatLocation(
            fmt='{date}/#{counter}_{time}_{name}_{label}')
        loc = loc_provider(DiskIO('.'),
                           record={'name': 'Rainbow', 'label': 'test'})
        loc
        > '2016-04-30/#001_13-28-15_Rainbow_test'
    ```
    Default format string is '{date}/{time}', and if `name` exists in record,
    it is '{date}/{time}_{name}'
    with `fmt_date='%Y-%m-%d'` and `fmt_time='%H-%M-%S'`
    """
    default_fmt = '{date}/{time}'

    def __init__(self, fmt=None, fmt_date=None, fmt_time=None,
                 fmt_counter=None, record=None):

        self.fmt = fmt or self.default_fmt
        self.fmt_date = fmt_date or '%Y-%m-%d'
        self.fmt_time = fmt_time or '%H-%M-%S'
        self.fmt_counter = fmt_counter or '{:03}'
        self.base_record = record
        self.formatter = SafeFormatter()

        for testval in (1, 23, 456, 7890):
            if self._findint(self.fmt_counter.format(testval)) != testval:
                raise ValueError('fmt_counter must produce a correct integer '
                                 'representation of its argument (eg "{:03}")',
                                 fmt_counter)

    def _findint(self, s):
        try:
            return int(re.findall(r'\d+', s)[0])
        except:
            return 0

    def __call__(self, io, record=None):
        loc_fmt = self.fmt

        time_now = datetime.now()
        date = time_now.strftime(self.fmt_date)
        time = time_now.strftime(self.fmt_time)
        format_record = {'date': date, 'time': time}

        if self.base_record:
            format_record.update(self.base_record)
        if record:
            format_record.update(record)

        if 'counter' in format_record:
            raise KeyError('you must not provide a counter in your record.',
                           format_record)

        if ('name' in format_record) and ('{name}' not in loc_fmt):
            loc_fmt += '_{name}'

        if '{counter}' not in loc_fmt:
            location = self.formatter.format(loc_fmt, **format_record)
            if io.list(location):
                loc_fmt += '_{counter}'
                # redirect to the counter block below, but starting from 2
                # because the already existing file counts like 1
                existing_count = 1
            else:
                return location
        else:
            # if counter is already in loc_fmt, start from 1
            existing_count = 0

        # now search existing files for the next allowed counter

        head_fmt = loc_fmt.split('{counter}', 1)[0]
        # io.join will normalize slashes in head to match the locations
        # returned by io.list
        head = io.join(self.formatter.format(head_fmt, **format_record))

        file_list = io.list(head + '*', maxdepth=0, include_dirs=True)

        for f in file_list:
            cnt = self._findint(f[len(head):])
            existing_count = max(existing_count, cnt)

        format_record['counter'] = self.fmt_counter.format(existing_count + 1)
        location = self.formatter.format(loc_fmt, **format_record)

        return location


class DataSet(DelegateAttributes):
    """
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
    """

    # ie data_array.arrays['vsd'] === data_array.vsd
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
        """
        Read the whole DataSet from storage, overwriting the local data
        """
        if self.location is False:
            return
        self.formatter.read(self)

    def write(self):
        """
        Write the whole (or only changed parts) DataSet to storage,
        overwriting the existing storage if any.
        """
        if self.mode != DataMode.LOCAL:
            raise RuntimeError('This object is connected to a DataServer, '
                               'which handles writing automatically.')

        if self.location is False:
            return
        self.formatter.write(self)

    def finalize(self):
        """
        Mark the DataSet as complete
        """
        if self.mode == DataMode.PUSH_TO_SERVER:
            self.data_manager.ask('end_data')
        elif self.mode == DataMode.LOCAL:
            self.write()
        else:
            raise RuntimeError('This mode does not allow finalizing',
                               self.mode)

    def __repr__(self):
        out = '{}: {}, location={}'.format(
            self.__class__.__name__, self.mode, repr(self.location))
        for array_id, array in self.arrays.items():
            out += '\n   {}: {}'.format(array_id, array.name)

        return out
