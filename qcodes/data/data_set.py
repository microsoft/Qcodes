from enum import Enum

from .manager import get_data_manager
from .format import GNUPlotFormat
from .io import DiskIO
from qcodes.utils.helpers import safe_getattr


class DataMode(Enum):
    LOCAL = 1
    PUSH_TO_SERVER = 2
    PULL_FROM_SERVER = 3


class DataSet(object):
    '''
    A container for one complete measurement loop
    May contain many individual arrays with potentially different
    sizes and dimensionalities.

    location: where this data set is stored, also its identifier.
        what exactly this means depends on io and formatter
        if you omit *everything*, will try to pull location (and mode)
        from the live measurement

    arrays: a dict of array_id: DataArray's contained in this DataSet

    mode: sets whether and how this instance connects to a DataServer
        DataMode.LOCAL: this DataSet doesn't communicate across processes,
            ie it lives entirely either in the main proc, or in the DataServer
        DataMode.PUSH_TO_SERVER: no local copy of data, just pushes each
            measurement to a DataServer
        DataMode.PULL_FROM_SERVER: pulls changes from the DataServer
            on calling sync(). Reverts to local if and when this
            DataSet stops being the live measurement

    formatter: knows how to read and write the file format

    io: knows how to connect to the storage (disk vs cloud etc)
    '''

    default_io = DiskIO('.')
    default_formatter = GNUPlotFormat()
    SERVER_MODES = set((DataMode.PULL_FROM_SERVER, DataMode.PUSH_TO_SERVER))

    def __init__(self, location=None, mode=None, arrays=None,
                 data_manager=None, formatter=None, io=None):
        self.location = location
        self.formatter = formatter or self.default_formatter
        self.io = io or self.default_io
        self.mode = mode

        if mode is None:
            if arrays:
                # no mode but arrays provided - assume the user is doing
                # local analysis and making a new local DataSet
                mode = DataMode.LOCAL
            else:
                # check if this is the live measurement, make it sync if it is
                mode = DataMode.PULL_FROM_SERVER

        self.arrays = {}
        if arrays:
            self.action_id_map = self._clean_array_ids(arrays)
            for array in arrays:
                self.add_array(array)

        if not data_manager:
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

        # need to set data_manager *after* _init_new_data because
        # we can't (and shouldn't) send data_manager through a queue
        self.data_manager = data_manager

    def init_on_server(self):
        if not self.arrays:
            raise RuntimeError('A server-side DataSet needs DataArrays.')

        self._init_local()

    def _init_live(self, data_manager):
        self.data_manager = data_manager
        with self.data_manager.query_lock:
            if self.check_live_data():
                live_obj = data_manager.ask('get_data')
                self.arrays = live_obj.arrays
                return

        self._init_local()

    @property
    def is_live(self):
        '''
        indicate whether this DataSet thinks it is live in the DataServer
        without actually talking to the DataServer or syncing with it
        '''
        return self.mode in self.SERVER_MODES and self.data_manager

    def check_live_data(self):
        '''
        check whether this DataSet really *is* the one in the DataServer
        and if it thought it was but isn't, convert it to mode=LOCAL
        '''
        if self.is_live:
            live_location = self.data_manager.ask('get_data', 'location')

            if self.location is None:
                # no location given yet, pull it from the live data
                self.location = live_location

            if self.location == live_location:
                return True

        if self.mode != DataMode.LOCAL:
            # just stopped being the live data - drop the data and reload it,
            # in case more arrived after the live measurement ended
            self.arrays = {}
            self._init_local()

        return False

    def sync(self):
        # TODO: anything we can do to reduce the amount of data to send?
        # seems like in the most general case this would need to remember
        # each client DataSet on the server, and what has changed since
        # that particular client last synced (at least first and last pt)
        if self.mode not in self.SERVER_MODES:
            return
        with self.data_manager.query_lock:
            if not self.check_live_data():
                # note that check_live_data switches us to LOCAL
                # and reads the data set back from storage,
                # if the server says the data isn't there anymore
                # (ie the measurement is done)
                return
            live_data = self.data_manager.ask('get_data').arrays

        for array_id in self.arrays:
            self.arrays[array_id].data = live_data[array_id].data

    def add_array(self, data_array):
        self.arrays[data_array.array_id] = data_array

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

            # otherwise, strip off as many leading equal indices as possible
            # and append the rest to the back of the name with underscores
            param_action_indices = [list(array.action_indices)
                                    for array in param_arrays]
            while all(len(ai) for ai in param_action_indices):
                if len(set(ai[0] for ai in param_action_indices)) == 1:
                    for ai in param_action_indices:
                        ai[:1] = []
                else:
                    break
            for array, ai in zip(param_arrays, param_action_indices):
                array.array_id = name + '_' + '_'.join(str(i) for i in ai)

        array_ids = [array.array_id for array in arrays]
        return dict(zip(action_indices, array_ids))

    def store(self, loop_indices, ids_values):
        '''
        set some collection of data points
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

    def read(self):
        self.formatter.read(self)

    def write(self):
        if self.mode != DataMode.LOCAL:
            raise RuntimeError('This object is connected to a DataServer '
                               'and should be saved from there.')

        self.formatter.write(self)

    def close(self):
        if self.mode == DataMode.PUSH_TO_SERVER:
            self.data_manager.ask('end_data')

    def __getattr__(self, key):
        '''
        alias arrays items as attributes
        ie data_array.arrays['vsd'] === data_array.vsd
        '''
        return safe_getattr(self, key, 'arrays')

    def __repr__(self):
        out = '{}: {}, location=\'{}\''.format(self.__class__.__name__,
                                               self.mode, self.location)
        for array_id, array in self.arrays.items():
            out += '\n   {}: {}'.format(array_id, array.name)

        return out
