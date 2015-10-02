import math
import numpy as np
from datetime import datetime
import time
import csv
import re


class NoSweep(object):
    def data_point(self, *args, **kwargs):
        raise RuntimeError('no sweep to add data to')

    def update_storage_wrapper(self, *args, **kwargs):
        pass


class SweepStorage(object):
    '''
    base class for holding and storing sweep data
    subclasses should override update_storage and read

    inputs:
        location: a string, can represent different things
            depending on what the storage physically represents.
            can be a folder path (if the data is in separate files)
            or a file path (if all the data is in one file)
            or some other resource locator (for azure, AWS, etc)
        param_names: a sequence of strings listing all parameters to store,
            starting with the sweep parameters, from outer to innermost loop
        dim_sizes: a sequence containing the size of each dimension of
            the sweep, from outer to innermost loop
        storage_manager: if we're given a StorageManager here, then this
            instance is to be potentially connected to the storage server
            process via this storage_manager:
            If this is a new sweep, we create a mirror of it on the server.
            If it's an existing sweep, we check whether it's the current live
            sweep on the server, then allow syncing data from there
        passthrough: if True and we have a storage_manager, don't keep a copy
            of the data in this object, only in the server copy

        param_names and dim_sizes ust be provided if and only if this is a
        new sweep. If omitted, location must reference an existing sweep
    '''
    def __init__(self, location, param_names=None, dim_sizes=None,
                 storage_manager=None, passthrough=False):
        self.location = location  # TODO: auto location for new sweeps?

        self._storage_manager = storage_manager

        self._passthrough = storage_manager and passthrough

        if param_names and dim_sizes:
            self._init_new_sweep(param_names, dim_sizes)

        elif param_names is None and dim_sizes is None:
            # omitted names and dim_sizes? we're reading from a saved file
            self._init_existing_sweep()
        else:
            raise TypeError('you must provide either both or neither of '
                            'param_names and dim_sizes')

    def _init_new_sweep(self, param_names, dim_sizes):
        self.init_data(param_names, dim_sizes)
        self.new_indices = set()
        self.last_saved_index = -1

        if self._storage_manager:
            # If this class was not available when storage_manager was started,
            # we can't unpickle it on the other end.
            # So we'll try, then restart if this error occurs, then try again.
            #
            # This still has a pitfall, if the class has been *changed* since
            # starting the server, it will still have the old version.
            # If the user does that, they need to manually restart the server,
            # using:
            #     storage_manager.restart()
            try:
                # The copy to be sent to the server should NOT be marked as
                # syncable with the server, it's ON the server!
                self._sync_to_server = False
                self._storage_manager.ask('new_sweep', self)
            except AttributeError:
                self._storage_manager.restart()
                self._storage_manager.ask('new_sweep', self)

            self._sync_to_server = True

        else:
            self._sync_to_server = False

    def _init_existing_sweep(self):
        initialized = False
        if self._storage_manager:
            with self._storage_manager._query_lock:
                if self.check_live_sweep():
                    live_obj = self._storage_manager.ask('get_sweep')
                    self.init_data(live_obj.param_names,
                                   live_obj.dim_sizes,
                                   live_obj.data)
                    initialized = True

        if not initialized:
            self._sync_to_server = False
            self.read()

    def init_on_server(self):
        '''
        any changes we have to make when this object first arrives
        on the storage server?
        '''
        if self._passthrough:
            # this is where we're passing FROM!
            # so turn off passthrough here, and make the data arrays
            self._passthrough = False
            self.init_data(self.param_names, self.dim_sizes,
                           getattr(self, 'data', None))

    def check_live_sweep(self):
        if self._storage_manager:
            live_location = self._storage_manager.ask('get_sweep', 'location')
            self._sync_to_server = (self.location == live_location)
        else:
            self._sync_to_server = False

        return self._sync_to_server

    def sync_live(self):
        if not (self._sync_to_server and self._storage_manager):
            # assume that once we determine it's not the live sweep,
            # it will never be the live sweep again
            return

        self._passthrough = False
        with self._storage_manager._query_lock:
            self.check_live_sweep()
            if not self._sync_to_server:
                return
            self.data = self._storage_manager.ask('get_sweep', 'data')

    def init_data(self, param_names, dim_sizes, data=None):
        self.param_names = tuple(param_names)
        self.dim_sizes = tuple(dim_sizes)

        if data:
            self.data = data
        elif not self._passthrough:
            self.data = {}
            for pn in self.param_names + ('ts',):
                arr = np.ndarray(self.dim_sizes)
                arr.fill(math.nan)
                self.data[pn] = arr

    def set_point(self, indices, values):
        indices = tuple(indices)

        if self._sync_to_server:
            self._storage_manager.write('set_sweep_point', indices, values)

        if not self._passthrough:
            for pn, val in zip(self.param_names, values):
                self.data[pn][indices] = val

            self.data['ts'] = time.time()

            flat_index = np.ravel_multi_index(tuple(zip(indices)),
                                              self.dim_sizes)[0]
            self.new_indices.add(flat_index)

    def get(self, attr=None):
        '''
        getter for use by storage server - either return the whole
        object, or an attribute of it.
        '''
        if attr is None:
            return self
        else:
            return getattr(self, attr)

    def update_storage_wrapper(self):
        if self._passthrough:
            raise RuntimeError('This cobject has no data to save, '
                               'it\'s just a passthrough to the server.')
        if not self.new_indices:
            return

        self.update_storage()

        self.new_indices = set()
        self.last_saved_index = max(self.last_saved_index, *self.new_indices)

    def update_storage(self):
        '''
        write the data set (or changes to it) to storage
        based on the data and definition attributes:
            data
            param_names
            dim_sizes
        and also info about what has changed since last write:
            new_indices
            last_saved_index
        '''
        raise NotImplementedError

    def read(self):
        '''
        read from a file into the data and definition attributes:
            data (dict of numpy ndarray's)
            param_names
            dim_sizes
        the file format is expected to provide all of this info
        '''
        raise NotImplementedError


class MergedCSVStorage(SweepStorage):
    '''
    class that stores any sweep as a single file
    with all parameters together, one line per data point
    '''

    index_head_re = re.compile('^i_.+\((\d+)\)$')

    def __init__(self, location, *args, **kwargs):
        # set _path before __init__ so it can call .read if necessary
        if location[-4:] == '.csv':
            self._path = location
        else:
            self._path = location + '.csv'

        super().__init__(location, *args, **kwargs)

    def update_storage(self):
        first_new_index = min(self.new_indices)
        last_new_index = max(self.new_indices)

        if 0 < self.last_saved_index < first_new_index:
            with open(self._path, 'a') as f:
                writer = csv.writer(f)
                self._writerange(writer, self.last_saved_index,
                                 last_new_index + 1)
        else:
            with open(self._path, 'w') as f:
                writer = csv.writer(f)
                self._writeheader(writer)
                self._writerange(writer, 0, last_new_index + 1)

    def _writerange(self, writer, istart, iend):
        for i in range(istart, iend):
            indices = np.unravel_index(i, self.dim_sizes)

            tsraw = self.data['ts'][indices]
            if not math.isfinite(tsraw):
                continue  # don't store empty data points

            ts = datetime.fromtimestamp(tsraw)
            rowhead = (ts.strftime('%Y-%m-%d %H:%M:%S:%f'),) + indices

            row = tuple(self.data[pn][indices] for pn in self.param_names)

            writer.writerow(rowhead + row)

    def _writeheader(self, writer):
        loop_names = tuple('i_{}({})'.format(pn, ds)
                           for ds, pn in zip(self.dim_sizes, self.param_names))
        writer.writerow(('ts',) + loop_names + self.param_names)

    def _row(self, i):
        indices = np.unravel_index(i, self.dim_sizes)

        ts = datetime.fromtimestamp(self.data['ts'][indices])

        out = [ts.strftime('%Y-%m-%d %H:%M:%S:%f')]
        out += [self.data[pn][indices] for pn in self.param_names]

        return out

    def read(self):
        with open(self._path, 'r') as f:
            reader = csv.reader(f)
            head = reader.next()
            self._parse_header(head)

            dimensions = len(self.dim_sizes)

            for row in reader:
                indices = tuple(map(int, row[1: dimensions + 1]))
                for val, pn in zip(row[dimensions + 1:], self.param_names):
                    self.data[pn][indices] = val

    def _parse_header(self, head):
        if head[0] != 'ts':
            raise ValueError('ts must be the first header for this format')

        dim_sizes = []
        for col in range(1, len(head)):
            index_match = self.index_head_re.match(head[col])
            if not index_match:
                break
            dim_sizes += [int(index_match.groups()[0])]

        if not dim_sizes:
            raise ValueError('no sweep dimensions found in header row')

        param_names = []
        for param_col in range(col + 1, len(head)):
            param_names.append(head[param_col])
        if not param_names:
            raise ValueError('no param_names found in header row')

        self.init_data(param_names, dim_sizes)
