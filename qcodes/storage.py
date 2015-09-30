import math
import numpy as np
import multiprocessing as mp
from queue import Empty as EmptyQueue
from traceback import format_exc
from datetime import datetime, timedelta
import time
import csv

from qcodes.utils.helpers import wait_secs


def get_storage_manager():
    '''
    create or retrieve the storage manager
    makes sure we don't accidentally create multiple StorageManager processes
    '''
    sm = StorageManager.default
    if sm and sm._server.is_alive():
        return sm
    return StorageManager()


class StorageManager(object):
    default = None
    '''
    creates a separate process (StorageServer) that holds running sweeps
    and monitor data, and manages writing these to disk or other storage

    To talk to the server, get a client with StorageManager.new_client()

    StorageServer communicates with other processes through messages
    I'll write this using multiprocessing Queue's, but should be easily
    extensible to other messaging systems
    '''
    def __init__(self, query_timeout=5, storage_class=MergedCSVStorage):
        StorageManager.default = self

        self._query_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._error_queue = mp.Queue()
        self._storage_class = storage_class

        # lock is only used with queries that get responses
        # to make sure the process that asked the question is the one
        # that gets the response.
        self._query_lock = mp.RLock()

        self.query_timeout = query_timeout

        self._server = mp.Process(target=self._run_server, daemon=True)
        self._server.start()

    @property
    def storage_class(self):
        return self._storage_class

    def _run_server(self):
        StorageServer(self._query_queue, self._response_queue,
                      self._error_queue, self._storage_class)

    def new_client(self, timeout=None):
        return StorageClient(self, timeout)

    def write(self, *query):
        self._query_queue.put(query)
        self.check_for_errors()

    def check_for_errors(self):
        if not self._error_queue.empty():
            raise self._error_queue.get()

    def ask(self, *query, timeout=None):
        timeout = timeout or self.query_timeout

        with self._query_lock:
            self._query_queue.put(query)
            try:
                res = self._response_queue.get(timeout=timeout)
            except EmptyQueue as e:
                if self._error_queue.empty():
                    # only raise if we're not about to find a deeper error
                    # I do it this way rather than just checking for errors
                    # now) because ipython obfuscates the real error
                    # by noting that it occurred while processing this one
                    raise e
            self.check_for_errors()

            return res

    def halt(self):
        if self._server.is_alive():
            self.write('halt')
        self._server.join()

    def set(self, key, value):
        self.write('set', key, value)

    def get(self, key, timeout=None):
        return self.ask('get', key, timeout=timeout)


class StorageServer(object):
    default_period = 1  # seconds between data storage calls
    queries_per_store = 5

    def __init__(self, query_queue, response_queue, error_queue,
                 storage_class):
        self._query_queue = query_queue
        self._response_queue = response_queue
        self._error_queue = error_queue
        self._storage_class = storage_class
        self._period = self.default_period

        self._dict = {}  # flexible storage, for testing purposes

        self._sweep = NoSweep()
        self._sweeping = False
        self._sweep_queue = mp.Queue()

        self._run()

    def _run(self):
        self._running = True
        self._next_store_ts = datetime.now()

        while self._running:
            query_timeout = self._period / self.queries_per_store
            try:
                query = self._query_queue.get(timeout=query_timeout)
                getattr(self, 'handle_' + query[0])(*(query[1:]))
            except EmptyQueue:
                pass
            except Exception as e:
                self._post_error(e)

            try:
                if datetime.now() > self._next_store_ts:
                    self._next_store_ts = (datetime.now() +
                                           timedelta(seconds=self._period))
                    self._sweep.update_storage()
                    # TODO: monitor too? maybe with its own timer?
            except Exception as e:
                self._post_error(e)

    def _reply(self, response):
        self._response_queue.put(response)

    def _post_error(self, e):
        e.args = e.args + (format_exc(), )
        self._error_queue.put(e)

    ######################################################################
    # query handlers                                                     #
    #                                                                    #
    # method: handle_<type>(self, arg1, arg2, ...)                       #
    # will capture queries ('<type>', arg1, arg2, ...)                   #
    ######################################################################

    def handle_set(self, key, value):
        self._dict[key] = value

    def handle_get(self, key):
        self._reply(self._dict[key])

    def handle_halt(self):
        self._running = False

    def handle_new_sweep(self, location, param_names, dim_sizes):
        if self._sweeping:
            raise RuntimeError('Already executing a sweep')
        self._sweep = SweepStorage(location, param_names, dim_sizes, None)
        self._sweeping = True
        self._reply(self._sweep.location)

    def handle_end_sweep(self):
        self._sweep.update_storage()
        self._sweeping = False

    def handle_sweep_data(self, indices, values):
        self._sweep.set_data_point(tuple(indices), values)

    def handle_sweeping(self):
        self._reply(self._sweeping)

    def handle_get_live_sweep(self):
        if self._sweeping:
            self._reply(self._sweep.location)
        else:
            self._reply(False)

    def handle_get_sweep_data(self):
        self._reply(self._sweep.get_data())

    def handle_get_live_sweep(self):
        # the whole SweepStorage should be pickable, right?
        self._reply(self._sweep)


class NoSweep(object):
    def data_point(self, *args, **kwargs):
        raise RuntimeError('no sweep to add data to')

    def update_storage(self, *args, **kwargs):
        pass


class SweepStorage(object):
    '''
    base class for holding and storing sweep data
    subclasses should override update_storage

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
        storage_manager: if we're given a StorageManager here, check if this
            is the live sweep, then allow syncing data from there
    '''
    def __init__(self, location, param_names=None, dim_sizes=None,
                 storage_manager=None):
        self.location = location  # TODO: auto location?

        if param_names and dim_sizes:
            self.init_data(param_names, dim_sizes)
            self.last_saved_index = -1
        elif param_names is None and dim_sizes is None:
            # omitted names and dim_sizes? assume we're reading from
            # a saved file instead
            self.read()
        else:
            raise TypeError('you must provide either both or neither of '
                            'param_names and dim_sizes')

        self.new_indices = set()
        if storage_manager:
            self._storage_manager = storage_manager
        self.check_live_sweep()
        self.sync_live()

    def check_live_sweep(self):
        if self._storage_manager:
            live_sweep = storage_manager.ask('get_live_sweep')
            self._is_live_sweep = (self.location == live_sweep)
        else:
            self._is_live_sweep = False

    def sync_live(self):
        if not self._is_live_sweep:
            # assume that once we determine it's not the live sweep,
            # it will never be the live sweep again
            return
        with self._storage_manager._query_lock:
            self.check_live_sweep()
            if not self._is_live_sweep:
                return
            self.data = self._storage_manager.ask('get_sweep_data')

    def init_data(self, param_names, dim_sizes):
        self.param_names = tuple(param_names)
        self.dim_sizes = tuple(dim_sizes)
        self.data = {}
        for pn in self.param_names + ('ts',):
            arr = np.ndarray(self.dim_sizes)
            arr.fill(math.nan)
            self.data[pn] = arr

    def set_data_point(self, indices, values):
        for pn, val in zip(self.param_names, values):
            self.data[pn][indices] = val

        self.data['ts'] = time.time()

        flat_index = np.ravel_multi_index(tuple(zip(indices)),
                                          self.dim_sizes)[0]
        new_indices.add(flat_index)

    def get_data(self):
        '''
        return the entire data set as it stands right now
        later, if performance dictates, we can reply with a subset,
        just the things that have changed (based on what?)
        '''
        return self.data

    def update_storage(self):
        raise NotImplementedError('you must subclass SweepStorage '
                                  'and define update_storage and read')

    def read(self):
        raise NotImplementedError('you must subclass SweepStorage '
                                  'and define update_storage and read')


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
        if not self.new_indices:
            return

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

        self.new_indices = set()
        self.last_saved_index = max(self.last_saved_index, last_new_index)

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
            raise ValueError('unrecognized header {}'.format(head))

        dim_sizes = []
        for col in range(1, len(lead)):
            index_match = index_head_re.match(head[col])
            if not index_match:
                break
            dim_sizes += [int(index_match.groups()[0])]

        if not dim_sizes:
            raise ValueError('no sweep dimensions found in header row')

        param_names = []
        for col in range(col + 1, len(head)):
            param_names.append(head[col])
        if not param_names:
            raise ValueError('no param_names found in header row')

        self.init_data(param_names, dim_sizes)
