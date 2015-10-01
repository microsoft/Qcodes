import multiprocessing as mp
from queue import Empty as EmptyQueue
from traceback import format_exc
from datetime import datetime, timedelta

from qcodes.sweep_storage import NoSweep, SweepStorage, MergedCSVStorage


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
                    # TODO: monitor too? with its own timer?
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
            # the whole SweepStorage should be pickable, right?
            self._reply(self._sweep)
        else:
            self._reply(False)

    def handle_get_sweep_data(self):
        self._reply(self._sweep.get_data())
