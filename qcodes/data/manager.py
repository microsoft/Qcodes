from datetime import datetime, timedelta
import multiprocessing as mp
from queue import Empty
from traceback import format_exc
from sys import stderr

from qcodes.utils.helpers import PrintableProcess


def get_data_manager():
    '''
    create or retrieve the storage manager
    makes sure we don't accidentally create multiple DataManager processes
    '''
    dm = DataManager.default
    if dm and dm._server.is_alive():
        return dm
    return DataManager()


class NoData(object):
    def store(self, *args, **kwargs):
        raise RuntimeError('no DataSet to add to')

    def write(self, *args, **kwargs):
        pass


class DataManager(object):
    default = None
    '''
    creates a separate process (DataServer) that holds running measurement
    and monitor data, and manages writing these to disk or other storage

    DataServer communicates with other processes through messages
    Written using multiprocessing Queue's, but should be easily
    extensible to other messaging systems
    '''
    def __init__(self, query_timeout=2):
        DataManager.default = self

        self._query_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._error_queue = mp.Queue()

        # query_lock is only used with queries that get responses
        # to make sure the process that asked the question is the one
        # that gets the response.
        # Any query that does NOT expect a response can just dump it in
        # and more on.
        self.query_lock = mp.RLock()

        self.query_timeout = query_timeout
        self._start_server()

    def _start_server(self):
        self._server = DataServerProcess(target=self._run_server, daemon=True)
        self._server.start()

    def _run_server(self):
        DataServer(self._query_queue, self._response_queue, self._error_queue)

    def write(self, *query):
        self._query_queue.put(query)
        self._check_for_errors()

    def _check_for_errors(self):
        if not self._error_queue.empty():
            errstr = self._error_queue.get()
            errhead = '*** error on DataServer ***'
            print(errhead + '\n\n' + errstr, file=stderr)
            raise RuntimeError(errhead)

    def ask(self, *query, timeout=None):
        timeout = timeout or self.query_timeout

        with self.query_lock:
            self._query_queue.put(query)
            try:
                res = self._response_queue.get(timeout=timeout)
            except Empty as e:
                if self._error_queue.empty():
                    # only raise if we're not about to find a deeper error
                    raise e
            self._check_for_errors()

            return res

    def halt(self):
        if self._server.is_alive():
            self.ask('halt')
        self._server.join()

    def restart(self, force=False):
        if (not force) and self.ask('get_data', 'location'):
            raise RuntimeError('A measurement is running. Use '
                               'restart(force=True) to override.')
        self.halt()
        self._start_server()


class DataServerProcess(PrintableProcess):
    name = 'DataServer'


class DataServer(object):
    default_storage_period = 1  # seconds between data storage calls
    queries_per_store = 5
    default_monitor_period = 60  # seconds between monitoring storage calls

    def __init__(self, query_queue, response_queue, error_queue):
        self._query_queue = query_queue
        self._response_queue = response_queue
        self._error_queue = error_queue
        self._storage_period = self.default_storage_period
        self._monitor_period = self.default_monitor_period

        self._data = NoData()
        self._measuring = False

        self._run()

    def _run(self):
        self._running = True
        next_store_ts = datetime.now()
        next_monitor_ts = datetime.now()

        while self._running:
            read_timeout = self._storage_period / self.queries_per_store
            try:
                query = self._query_queue.get(timeout=read_timeout)
                getattr(self, 'handle_' + query[0])(*(query[1:]))
            except Empty:
                pass
            except Exception as e:
                self._post_error(e)

            try:
                now = datetime.now()

                if self._measuring and now > next_store_ts:
                    td = timedelta(seconds=self._storage_period)
                    next_store_ts = now + td
                    self._data.write()

                if now > next_monitor_ts:
                    td = timedelta(seconds=self._monitor_period)
                    next_monitor_ts = now + td
                    # TODO: update the monitor data storage

            except Exception as e:
                self._post_error(e)

    def _reply(self, response):
        self._response_queue.put(response)

    def _post_error(self, e):
        self._error_queue.put(format_exc())

    ######################################################################
    # query handlers                                                     #
    #                                                                    #
    # method: handle_<type>(self, arg1, arg2, ...)                       #
    # will capture queries ('<type>', arg1, arg2, ...)                   #
    #                                                                    #
    # All except store_data return something, so should be used with ask #
    # rather than write. That way they wait for the queue to flush and   #
    # will receive errors right anyway                                   #
    ######################################################################

    def handle_halt(self):
        self._running = False
        self._reply(True)

    def handle_new_data(self, data_set):
        if self._measuring:
            raise RuntimeError('Already executing a measurement')

        self._data = data_set
        self._data.init_on_server()
        self._measuring = True
        self._reply(True)

    def handle_end_data(self):
        self._data.write()
        self._measuring = False
        self._reply(True)

    def handle_store_data(self, *args):
        self._data.store(*args)

    def handle_get_measuring(self):
        self._reply(self._measuring)

    def handle_get_data(self, attr=None):
        self._reply(getattr(self._data, attr) if attr else self._data)
