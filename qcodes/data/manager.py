from datetime import datetime, timedelta
from queue import Empty
from traceback import format_exc
import logging

from qcodes.process.server import ServerManager, BaseServer


def get_data_manager(only_existing=False):
    """
    create or retrieve the storage manager
    makes sure we don't accidentally create multiple DataManager processes
    """
    dm = DataManager.default
    if dm and dm._server.is_alive():
        return dm
    elif only_existing:
        return None
    return DataManager()


class NoData:
    """
    A placeholder object for DataServer to hold
    when there is no loop running.
    """
    location = None

    def store(self, *args, **kwargs):
        raise RuntimeError('no DataSet to add to')

    def write(self, *args, **kwargs):
        pass


class DataManager(ServerManager):
    default = None
    """
    creates a separate process (DataServer) that holds running measurement
    and monitor data, and manages writing these to disk or other storage

    DataServer communicates with other processes through messages
    Written using multiprocessing Queue's, but should be easily
    extensible to other messaging systems
    """
    def __init__(self):
        type(self).default = self
        super().__init__(name='DataServer', server_class=DataServer)

    def restart(self, force=False):
        """
        Restart the DataServer
        Use force=True to abort a running measurement.
        """
        if (not force) and self.ask('get_data', 'location'):
            raise RuntimeError('A measurement is running. Use '
                               'restart(force=True) to override.')
        super().restart()


class DataServer(BaseServer):
    """
    Running in its own process, receives, holds, and returns current `Loop` and
    monitor data, and writes it to disk (or other storage)

    When a `Loop` is *not* running, the DataServer also calls the monitor
    routine. But when a `Loop` *is* running, *it* calls the monitor so that it
    can avoid conflicts. Also while a `Loop` is running, there are
    complementary `DataSet` objects in the loop and `DataServer` processes -
    they are nearly identical objects, but are configured differently so that
    the loop `DataSet` doesn't hold any data itself, it only passes that data
    on to the `DataServer`
    """
    default_storage_period = 1  # seconds between data storage calls
    queries_per_store = 5
    default_monitor_period = 60  # seconds between monitoring storage calls

    def __init__(self, query_queue, response_queue, extras=None):
        super().__init__(query_queue, response_queue, extras)

        self._storage_period = self.default_storage_period
        self._monitor_period = self.default_monitor_period

        self._data = NoData()
        self._measuring = False

        self.run_event_loop()

    def run_event_loop(self):
        self.running = True
        next_store_ts = datetime.now()
        next_monitor_ts = datetime.now()

        while self.running:
            read_timeout = self._storage_period / self.queries_per_store
            try:
                query = self._query_queue.get(timeout=read_timeout)
                self.process_query(query)
            except Empty:
                pass

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

            except:
                logging.error(format_exc())

    ######################################################################
    # query handlers                                                     #
    ######################################################################

    def handle_new_data(self, data_set):
        """
        Load a new (normally empty) DataSet into the DataServer, and
        prepare it to start receiving and storing data
        """
        if self._measuring:
            raise RuntimeError('Already executing a measurement')

        self._data = data_set
        self._data.init_on_server()
        self._measuring = True

    def handle_finalize_data(self):
        """
        Mark this DataSet as complete and write its final changes to storage
        """
        self._data.finalize()
        self._measuring = False

    def handle_store_data(self, *args):
        """
        Put some data into the DataSet
        """
        self._data.store(*args)

    def handle_get_measuring(self):
        """
        Is a measurement loop presently running?
        """
        return self._measuring

    def handle_get_data(self, attr=None):
        """
        Return the active DataSet or some attribute of it
        """
        return getattr(self._data, attr) if attr else self._data

    def handle_get_changes(self, synced_indices):
        """
        Return all new data after the last sync
        """
        return self._data.get_changes(synced_indices)
