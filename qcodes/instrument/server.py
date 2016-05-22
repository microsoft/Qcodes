from traceback import format_exc
import multiprocessing as mp
from queue import Empty

from qcodes.utils.multiprocessing import ServerManager, SERVER_ERR


def get_instrument_server(server_name, shared_kwargs={}):
    '''
    Find or make an instrument server process with the given name
    and shared attributes. An InstrumentServer can hold the connections
    to one or more instruments, but any unpicklable attributes need to be
    provided when the server is started and shared between all such
    instruments.

    server_name: which server to put the instrument on. If a server with
        this name exists, the instrument will be added to it. If not, a
        new server is created with this name. Default 'Instruments'

    shared_kwargs: unpicklable items needed by the instruments on the
        server, will get sent with the manager when it's started up
        and included in the kwargs to construct each new instrument
    '''

    if not server_name:
        server_name = 'Instruments'

    instances = InstrumentManager.instances
    manager = instances.get(server_name, None)

    if manager and manager._server in mp.active_children():
        if shared_kwargs and manager.shared_kwargs != shared_kwargs:
            # it's OK to add another instrument that has  *no* shared_kwargs
            # but if there are some and they're different from what's
            # already associated with this server, that's an error.
            raise ValueError(('An InstrumentServer with name "{}" already '
                              'exists but with different shared_attrs'
                              ).format(server_name))
    else:
        manager = InstrumentManager(server_name, shared_kwargs)

    return manager


class InstrumentManager(ServerManager):
    '''
    Creates and manages connections to an InstrumentServer

    name: the name of the server to create
    kwargs: extra items to send to the server on creation (such as
        additional queues, that can only be shared on creation)
        These items will be set as attributes of any instrument that
        connects to the server
    '''
    instances = {}

    def __init__(self, name, shared_kwargs=None):
        self.name = name
        self.shared_kwargs = shared_kwargs
        self.instances[name] = self

        self.instruments = {}

        super().__init__(name=name, server_class=InstrumentServer,
                         shared_attrs=shared_kwargs)

    def restart(self):
        '''
        Restart the InstrumentServer and reconnect the instruments that
        had been connected previously
        '''
        super().restart()

        instruments = self.instruments
        self.instruments = {}
        for connection_info in instruments.values():
            self.connect(**connection_info)

    def connect(self, remote_instrument, instrument_class, args, kwargs):
        new_id = self.ask('new_id')
        try:
            conn = InstrumentConnection(
                manager=self, instrument_class=instrument_class,
                new_id=new_id, args=args, kwargs=kwargs)

            # save the information to recreate this instrument on the server
            # in case of a restart
            self.instruments[conn.id] = dict(
                remote_instrument=remote_instrument,
                instrument_class=instrument_class,
                args=args,
                kwargs=kwargs)

            # attach the connection to the remote instrument here.
            # this is placed *here* rather than in the RemoteInstrument also to
            # facilitate restarting, so the RemoteInstrument itself doesn't
            # need to do anything on a restart
            remote_instrument.connection = conn
        except:
            # if anything went wrong adding a new instrument, delete it
            # in case it still exists there half-formed.
            self.delete(new_id)
            raise

    def delete(self, instrument_id):
        self.write('delete', instrument_id)

        if self.instruments.get(instrument_id, None):
            del self.instruments[instrument_id]

            if not self.instruments:
                self.close()
                self.instances.pop(self.name, None)


class InstrumentConnection:
    '''
    A connection between one particular instrument and its server process

    This should be instantiated by InstrumentManager.connect, not directly
    '''
    def __init__(self, manager, instrument_class, new_id, args, kwargs):
        self.manager = manager

        info = manager.ask('new', instrument_class, new_id, args, kwargs)
        for k, v in info.items():
            setattr(self, k, v)

    def ask(self, func_name, *args, **kwargs):
        '''
        Query the server copy of this instrument, expecting a response
        '''
        return self.manager.ask('ask', self.id, func_name, args, kwargs)

    def write(self, func_name, *args, **kwargs):
        '''
        Send a command to the server copy of this instrument, without
        waiting for a response
        '''
        self.manager.write('write', self.id, func_name, args, kwargs)

    def close(self):
        '''
        Take this instrument off the server and irreversibly stop
        this connection. You can only do this from the process that
        started the manager (ie the main process) so that other processes
        do not delete instruments when they finish
        '''
        if hasattr(self, 'manager'):
            if self.manager._server in mp.active_children():
                self.manager.delete(self.id)


class InstrumentServer:
    # just for testing - how long to allow it to wait on a queue.get
    timeout = None

    def __init__(self, query_queue, response_queue, error_queue,
                 shared_kwargs):
        self._query_queue = query_queue
        self._response_queue = response_queue
        self._error_queue = error_queue

        self.shared_kwargs = shared_kwargs

        self.instruments = {}
        self.next_id = 0
        self.running = True

        while self.running:
            try:
                query = None
                query = self._query_queue.get(timeout=self.timeout)
                self.process_query(query)
            except Empty:
                raise
            except Exception as e:
                self.post_error(e, query)

    def process_query(self, query):
        getattr(self, 'handle_' + query[0])(*(query[1:]))

    def reply(self, response):
        self._response_queue.put(response)

    def post_error(self, e, query=None):
        if query:
            e.args = e.args + ('error processing query ' + repr(query),)
        self._error_queue.put(format_exc())
        # the caller is waiting on _response_queue, so put a signal there
        # to say there's an error coming
        self._response_queue.put(SERVER_ERR)

    def handle_halt(self, *args, **kwargs):
        '''
        Quit this InstrumentServer
        '''
        self.running = False

    def handle_new_id(self):
        '''
        split out id generation from adding an instrument
        so that we can delete it if something goes wrong!
        '''
        new_id = self.next_id
        self.next_id += 1
        self.reply(new_id)

    def handle_new(self, instrument_class, new_id, args, kwargs):
        '''
        Add a new instrument to the server
        after the initial load, the instrument is referred to by its ID
        '''

        # merge shared_kwargs into kwargs for the constructor,
        # but only if this instrument_class is expecting them.
        # The *first* instrument put on a given server must have
        # all the shared_kwargs sent with it, but others may skip
        # (for now others must have *none* but later maybe they could
        # just skip some of them)
        for key, value in self.shared_kwargs.items():
            if key in instrument_class.shared_kwargs:
                kwargs[key] = value
        ins = instrument_class(*args, server_name=None, **kwargs)

        self.instruments[new_id] = ins

        # info to reconstruct the instrument API in the RemoteInstrument
        self.reply({
            'instrument_name': ins.name,
            'id': new_id,
            'parameters': {name: p.get_attrs()
                           for name, p in ins.parameters.items()},
            'functions': {name: f.get_attrs()
                          for name, f in ins.functions.items()},
            'methods': ins._get_method_attrs()
        })

    def handle_delete(self, instrument_id):
        '''
        Delete an instrument from the server, and stop the server if their
        are no more instruments left after this.
        '''
        if instrument_id in self.instruments:
            self.instruments[instrument_id].close()

            del self.instruments[instrument_id]

            if not any(self.instruments):
                self.handle_halt()

    def handle_ask(self, instrument_id, func_name, args, kwargs):
        '''
        Run some method of an instrument, and post the return to the
        response queue
        '''
        func = getattr(self.instruments[instrument_id], func_name)
        response = func(*args, **kwargs)
        self.reply(response)

    def handle_write(self, instrument_id, func_name, args, kwargs):
        '''
        Run some method of an instrument but ignore any response it may give
        (errors will still go to the error queue, but will be picked up by
        some later query)
        '''
        func = getattr(self.instruments[instrument_id], func_name)
        func(*args, **kwargs)
