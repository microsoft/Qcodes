import time
from traceback import format_exc
from functools import update_wrapper
import weakref

from qcodes.utils.multiprocessing import ServerManager


def connect_instrument_server(server_name, instrument, server_extras={}):
    '''
    Find or make an instrument server process with the given name
    and add this instrument to it. An InstrumentServer can hold the
    connections to one or more instruments.

    server_name: which server to put the instrument on. If a server with
        this name exists, the instrument will be added to it. If not, a
        new server is created with this name

    instrument: the instrument object to attach to the server

    extras: any extra arguments to the InstrumentManager, will be set
        as attributes of any instrument that connects to it
    '''
    instances = InstrumentManager.instances
    if server_name in instances and instances[server_name]():
        # kwargs get ignored for existing servers - lets hope they're the same!
        manager = instances[server_name]()
    else:
        manager = InstrumentManager(server_name, server_extras)
        instances[server_name] = weakref.ref(manager)

    return manager.connect(instrument)


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

    def __init__(self, name, server_extras={}):
        self.name = name
        self.instruments = {}

        super().__init__(name=name, server_class=InstrumentServer,
                         server_extras=server_extras)

    def restart(self):
        '''
        Restart the InstrumentServer and reconnect the instruments that
        had been connected previously
        '''
        super().restart()

        for instrument in enumerate(self.instruments.values()):
            self.connect(instrument())

    def connect(self, instrument):
        conn = InstrumentConnection(manager=self, instrument=instrument)
        self.instruments[instrument.uuid] = weakref.ref(instrument)
        return conn

    def delete(self, instrument):
        try:
            self.write('delete', instrument.uuid)
        except:
            pass

        if instrument.uuid in self.instruments:
            del self.instruments[instrument.uuid]

            if not self.instruments:
                self.close()
                del self.instances[self.name]


class InstrumentConnection:
    '''
    A connection between one particular instrument and its server process

    This should be instantiated by connect_instrument_server, not directly
    '''
    def __init__(self, manager, instrument):
        self.manager = manager
        self.instrument = instrument

        self.manager.ask('new', instrument)

    def ask(self, func_name, *args, **kwargs):
        '''
        Query the server copy of this instrument, expecting a response
        '''
        return self.manager.ask('ask', self.instrument.uuid,
                                func_name, args, kwargs)

    def write(self, func_name, *args, **kwargs):
        '''
        Send a command to the server copy of this instrument, without
        waiting for a response
        '''
        self.manager.write('write', self.instrument.uuid,
                           func_name, args, kwargs)

    def close(self):
        '''
        Take this instrument off the server and irreversibly stop
        this connection
        '''
        if hasattr(self, 'manager'):
            self.manager.delete(self.instrument)


class ask_server:
    '''
    decorator for methods of an Instrument that should be executed
    on the relevant InstrumentServer process if one exists, that should
    wait for a response

    (if no response is needed, use write_server)
    '''

    doc_prefix = ('** Method <{}> decorated with @ask_server'
                  '** This method is decorated so that it will execute\n'
                  '** on the InstrumentServer process even if you call\n'
                  '** it from a different process.\n'
                  '** This variant (ask_server) will wait for a response.')

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

        doc = self.doc_prefix.format(func.__qualname__)
        if func.__doc__:
            doc = doc + '\n\n' + func.__doc__

        # make the docs and signature of the decorated method point back
        # to the original method
        update_wrapper(self, func)
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        '''
        When we wrap a method this way, it stops being bound and
        is just a function. This is the only place we see which object
        it should be bound to, so we grab and store it. Note this is not
        the way I've seen this done online, see eg:
        http://blog.dscpl.com.au/2014/01/how-you-implemented-your-python.html,
        uses functools.partial to actually change what you return from __get__,
        but functools.partial is not compatible with functools.update_wrapper
        so you can't get the docs right. This way self is a regular callable
        so update_wrapper works on it.
        '''
        self.instrument = obj
        return self

    def __call__(self, *args, **kwargs):
        if self.instrument.connection:
            return self.instrument.connection.ask(self.name, *args, **kwargs)
        else:
            return self.func(self.instrument, *args, **kwargs)


class write_server(ask_server):
    '''
    decorator for methods of an Instrument that should be executed
    on the relevant InstrumentServer process if one exists, that DO NOT
    get a response

    (if a response is needed, use ask_server)
    '''

    doc_prefix = ('** This method is decorated so that it will execute\n'
                  '** on the InstrumentServer process even if you call\n'
                  '** it from a different process.\n'
                  '** This variant (write_server) DOES NOT WAIT FOR OR \n'
                  '** ALLOW A RESPONSE FROM THE INSTRUMENT\n')

    def __call__(self, *args, **kwargs):
        if self.instrument.connection:
            self.instrument.connection.write(self.name, *args, **kwargs)
        else:
            self.func(self.instrument, *args, **kwargs)


class InstrumentServer:
    def __init__(self, query_queue, response_queue, error_queue, extras):
        self._query_queue = query_queue
        self._response_queue = response_queue
        self._error_queue = error_queue

        self.extras = extras

        self.instruments = {}
        self.running = True

        while self.running:
            try:
                query = None
                query = self._query_queue.get()
                self.process_query(query)
            except Exception as e:
                self.post_error(e, query)

    def process_query(self, query):
        getattr(self, query[0])(*(query[1:]))

    def reply(self, response):
        self._response_queue.put(response)

    def post_error(self, e, query=None):
        if query:
            e.args = e.args + ('error processing query ' + repr(query),)
        self._error_queue.put(format_exc())
        time.sleep(0.05)  # give the error queue has time to register not-empty
        self._response_queue.put('ERR')  # to short-circuit timeout

    def halt(self, *args, **kwargs):
        '''
        Quit this InstrumentServer
        '''
        self.running = False

    def new(self, instrument):
        self.instruments[instrument.uuid] = instrument
        instrument.server_extras = self.extras
        self.reply(True)

    def delete(self, instrument_id):
        if instrument_id in self.instruments:
            del self.instruments[instrument_id]

            if not self.instruments:
                self.halt()

    def ask(self, instrument_id, func_name, args, kwargs):
        func = getattr(self.instruments[instrument_id], func_name)
        response = func(*args, **kwargs)
        self.reply(response)

    def write(self, instrument_id, func_name, args, kwargs):
        func = getattr(self.instruments[instrument_id], func_name)
        func(*args, **kwargs)
