import asyncio
import time
from datetime import datetime
from traceback import format_exc

from .base import Instrument
from .server import ask_server, write_server
from qcodes.utils.multiprocessing import ServerManager, SERVER_ERR


class MockInstrument(Instrument):
    '''
    Creates a software instrument, for modeling or testing
    inputs:
        name: (string) the name of this instrument
        delay: the time (in seconds) to wait after any operation
            to simulate communication delay
        model: an object with write and ask methods, taking 2 or 3 args:
            instrument: the name of the instrument
            parameter: the name of the parameter
            value (write only): the value to write, as a string

    parameters to pass to model should be declared with:
        get_cmd = param_name + '?'
        set_cmd = param_name + ':{:.3f}' (specify the format & precision)
    these will get self.name + ':' prepended to fit the syntax expected
        by MockModel servers
    alternatively independent functions may still be provided.
    '''
    def __init__(self, name, delay=0, model=None, keep_history=True,
                 use_async=False, read_response=None,
                 server_name='', **kwargs):

        if not isinstance(delay, (int, float)) or delay < 0:
            raise TypeError('delay must be a non-negative number')
        self._delay = delay

        # try to access write and ask so we know they exist
        model.write
        model.ask
        self.model_id = model.uuid

        # can't pass the model itself through the queue to the server,
        # so send it to the server on creation and have the server
        # attach it to each instrument.
        server_extras = {model.uuid: model}

        # keep a record of every command sent to this instrument
        # for debugging purposes
        if keep_history:
            self.keep_history = True
            self.history = []

        # do we want a sync or async model?
        if use_async:
            self.write_async = self._write_async
            self.read_async = self._read_async
            self.ask_async = self._ask_async
        else:
            self.write = self._write
            self.read = self._read
            self.ask = self._ask

        # just for test purposes
        self._read_response = read_response

        # to make sure we make one server per model,
        # we give the server a (pretty much) unique name
        if server_name == '':
            server_name = model.name.replace('Model', 'MockServer')
        super().__init__(name, server_name, server_extras, **kwargs)

    @ask_server
    def on_connect(self):
        '''
        to get around the fact that you can't send a model to an
        already-existing server process, we send the model with
        server creation, then find it on the other end by its uuid
        '''
        self._model = self.server_extras[self.model_id]

    @write_server
    def _write_inner(self, cmd):
        try:
            parameter, value = cmd.split(':', 1)
        except ValueError:
            parameter, value = cmd, None  # for functions with no value

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'write', parameter, value))

        self._model.write(self.name + ':' + cmd)

    def _write(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        self._write_inner(cmd)

    @asyncio.coroutine
    def _write_async(self, cmd):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        self._write_inner(cmd)

    @ask_server
    def _ask_inner(self, cmd):
        parameter, blank = cmd.split('?')
        if blank:
            raise ValueError('text found after end of query')

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'ask', parameter))

        return self._model.ask(self.name + ':' + cmd)

    def _read(self):
        if self._delay:
            time.sleep(self._delay)

        return self._read_response

    @asyncio.coroutine
    def _read_async(self):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        return self._read_response

    def _ask(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        return self._ask_inner(cmd)

    @asyncio.coroutine
    def _ask_async(self, cmd):
        if self._delay:
            yield from asyncio.sleep(self._delay)

        return self._ask_inner(cmd)


class MockModel(ServerManager):
    '''
    Base class for models to connect to various MockInstruments

    Creates a separate process that holds the model state, so that
    any process can interact with the model and get the same state.

    write and ask support single string queries of the form:
        <instrument>:<parameter>:<value> (for setting)
        <instrument>:<parameter>? (for getting)

    for every instrument the model understands, create two methods:
        <instrument>_set(param, value)
        <instrument>_get(param) -> returns the value
    both param and the set/return values should be strings

    If anything is not recognized, raise an error, and the query will be
    added to it
    '''
    def __init__(self, name='Model{:.7s}'):
        # Most of the other uses of ServerManager use a separate class
        # for the server itself. But here I put the two together into
        # a single class, just to make it easier to define models.
        # The downside is you have a local object with methods you
        # shouldn't call, the extras (<instrument>_(set|get)) should
        # only be called on the server copy. But I think that's OK because
        # this will primarily be called via the attached instruments.
        super().__init__(name, server_class=None)

    def _run_server(self):
        while True:
            try:
                # make sure no matter what there is a query for error handling
                query = None
                query = self._query_queue.get()
                query = query[0].split(':')

                instrument = query[0]

                if instrument == 'halt':
                    self._response_queue.put(True)
                    break

                param = query[1]
                if param[-1] == '?' and len(query) == 2:
                    getter = getattr(self, instrument + '_get')
                    self._response_queue.put(getter(param[:-1]))
                elif len(query) <= 3:
                    value = query[2] if len(query) == 3 else None
                    setter = getattr(self, instrument + '_set')
                    setter(param, value)
                else:
                    raise ValueError

            except Exception as e:
                e.args = e.args + ('error processing query ' + repr(query),)
                self._error_queue.put(format_exc())
                self._response_queue.put(SERVER_ERR)
