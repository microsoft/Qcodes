import time
from datetime import datetime
from traceback import format_exc

from .base import Instrument
from qcodes.utils.multiprocessing import ServerManager, SERVER_ERR


class MockInstrument(Instrument):
    '''
    Creates a software instrument, for modeling or testing

    name: (string) the name of this instrument
    delay: the time (in seconds) to wait after any operation
        to simulate communication delay
    model: a MockModel object to connect this MockInstrument to.
        Subclasses MUST accept `model` as a constructor kwarg ONLY, even
        though it is required. See notes in `Instrument` docstring.
        A model should have one or two methods related directly to this
        instrument:
        <name>_set(param, value) -> set a parameter on the model
        <name>_get(param) -> returns the value
    keep_history: record (in self.history) every command sent to this
        instrument (default True)
    use_async: use the async form of ask and write (default False)
    read_response: simple constant response to send to self.read(),
        just for testing
    server_name: leave default ('') to make a MockServer-#######
        with the number matching the model server id, or set None
        to not use a server.

    parameters to pass to model should be declared with:
        get_cmd = param_name + '?'
        set_cmd = param_name + ':{:.3f}' (specify the format & precision)
    these will get self.name + ':' prepended to fit the syntax expected
        by MockModel servers
    alternatively independent functions may still be provided.
    '''
    shared_kwargs = ['model']

    def __init__(self, name, delay=0, model=None, keep_history=True,
                 read_response=None, **kwargs):

        super().__init__(name, **kwargs)

        if not isinstance(delay, (int, float)) or delay < 0:
            raise TypeError('delay must be a non-negative number')
        self._delay = delay

        # try to access write and ask so we know they exist
        model.write
        model.ask
        self._model = model

        # keep a record of every command sent to this instrument
        # for debugging purposes
        if keep_history:
            self.keep_history = True
            self.history = []

        # just for test purposes
        self._read_response = read_response

    @classmethod
    def default_server_name(cls, **kwargs):
        model = kwargs.get('model', None)
        if model:
            return model.name.replace('Model', 'MockInsts')
        return 'MockInstruments'

    def write(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        try:
            parameter, value = cmd.split(':', 1)
        except ValueError:
            parameter, value = cmd, None  # for functions with no value

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'write', parameter, value))

        self._model.write(self.name + ':' + cmd)

    def ask(self, cmd):
        if self._delay:
            time.sleep(self._delay)

        parameter, blank = cmd.split('?')
        if blank:
            raise ValueError('text found after end of query')

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'ask', parameter))

        return self._model.ask(self.name + ':' + cmd)

    def read(self):
        if self._delay:
            time.sleep(self._delay)

        return self._read_response


class MockModel(ServerManager):  # pragma: no cover
    # this is purely in service of mock instruments which *are* tested
    # so coverage testing this (by running it locally) would be a waste.
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
    def __init__(self, name='Model-{:.7s}'):
        # Most of the other uses of ServerManager use a separate class
        # for the server itself. But here I put the two together into
        # a single class, just to make it easier to define models.
        # The downside is you have a local object with methods you
        # shouldn't call, the extras (<instrument>_(set|get)) should
        # only be called on the server copy. But I think that's OK because
        # this will primarily be called via the attached instruments.
        super().__init__(name, server_class=None)

    def get_attribute(self, name):
        ''' Get an attribute from the model (server side) '''        
        return self.ask('_magicget:%s' % (name, ))        

    def call_function(self, name, *args):
        ''' Call a function from the model (server side) '''        
        return self.ask('_magiccall:%s' % (name, ), *args)        
        
    def _run_server(self):
        while True:
            try:
                # make sure no matter what there is a query for error handling
                query = None
                query = self._query_queue.get()
                query_args = query[1:]
                query = query[0].split(':')

                instrument = query[0]

                if instrument == '_magicget':
                    name = query[1]
                    value = getattr(self, name)
                    self._response_queue.put(value)
                    continue
                if instrument == '_magiccall':
                    name = query[1]
                    func = getattr(self, name)
                    self._response_queue.put( func(*query_args) )
                    continue
                
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
