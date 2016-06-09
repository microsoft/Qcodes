"""Mock instruments for testing purposes."""
import time
from datetime import datetime

from .base import Instrument
from qcodes.process.server import ServerManager, BaseServer


class MockInstrument(Instrument):

    """
    Create a software instrument, mostly for testing purposes.

    Also works for simulatoins, but usually this will be simpler, easier to
    use, and faster if made as a single ``Instrument`` subclass.

    ``MockInstrument``s have extra overhead as they serialize all commands
    (to mimic a network communication channel) and use at least two processes
    (instrument server and model server) both of which must be involved in any
    given query.

    Args:
        name (str): The name of this instrument.

        delay (number): Time (in seconds) to wait after any operation
            to simulate communication delay. Default 0.

        model (MockModel): A model to connect to. Subclasses MUST accept
            ``model`` as a constructor kwarg ONLY, even though it is required.
            See notes in ``Instrument`` docstring.
            The model should have one or two methods related directly to this
            instrument by ``name``:
            ``<name>_set(param, value)``: set a parameter on the model
            ``<name>_get(param)``: returns the value of a parameter

        keep_history (bool): Whether to record (in self.history) every command
            sent to this instrument. Default True.

        server_name (Union[str, None]): leave default ('') to make a
            MockInsts-####### server with the number matching the model server
            id, or set None to not use a server.

    parameters to pass to model should be declared with:
        get_cmd = param_name + '?'
        set_cmd = param_name + ':{:.3f}' (specify the format & precision)
    alternatively independent set/get functions may still be provided.
    """

    shared_kwargs = ['model']

    def __init__(self, name, delay=0, model=None, keep_history=True, **kwargs):
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

    @classmethod
    def default_server_name(cls, **kwargs):
        """
        Default MockInstrument server name is MockInsts-#######.

        ####### is the first 7 characters of the MockModel's uuid.
        """
        model = kwargs.get('model', None)
        if model:
            return model.name.replace('Model', 'MockInsts')
        return 'MockInstruments'

    def write_raw(self, cmd):
        """
        Low-level interface to ``model.write``.

        Prepends self.name + ':' to the command, so the ``MockModel``
        will direct this query to its ``<name>_set`` method
        """
        if self._delay:
            time.sleep(self._delay)

        try:
            parameter, value = cmd.split(':', 1)
        except ValueError:
            parameter, value = cmd, None  # for functions with no value

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'write', parameter, value))

        self._model.write('cmd', self.name + ':' + cmd)

    def ask_raw(self, cmd):
        """
        Low-level interface to ``model.ask``.

        Prepends self.name + ':' to the command, so the ``MockModel``
        will direct this query to its ``<name>_get`` method
        """
        if self._delay:
            time.sleep(self._delay)

        parameter, blank = cmd.split('?')
        if blank:
            raise ValueError('text found after end of query')

        if self.keep_history:
            self.history.append((datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 'ask', parameter))

        return self._model.ask('cmd', self.name + ':' + cmd)


# MockModel is purely in service of mock instruments which *are* tested
# so coverage testing this (by running it locally) would be a waste.
class MockModel(ServerManager, BaseServer):  # pragma: no cover

    """
    Base class for models to connect to various MockInstruments.

    Creates a separate process that holds the model state, so that
    any process can interact with the model and get the same state.

    Args:
        name (str): The server name to create for the model.
            Default 'Model-{:.7s}' uses the first 7 characters of
            the server's uuid.

    for every instrument that connects to this model, create two methods:
        ``<instrument>_set(param, value)``: set a parameter on the model
        ``<instrument>_get(param)``: returns the value of a parameter
    ``param`` and the set/return values should all be strings

    If ``param`` and/or ``value`` is not recognized, the method should raise
    an error.

    Other uses of ServerManager use separate classes for the server and its
    manager, but here I put the two together into a single class, to make it
    easier to define models. The downside is you have a local object with
    methods you shouldn't call: the extras (<instrument>_(set|get)) should
    only be called on the server copy. Normally this should only be called via
    the attached instruments anyway.
    """

    def __init__(self, name='Model-{:.7s}'):
        super().__init__(name, server_class=None)

    def _run_server(self):
        self.run_event_loop()

    def handle_cmd(self, cmd):
        """
        Handler for all model queries.

        Args:
            cmd (str): Can take several forms:
                '<instrument>:<parameter>?':
                    calls ``self.<instrument>_get(<parameter>)`` and forwards
                    the return value.
                '<instrument>:<parameter>:<value>':
                    calls ``self.<instrument>_set(<parameter>, <value>)``
                '<instrument>:<parameter>'.
                    calls ``self.<instrument>_set(<parameter>, None)``
        """
        query = cmd.split(':')

        instrument = query[0]
        param = query[1]

        if param[-1] == '?' and len(query) == 2:
            return getattr(self, instrument + '_get')(param[:-1])

        elif len(query) <= 3:
            value = query[2] if len(query) == 3 else None
            getattr(self, instrument + '_set')(param, value)

        else:
            raise ValueError()
