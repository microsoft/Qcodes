"""Mock instruments for testing purposes."""
import time
from datetime import datetime

from .base import Instrument
from qcodes.process.server import ServerManager, BaseServer
from qcodes.utils.nested_attrs import _NoDefault


class MockInstrument(Instrument):

    """
    Create a software instrument, mostly for testing purposes.

    Also works for simulatoins, but usually this will be simpler, easier to
    use, and faster if made as a single ``Instrument`` subclass.

    ``MockInstrument``s have extra overhead as they serialize all commands
    (to mimic a network communication channel) and use at least two processes
    (instrument server and model server) both of which must be involved in any
    given query.

    parameters to pass to model should be declared with:
        get_cmd = param_name + '?'
        set_cmd = param_name + ':{:.3f}' (specify the format & precision)
    alternatively independent set/get functions may still be provided.

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

    Attributes:
        shared_kwargs (List[str]): Class attribute, constructor kwargs to
            provide via server init. For MockInstrument this should always be
            ['model'] at least.

        keep_history (bool): Whether to record all commands and responses. Set
            on init, but may be changed at any time.

        history (List[tuple]): All commands and responses while keep_history is
            enabled, as tuples:
                (timestamp, 'ask' or 'write', param_name[, value])
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
        # for debugging purposes?
        self.keep_history = bool(keep_history)
        self.history = []

    @classmethod
    def default_server_name(cls, **kwargs):
        """
        Get the default server name for this instrument.

        Args:
            **kwargs: All the kwargs supplied in the constructor.

        Returns:
            str: Default MockInstrument server name is MockInsts-#######, where
                ####### is the first 7 characters of the MockModel's uuid.
        """
        model = kwargs.get('model', None)
        if model:
            return model.name.replace('Model', 'MockInsts')
        return 'MockInstruments'

    def get_idn(self):
        """Shim for IDN parameter."""
        return {
            'vendor': None,
            'model': type(self).__name__,
            'serial': self.name,
            'firmware': None
        }

    def write_raw(self, cmd):
        """
        Low-level interface to ``model.write``.

        Prepends self.name + ':' to the command, so the ``MockModel``
        will direct this query to its ``<name>_set`` method

        Args:
            cmd (str): The command to send to the instrument.
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

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.

        Raises:
            ValueError: If ``cmd`` is malformed in that it contains text
                after the '?'
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

    The model supports ``NestedAttrAccess`` calls ``getattr``, ``setattr``,
    ``callattr``, and ``delattr`` Because the manager and server are the same
    object, we override these methods with proxy methods after the server has
    been started.
    """

    def __init__(self, name='Model-{:.7s}'):
        super().__init__(name, server_class=None)

        # now that the server has started, we can remap attribute access
        # from the private methods (_getattr) to the public ones (getattr)
        # but the server copy will still have the NestedAttrAccess ones
        self.getattr = self._getattr
        self.setattr = self._setattr
        self.callattr = self._callattr
        self.delattr = self._delattr

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

        Returns:
            Union(str, None): The parameter value, if ``cmd`` has the form
                '<instrument>:<parameter>?', otherwise no return.

        Raises:
            ValueError: if cmd does not match one of the patterns above.
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

    def _getattr(self, attr, default=_NoDefault):
        """
        Get a (possibly nested) attribute of this model on its server.

        See NestedAttrAccess for details.
        """
        return self.ask('method_call', 'getattr', attr, default)

    def _setattr(self, attr, value):
        """
        Set a (possibly nested) attribute of this model on its server.

        See NestedAttrAccess for details.
        """
        self.ask('method_call', 'setattr', attr, value)

    def _callattr(self, attr, *args, **kwargs):
        """
        Call a (possibly nested) method of this model on its server.

        See NestedAttrAccess for details.
        """
        return self.ask('method_call', 'callattr', attr, *args, **kwargs)

    def _delattr(self, attr):
        """
        Delete a (possibly nested) attribute of this model on its server.

        See NestedAttrAccess for details.
        """
        self.ask('method_call', 'delattr', attr)
