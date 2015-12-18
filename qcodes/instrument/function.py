import asyncio

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import syncable_command
from qcodes.utils.validators import Validator


class Function(Metadatable):
    def __init__(self, instrument, name, call_cmd=None, async_call_cmd=None,
                 parameters=[], parse_function=None, **kwargs):
        '''
        defines a function (with arbitrary parameters) that this instrument
        can execute.

        You execute this function object like a normal function, or use its
        .call method; or call it async with the .call_async method.

        instrument: an instrument that handles this function
        name: the local name of this parameter
        call_cmd: command to execute on instrument
            - a string (with positional fields to .format, "{}" or "{0}" etc)
            - a function (with parameter count matching parameters list)
        async_call_cmd: an async function to use for call_async, or for both
            sync and async if call_cmd is missing or None.
        parameters: list of Validator objects,
            one for each parameter to the Function
        parse_function: function to parse the return value of cmd,
            may be a type casting function like int or float.
            If None, will not wait for or read any response
        '''
        super().__init__(**kwargs)

        self._instrument = instrument
        self.name = name

        self._set_params(parameters)
        self._set_call(call_cmd, async_call_cmd, parse_function)

    def _set_params(self, parameters):
        for param in parameters:
            if not isinstance(param, Validator):
                raise TypeError('all parameters must be Validator objects')
        self._parameters = parameters
        self._param_count = len(parameters)

    def _set_call(self, call_cmd, async_call_cmd, parse_function):
        ask_or_write = self._instrument.write
        ask_or_write_async = self._instrument.write_async
        if isinstance(call_cmd, str) and parse_function:
            ask_or_write = self._instrument.ask
            ask_or_write_async = self._instrument.ask_async

        self._call, self._call_async = syncable_command(
            self._param_count, call_cmd, async_call_cmd,
            ask_or_write, ask_or_write_async, parse_function)

    def validate(self, args):
        '''
        check that all arguments to this Function are allowed
        '''
        if len(args) != self._param_count:
            raise TypeError(
                '{} called with {} parameters but requires {}'.format(
                    self.name, len(args), self._param_count))

        for i in range(self._param_count):
            value = args[i]
            param = self._parameters[i]
            if not param.is_valid(value):
                raise ValueError(
                    '{} is not a valid value for parameter {} of {}'.format(
                        value, i, self.name))

    def __call__(self, *args):
        self.validate(args)
        return self._call(*args)

    def call(self, *args):
        self.validate(args)
        return self._call(*args)

    @asyncio.coroutine
    def call_async(self, *args):
        self.validate(args)
        return (yield from self._call_async(*args))
