import asyncio

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import syncable_command
from qcodes.utils.validators import Validator


class Function(Metadatable):
    '''
    defines a function (with arbitrary parameters) that this instrument
    can execute.

    You execute this function object like a normal function, or use its
    .call method; or call it async with the .call_async method.

    name: the local name of this parameter
    instrument: an instrument that handles this function
        default None

    call_cmd: command to execute on instrument
        - a string (with positional fields to .format, "{}" or "{0}" etc)
          you can only use a string if an instrument is provided,
          this string will be passed to instrument.write
        - a function (with parameter count matching parameters list)
    async_call_cmd: an async function to use for call_async, or for both
        sync and async if call_cmd is missing or None.

    parameters: list of Validator objects,
        one for each parameter to the Function

    parameter_parser: function to transform the input parameter(s)
        to encoded value(s) sent to the instrument.
        If there are multiple arguments, this function should accept all
        the arguments in order, and return a tuple of values.
    return_parser: function to transform the response from the instrument
        to the final output value.
        may be a type casting function like `int` or `float`.
        If None (default), will not wait for or read any response
    NOTE: parsers only apply if call_cmd is a string. The function forms
        of call_cmd and async_call_cmd should do their own parsing.

    parse_function: DEPRECATED - use return_parser instead
    '''
    def __init__(self, name, instrument=None,
                 call_cmd=None, async_call_cmd=None,
                 parameters=[], parameter_parser=None, return_parser=None,
                 parse_function=None, **kwargs):
        super().__init__(**kwargs)

        self._instrument = instrument
        self.name = name

        # push deprecated parse_function argument to get_parser
        if return_parser is None:
            return_parser = parse_function

        self._set_params(parameters)
        self._set_call(call_cmd, async_call_cmd,
                       parameter_parser, return_parser)

    def _set_params(self, parameters):
        for param in parameters:
            if not isinstance(param, Validator):
                raise TypeError('all parameters must be Validator objects')
        self._parameters = parameters
        self._param_count = len(parameters)

    def _set_call(self, call_cmd, async_call_cmd,
                  parameter_parser, return_parser):
        if self._instrument:
            ask_or_write = self._instrument.write
            ask_or_write_async = self._instrument.write_async
            if isinstance(call_cmd, str) and return_parser:
                ask_or_write = self._instrument.ask
                ask_or_write_async = self._instrument.ask_async
        else:
            ask_or_write, ask_or_write_async = None, None

        self._call, self._call_async = syncable_command(
            param_count=self._param_count,
            cmd=call_cmd, acmd=async_call_cmd,
            exec_str=ask_or_write, aexec_str=ask_or_write_async,
            input_parser=parameter_parser, output_parser=return_parser)

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
