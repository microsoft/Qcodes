import asyncio

from qcodes.utils.metadata import Metadatable
from qcodes.utils.sync_async import syncable_command
from qcodes.utils.validators import Validator, validate_all


class Function(Metadatable):
    '''
    defines a function (with arbitrary args) that this instrument
    can execute.

    This functionality is meant for simple cases, principally things that
    map to simple commands like '*RST' (reset) or those with just a few
    arguments. It requires a fixed argument count, and positional args
    only. If your case is more complicated, you're probably better off
    simply making a new method in your Instrument subclass definition.
    The function validators.validate_all can help reduce boilerplate code
    in this case.

    You execute this function object like a normal function, or use its
    .call method; or call it async with the .call_async method.

    name: the local name of this function
    instrument: an instrument that handles this function
        default None

    call_cmd: command to execute on instrument
        - a string (with positional fields to .format, "{}" or "{0}" etc)
          you can only use a string if an instrument is provided,
          this string will be passed to instrument.write
        - a function (with arg count matching args list)
    async_call_cmd: an async function to use for call_async, or for both
        sync and async if call_cmd is missing or None.

    args: list of Validator objects,
        one for each arg to the Function

    arg_parser: function to transform the input arg(s)
        to encoded value(s) sent to the instrument.
        If there are multiple arguments, this function should accept all
        the arguments in order, and return a tuple of values.
    return_parser: function to transform the response from the instrument
        to the final output value.
        may be a type casting function like `int` or `float`.
        If None (default), will not wait for or read any response
    NOTE: parsers only apply if call_cmd is a string. The function forms
        of call_cmd and async_call_cmd should do their own parsing.
    '''
    def __init__(self, name, instrument=None,
                 call_cmd=None, async_call_cmd=None,
                 args=[], arg_parser=None, return_parser=None,
                 **kwargs):
        super().__init__(**kwargs)

        self._instrument = instrument
        self.name = name

        self._set_args(args)
        self._set_call(call_cmd, async_call_cmd,
                       arg_parser, return_parser)

    def _set_args(self, args):
        for arg in args:
            if not isinstance(arg, Validator):
                raise TypeError('all args must be Validator objects')
        self._args = args
        self._arg_count = len(args)

    def _set_call(self, call_cmd, async_call_cmd,
                  arg_parser, return_parser):
        if self._instrument:
            ask_or_write = self._instrument.write
            ask_or_write_async = self._instrument.write_async
            if isinstance(call_cmd, str) and return_parser:
                ask_or_write = self._instrument.ask
                ask_or_write_async = self._instrument.ask_async
        else:
            ask_or_write, ask_or_write_async = None, None

        self._call, self._call_async = syncable_command(
            param_count=self._arg_count,
            cmd=call_cmd, acmd=async_call_cmd,
            exec_str=ask_or_write, aexec_str=ask_or_write_async,
            input_parser=arg_parser, output_parser=return_parser)

    def validate(self, args):
        '''
        check that all arguments to this Function are allowed
        '''
        if self._instrument:
            func_name = (getattr(self._instrument, 'name', '') or
                         str(self._instrument.__class__)) + '.' + self.name
        else:
            func_name = self.name

        if len(args) != self._arg_count:
            raise TypeError(
                '{} called with {} args but requires {}'.format(
                    func_name, len(args), self._arg_count))

        validate_all(*zip(self._args, args), context='Function: ' + func_name)

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
