import asyncio

from .helpers import is_function


def wait_for_async(f, *args, **kwargs):
    '''
    make an coroutine f block until it's totally finished
    effectively turning it into a synchronous function
    '''
    parent_loop = loop = asyncio.get_event_loop()

    # if we're already running an event loop, we need to make
    # a new one to run this coroutine, because the outer one
    # is already blocked
    #
    # this can happen if an async routine calls a sync routine
    # which calls another async routine. We should try to avoid this
    # but as long as there is a mixture of sync and async drivers
    # it'll still happen from time to time.
    nested = loop.is_running()
    if nested:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    out = loop.run_until_complete(f(*args, **kwargs))

    if nested:
        asyncio.set_event_loop(parent_loop)

    return out


class mock_sync(object):
    '''
    make a coroutine into a synchronous function
    written as a callable object rather than a closure
    so it's picklable on Windows
    '''
    def __init__(self, f):
        self._f = f

    def __call__(self, *args, **kwargs):
        return wait_for_async(self._f, *args, **kwargs)


class mock_async(object):
    '''
    make a synchronous function f awaitable
    written as a callable object rather than a closure
    so it's picklable on Windows
    '''

    # feels like a hack, but _is_coroutine is what
    # asyncio.iscoroutinefunction looks for, and it won't find it
    # in the decorator for __call__. Note that there's also
    # inspect.iscoroutinefunction and inspect.isfunction but
    # these do not treat objects with __call__ as functions at all
    # so will always fail on mock_sync and mock_async written as
    # callable objects
    _is_coroutine = True

    def __init__(self, f):
        self._f = f

    @asyncio.coroutine
    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)


class NoCommandError(Exception):
    pass


def syncable_command(param_count, cmd=None, acmd=None,
                     exec_str=None, aexec_str=None, parse_function=None,
                     no_cmd_function=None):
    '''
    create synchronous and asynchronous versions of a command
    inputs:
        param_count: the number of parameters to the command
        cmd: the command to execute. May be:
            - a string (with positional fields to .format, "{}" or "{0}" etc)
            - a function (with positional parameter count matching param_count)
            - None, if only an acmd is provided
        acmd: an async function (coroutine) to execute
            Must have positional parameter count matching param_count

        The next three inputs are only valid if cmd is a string:
        exec_str: a function of one parameter to execute the command string
        aexec_str: a coroutine of one parameter to execute the command string
        parse_function: a function to transform the return value of the command

        the last input is how to handle missing commands
        no_cmd_function: don't throw an error on definition if no command found
            instead, call this function when the command is invoked, which
            probably should thow an error of its own (ie NotImplementedError)

    return:
        2-tuple (call, acall)
            call is a function (that takes param_count arguments)
            acall is a coroutine
    '''
    return _SyncableCommand(param_count, cmd, acmd, exec_str, aexec_str,
                            parse_function, no_cmd_function
                            ).out()


class _SyncableCommand(object):
    def __init__(self, param_count, cmd, acmd, exec_str, aexec_str,
                 parse_function, no_cmd_function):
        self.param_count = param_count
        self.cmd = cmd
        self.acmd = acmd
        self.exec_str = exec_str
        self.aexec_str = aexec_str
        self.parse_function = parse_function
        self.no_cmd_function = no_cmd_function

        self.exec_function = None
        self.aexec_function = None

    # wrappers that may or may not be used below in constructing call / acall
    def call_by_str(self, *args):
        return self.exec_str(self.cmd.format(*args))

    @asyncio.coroutine
    def acall_by_str(self, *args):
        return (yield from self.aexec_str(self.cmd.format(*args)))

    def call_by_str_parsed(self, *args):
        return self.parse_function(self.exec_str(self.cmd.format(*args)))

    @asyncio.coroutine
    def acall_by_str_parsed(self, *args):
        raw_value = yield from self.aexec_str(self.cmd.format(*args))
        return self.parse_function(raw_value)

    def call_sync_by_afunction(self, *args):
        return wait_for_async(self.aexec_function, *args)

    # another layer to wrap the exec functions with parameter validation
    def validate_param_count(self, args):
        if len(args) != self.param_count:
            raise TypeError(
                'command takes exactly {} parameters'.format(self.param_count))

    def call(self, *args):
        self.validate_param_count(args)
        return self.exec_function(*args)

    @asyncio.coroutine
    def acall(self, *args):
        self.validate_param_count(args)
        return (yield from self.aexec_function(*args))

    def out(self):
        if isinstance(self.cmd, str):
            if self.parse_function is None:
                parsed = False
            elif is_function(self.parse_function, 1):
                parsed = True
            else:
                raise TypeError(
                    'parse_function must be a function with one arg,' +
                    ' not {}'.format(repr(self.parse_function)))

            if is_function(self.exec_str, 1):
                self.exec_function = (self.call_by_str_parsed if parsed else
                                      self.call_by_str)
            elif self.exec_str is not None:
                raise TypeError('exec_str must be a function with one arg,' +
                                ' not {}'.format(repr(self.exec_str)))

            if is_function(self.aexec_str, 1, coroutine=True):
                self.aexec_function = (self.acall_by_str_parsed if parsed else
                                       self.acall_by_str)
            elif self.aexec_str is not None:
                raise TypeError('aexec_str must be a coroutine with one arg')
        elif is_function(self.cmd, self.param_count):
            self.exec_function = self.cmd
        elif self.cmd is not None:
            raise TypeError('cmd must be a string or function with ' +
                            '{} parameters'.format(self.param_count))

        if is_function(self.acmd, self.param_count, coroutine=True):
            self.aexec_function = self.acmd
        elif self.acmd is not None:
            raise TypeError('acmd must be a coroutine with ' +
                            '{} parameters'.format(self.param_count))

        # do we need to create the sync or async version from the other?
        if self.exec_function is None:
            if self.aexec_function is None:
                # neither sync or async provided: either raise an error,
                # or return a default function (which probably itself just
                # raises an error when called, but need not raise an error
                # on creation, ie if it's OK for this command to be absent)
                if self.no_cmd_function is not None:
                    return (self.no_cmd_function,
                            mock_async(self.no_cmd_function))
                else:
                    raise NoCommandError(
                        'not enough information to construct this command')
            self.exec_function = self.call_sync_by_afunction
        elif self.aexec_function is None:
            self.aexec_function = mock_async(self.exec_function)

        return (self.call, self.acall)
