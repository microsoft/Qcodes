import asyncio
from functools import partial

from qcodes.utils.helpers import is_function


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


def mock_sync(f):
    '''
    make a coroutine into a synchronous function
    '''
    return lambda *args, **kwargs: wait_for_async(f, *args, **kwargs)


def mock_async(f):
    '''
    make a synchronous function f awaitable
    '''
    async def f_awaitable(*args, **kwargs):
        return f(*args, **kwargs)

    return f_awaitable


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

    # wrappers that may or may not be used below in constructing call / acall
    def call_by_str(*args):
        return exec_str(cmd.format(*args))

    async def acall_by_str(*args):
        return await aexec_str(cmd.format(*args))

    def call_by_str_parsed(*args):
        return parse_function(exec_str(cmd.format(*args)))

    async def acall_by_str_parsed(*args):
        return parse_function(await aexec_str(cmd.format(*args)))

    def call_sync_by_afunction(*args):
        return wait_for_async(aexec_function, *args)

    # pull in all the inputs to see what was explicitly provided
    exec_function = None
    aexec_function = None

    if isinstance(cmd, str):
        if parse_function is None:
            parsed = False
        elif is_function(parse_function, 1):
            parsed = True
        else:
            raise TypeError('parse_function must be a function with one arg,' +
                            ' not {}'.format(repr(parse_function)))

        if is_function(exec_str, 1):
            exec_function = call_by_str_parsed if parsed else call_by_str
        elif exec_str is not None:
            raise TypeError('exec_str must be a function with one arg,' +
                            ' not {}'.format(repr(exec_str)))

        if is_function(aexec_str, 1, coroutine=True):
            aexec_function = acall_by_str_parsed if parsed else acall_by_str
        elif aexec_str is not None:
            raise TypeError('aexec_str must be a coroutine with one arg')
    elif is_function(cmd, param_count):
        exec_function = cmd
    elif cmd is not None:
        raise TypeError('cmd must be a string or function with ' +
                        '{} parameters'.format(param_count))

    if is_function(acmd, param_count, coroutine=True):
        aexec_function = acmd
    elif acmd is not None:
        raise TypeError('acmd must be a coroutine with ' +
                        '{} parameters'.format(param_count))

    # do we need to create the sync or async version from the other?
    if exec_function is None:
        if aexec_function is None:
            # neither sync or async provided: either raise an error,
            # or return a default function (which probably itself just
            # raises an error when called, but need not raise an error
            # on creation, ie if it's OK for this command to be absent)
            if no_cmd_function is not None:
                return (no_cmd_function, mock_async(no_cmd_function))
            else:
                raise NoCommandError(
                    'not enough information to construct this command')
        exec_function = call_sync_by_afunction
    elif aexec_function is None:
        aexec_function = mock_async(exec_function)

    # now wrap with parameter count validation and return the two versions
    def validate_param_count(args):
        if len(args) != param_count:
            raise TypeError(
                'command takes exactly {} parameters'.format(param_count))

    def call(*args):
        validate_param_count(args)
        return exec_function(*args)

    async def acall(*args):
        validate_param_count(args)
        return await aexec_function(*args)

    return (call, acall)
