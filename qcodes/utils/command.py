from qcodes.utils.helpers import is_function


class NoCommandError(Exception):
    pass


class Command:
    """
    Create a callable command from a string or function.

    Args:
        arg_count (int): the number of arguments to the command

        cmd (Optional[Union[str, Callable]]): If a function, it will be called directly
            when the command is invoked. If a string, it should contain
            positional fields to ``.format`` like ``'{}'`` or ``'{0}'``,
            and it will be passed on to ``exec_str`` after formatting

        exec_str (Optional[Callable]): If provided, should be a callable
            taking one parameter, the ``cmd`` string after parameters
            are inserted. If not provided, ``cmd`` must not be a string.

        input_parser (Optional[Callable]): Transform the input arg(s) before
            sending them to the command. If there are multiple arguments, this
            function should accept all the arguments in order, and
            return a tuple of values.

        output_parser (Optional[Callable]): transform the return value of the
            command.

        no_cmd_function (Optional[Callable]): If provided and we cannot
            create a command to return, we won't throw an error on constructing
            the command. Instead, we call this function when the command is
            invoked, and it should probably throw an error of its own (eg
            ``NotImplementedError``)


    Raises:
        TypeError: if no_cmd_function is not the expected type
        TypeError: if input_parser is not the expected type
        TypeError: if output_parser is not the expected type
        TypeError: if exec_string is not the expected type
        NoCommandError: if no cmd is found no_cmd_function is missing
    """

    def __init__(self, arg_count, cmd=None, exec_str=None, input_parser=None,
                 output_parser=None, no_cmd_function=None):

        self.arg_count = arg_count

        if (no_cmd_function is not None and
                not is_function(no_cmd_function, arg_count)):
            raise TypeError('no_cmd_function must be none or a function '
                            'taking the same args as the command, not '
                            '{}'.format(no_cmd_function))

        if input_parser is None:
            parse_input = False
        elif is_function(input_parser, arg_count):
            parse_input = True if arg_count == 1 else 'multi'
            self.input_parser = input_parser
        else:
            raise TypeError(
                'input_parser must be a function with one arg_count args,'
                ' not {}'.format(repr(input_parser)))

        if output_parser is None:
            parse_output = False
        elif is_function(output_parser, 1):
            parse_output = True
            self.output_parser = output_parser
        else:
            raise TypeError(
                'output_parser must be a function with one arg,'
                ' not {}'.format(repr(output_parser)))

        if isinstance(cmd, str):
            self.cmd_str = cmd
            self.exec_str = exec_str

            if is_function(exec_str, 1):
                self.exec_function = {  # (parse_input, parse_output)
                    (False, False): self.call_by_str,
                    (False, True): self.call_by_str_parsed_out,
                    (True, False): self.call_by_str_parsed_in,
                    (True, True): self.call_by_str_parsed_in_out,
                    ('multi', False): self.call_by_str_parsed_in2,
                    ('multi', True): self.call_by_str_parsed_in2_out
                }[(parse_input, parse_output)]

            elif exec_str is not None:
                raise TypeError('exec_str must be a function with one arg,' +
                                ' not {}'.format(repr(exec_str)))

        elif is_function(cmd, arg_count):
            self._cmd = cmd
            self.exec_function = {  # (parse_input, parse_output)
                (False, False): cmd,
                (False, True): self.call_cmd_parsed_out,
                (True, False): self.call_cmd_parsed_in,
                (True, True): self.call_cmd_parsed_in_out,
                ('multi', False): self.call_cmd_parsed_in2,
                ('multi', True): self.call_cmd_parsed_in2_out
            }[(parse_input, parse_output)]

        elif cmd is None:
            if no_cmd_function is not None:
                self.exec_function = no_cmd_function
            else:
                raise NoCommandError('no ``cmd`` provided')

        else:
            raise TypeError('cmd must be a string or function with ' +
                            '{} args'.format(arg_count))

    # wrappers that may or may not be used in constructing call
    # these functions are not very DRY at all - this could be condensed
    # by composing them from a smaller set. But this is our hot path
    # during acquisition Loops, so for performance I wanted to minimize
    # overhead from branching or extra function calls and stack frames.
    # TODO(giulioungaretti) wihtout benchmarks this is "just like your opinion man"

    def call_by_str(self, *args):
        """Execute a formatted string."""
        return self.exec_str(self.cmd_str.format(*args))

    def call_by_str_parsed_out(self, *args):
        """Execute a formatted string with output parsing."""
        return self.output_parser(self.exec_str(self.cmd_str.format(*args)))

    def call_by_str_parsed_in(self, arg):
        """Execute a formatted string with 1-arg input parsing."""
        return self.exec_str(self.cmd_str.format(self.input_parser(arg)))

    def call_by_str_parsed_in_out(self, arg):
        """Execute a formatted string with 1-arg input and output parsing."""
        return self.output_parser(self.exec_str(
            self.cmd_str.format(self.input_parser(arg))))

    def call_by_str_parsed_in2(self, *args):
        """Execute a formatted string with multi-arg input parsing."""
        return self.exec_str(self.cmd_str.format(*self.input_parser(*args)))

    def call_by_str_parsed_in2_out(self, *args):
        """Execute a formatted string with multi-arg input & output parsing."""
        return self.output_parser(self.exec_str(
            self.cmd_str.format(*self.input_parser(*args))))

    # And the same for parsing + command as a function

    def call_cmd_parsed_out(self, *args):
        """Execute a function with output parsing."""
        return self.output_parser(self._cmd(*args))

    def call_cmd_parsed_in(self, arg):
        """Execute a function with 1-arg input parsing."""
        return self._cmd(self.input_parser(arg))

    def call_cmd_parsed_in_out(self, arg):
        """Execute a function with 1-arg input and output parsing."""
        return self.output_parser(self._cmd(self.input_parser(arg)))

    def call_cmd_parsed_in2(self, *args):
        """Execute a function with multi-arg input parsing."""
        return self._cmd(*self.input_parser(*args))

    def call_cmd_parsed_in2_out(self, *args):
        """Execute a function with multi-arg input & output parsing."""
        return self.output_parser(self._cmd(*self.input_parser(*args)))

    def __call__(self, *args):
        """Invoke the command."""
        if len(args) != self.arg_count:
            raise TypeError(
                'command takes exactly {} args'.format(self.arg_count))
        return self.exec_function(*args)
