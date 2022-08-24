from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from qcodes.metadatable import Metadatable
from qcodes.validators import Validator, validate_all

from .command import Command

if TYPE_CHECKING:
    from qcodes.instrument import InstrumentBase


class Function(Metadatable):
    """
    Defines a function  that an instrument can execute.

    This class is meant for simple cases, principally things that
    map to simple commands like ``*RST`` (reset) or those with just a few
    arguments.
    It requires a fixed argument count, and positional args
    only.

    You execute this function object like a normal function, or use its
    .call method.

    Note:
        Parsers only apply if call_cmd is a string. The function form of
        call_cmd should do its own parsing.

    Note:
        We do not recommend the usage of Function for any new driver.
        Function does not add any significant features over a method
        defined on the class.


    Args:
        name: the local name of this function

        instrument: an instrument that handles this
            function. Default None.

        call_cmd: command to execute on
            the instrument:

            - a string (with positional fields to .format, "{}" or "{0}" etc)
              you can only use a string if an instrument is provided,
              this string will be passed to instrument.write

            - a function (with arg count matching args list)

        args: list of Validator objects, one for
            each arg to the Function

        arg_parser: function to transform the input arg(s)
            to encoded value(s) sent to the instrument.  If there are multiple
            arguments, this function should accept all the arguments in order,
            and return a tuple of values.

        return_parser: function to transform the response
            from the instrument to the final output value.  may be a
            type casting function like `int` or `float`.  If None (default),
            will not wait for or read any response.

        docstring: documentation string for the __doc__
            field of the object. The __doc__ field of the instance is used by
            some help systems, but not all (particularly not builtin `help()`)

        **kwargs: Arbitrary keyword arguments passed to parent class

    """

    def __init__(
        self,
        name: str,
        instrument: InstrumentBase | None = None,
        call_cmd: str | Callable[..., Any] | None = None,
        args: Sequence[Validator[Any]] | None = None,
        arg_parser: Callable[..., Any] | None = None,
        return_parser: Callable[..., Any] | None = None,
        docstring: str | None = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        self._instrument = instrument
        self.name = name

        if docstring is not None:
            self.__doc__ = docstring
        if args is None:
            args = []
        self._set_args(args)
        self._set_call(call_cmd, arg_parser, return_parser)

    def _set_args(self, args: Sequence[Validator[Any]]) -> None:
        for arg in args:
            if not isinstance(arg, Validator):
                raise TypeError("all args must be Validator objects")
        self._args = args
        self._arg_count = len(args)

    def _set_call(
        self,
        call_cmd: str | Callable[..., Any] | None,
        arg_parser: Callable[..., Any] | None,
        return_parser: Callable[..., Any] | None,
    ) -> None:
        if self._instrument:
            ask_or_write = self._instrument.write
            if isinstance(call_cmd, str) and return_parser:
                ask_or_write = self._instrument.ask
        else:
            ask_or_write = None

        self._call = Command(
            arg_count=self._arg_count,
            cmd=call_cmd,
            exec_str=ask_or_write,
            input_parser=arg_parser,
            output_parser=return_parser,
        )

    def validate(self, *args: Any) -> None:
        """
        Check that all arguments to this Function are allowed.

        Args:
            *args: Variable length argument list, passed to the call_cmd
        """
        if self._instrument:
            func_name = (
                (
                    getattr(self._instrument, "name", "")
                    or str(self._instrument.__class__)
                )
                + "."
                + self.name
            )
        else:
            func_name = self.name

        if len(args) != self._arg_count:
            raise TypeError(
                "{} called with {} args but requires {}".format(
                    func_name, len(args), self._arg_count
                )
            )

        validate_all(*zip(self._args, args), context="Function: " + func_name)

    def __call__(self, *args: Any) -> Any:
        self.validate(*args)
        return self._call(*args)

    def call(self, *args: Any) -> Any:
        """
        Call methods wraps __call__

        Args:
           *args: argument to pass to Command __call__ function
        """
        return self.__call__(*args)

    def get_attrs(self) -> list[str]:
        """
        Attributes recreated as properties in the RemoteFunction proxy.

        Returns (list): __doc__, _args, and _arg_count get proxied
        """
        return ["__doc__", "_args", "_arg_count"]
