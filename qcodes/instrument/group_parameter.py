from collections import OrderedDict
from typing import List, Union, Callable, Dict, Any, Optional

from qcodes.instrument.parameter import Parameter
from qcodes import Instrument


class GroupParameter(Parameter):
    """
    Group parameter is a `Parameter` which value can be set or gotten only
    together with other group parameters. This happens when an instrument
    has commands which set and get more than one parameter per call.

    The `set_raw` method of a group parameter forwards the call to the
    group, and the group then makes sure that the values of other parameters
    within the group are left unchanged. The `get_raw` method of a group
    parameter also forwards the call to the group, and the group makes sure
    that the command output is parsed correctly, and the value of the
    parameter of interest is returned.

    After initialization, the group parameters need to be added to a group.
    See `Group` for more information.

    Args:
        name
            name of the parameter
        instrument
            instrument that this parameter belongs to; this instrument is
            used by the group to call its get and set commands

        **kwargs:
            All kwargs used by the Parameter class, except set_cmd and get_cmd
    """

    def __init__(self,
                 name: str,
                 instrument: Optional['Instrument'] = None,
                 **kwargs
                 ) -> None:

        if "set_cmd" in kwargs or "get_cmd" in kwargs:
            raise ValueError("A GroupParameter does not use 'set_cmd' or "
                             "'get_cmd' kwarg")

        self.group: Union[Group, None] = None
        super().__init__(name, instrument=instrument, **kwargs)

        self.set = self._wrap_set(self.set_raw)

        self.get_raw = lambda result=None: result if result is not None \
            else self._get_raw_value()

        self.get = self._wrap_get(self.get_raw)

    def _get_raw_value(self) -> Any:
        if self.group is None:
            raise RuntimeError("Trying to get Group value but no "
                               "group defined")
        self.group.update()
        return self.raw_value

    def set_raw(self, value: Any) -> None:
        if self.group is None:
            raise RuntimeError("Trying to set Group value but no "
                               "group defined")
        self.group.set(self, value)


class Group:
    """
    The group combines `GroupParameter`s that are to be gotten or set via the
    same command. The command has to be a string, for example, a VISA command.

    The `Group`'s methods are used within `GroupParameter` in order to
    properly implement setting and getting of a single parameter in the
    situation where one command sets or gets more than one parameter.

    The command used for setting values of parameters has to be a format
    string which contains the names of the parameters the group has been
    initialized with. For example, if a command has syntax `"CMD a_value,
    b_value"`, where `'a_value'` and `'b_value'` are values of two parameters
    with names `"a"` and `"b"`, then the command string has to be "CMD {a},
    {b}", and the group has to be initialized with two `GroupParameter`s
    `a_param` and `b_param`, where `a_param.name=="a"` and `b_param.name=="b"`.

    Note that by default, it is assumed that the command used for getting
    values returns a comma-separated list of values of parameters, and their
    order corresponds to the order of `GroupParameter`s in the list that is
    passed to the `Group`'s constructor. Through keyword arguments of the
    `Group`'s constructor, it is possible to change the separator, and even
    the parser of the output of the get command.

    The get and set commands are called via the instrument that the first
    parameter belongs to. It is assumed that all the parameters within the
    group belong to the same instrument.

    Example:
        ```
        class InstrumentWithGroupParameters(VisaInstrument):
            def __init__(self, name, address, **kwargs):
                super().__init__(name, address, **kwargs)

                ...

                # Here is how group of group parameters is defined for
                # a simple case of an example "SGP" command that sets and gets
                # values of "enabled" and "gain" parameters (it is assumed that
                # "SGP?" returns the parameter values as comma-separated list
                # "enabled_value,gain_value")
                self.add_parameter('enabled',
                                   label='Enabled',
                                   val_mapping={True: 1, False: 0},
                                   parameter_class=GroupParameter)
                self.add_parameter('gain',
                                   label='Some gain value',
                                   get_parser=float,
                                   parameter_class=GroupParameter)
                self.output_group = Group([self.enabled, self.gain],
                                          set_cmd='SGP {enabled}, {gain}',
                                          get_cmd='SGP?')

                ...
        ```

    Args:
        parameters
            a list of `GroupParameter` instances which have to be gotten and
            set via the same command; the order of parameters in the list
            should correspond to the order of the values returned by the
            `get_cmd`
        set_cmd
            format string of the command that is used for setting the values
            of the parameters; for example, "CMD {a}, {b}"
        get_cmd
            string of the command that is used for getting the values of the
            parameters; for example, "CMD?"
        separator
            a separator that is used when parsing the output of the `get_cmd`
            in order to obtain the values of the parameters; it is ignored in
            case a custom `get_parser` is used
        get_parser
            a callable with a single string argument that is used to parse
            the output of the `get_cmd`; the callable has to return a
            dictionary where parameter names are keys, and the values are the
            values (as directly obtained from the output of the get command;
            note that parsers within the parameters will take care of
            individual parsing of their values)
    """
    def __init__(self,
                 parameters: List[GroupParameter],
                 set_cmd: str=None,
                 get_cmd: str=None,
                 get_parser: Union[Callable[[str], Dict[str, Any]], None]=None,
                 separator: str=','
                 ) -> None:
        self.parameters = OrderedDict((p.name, p) for p in parameters)

        for p in parameters:
            p.group = self

        if len(set([p.root_instrument for p in parameters])) > 1:
            raise ValueError(
                "All parameters should belong to the same instrument")

        self.instrument = parameters[0].root_instrument

        self.set_cmd = set_cmd
        self.get_cmd = get_cmd

        if get_parser:
            self.get_parser = get_parser
        else:
            self.get_parser = self._separator_parser(separator)

    def _separator_parser(self, separator: str
                          ) -> Callable[[str], Dict[str, Any]]:
        """A default separator-based string parser"""
        def parser(ret_str: str) -> Dict[str, Any]:
            keys = self.parameters.keys()
            values = ret_str.split(separator)
            return dict(zip(keys, values))

        return parser

    def set(self, set_parameter: GroupParameter, value: Any):
        """
        Sets the value of the given parameter within a group to the given
        value by calling the `set_cmd`

        Args:
            set_parameter
                the parameter within the group to set
            value
                the new value for this parameter
        """
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.raw_value
                        for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = value
        if self.set_cmd is None:
            raise RuntimeError("Calling set but no `set_cmd` defined")
        command_str = self.set_cmd.format(**calling_dict)
        if self.instrument is None:
            raise RuntimeError("Trying to set GroupParameter not attached "
                               "to any instrument.")
        self.instrument.write(command_str)

    def update(self):
        """
        Update the values of all the parameters within the group by calling
        the `get_cmd`
        """
        ret = self.get_parser(self.instrument.ask(self.get_cmd))
        for name, p in list(self.parameters.items()):
            p.get(result=ret[name])
