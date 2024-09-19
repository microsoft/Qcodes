"""
This module implements a :class:`.Group` intended to hold multiple
parameters that are to be gotten and set by the same command. The parameters
should be of type :class:`GroupParameter`
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from qcodes.instrument.base import InstrumentBase

    from .parameter_base import ParamDataType, ParamRawDataType


class GroupParameter(Parameter):
    """
    Group parameter is a :class:`.Parameter`, whose value can be set or get
    only with other group parameters. This happens when an instrument
    has commands which set and get more than one parameter per call.

    The ``set_raw`` method of a group parameter forwards the call to the
    group, and the group then makes sure that the values of other parameters
    within the group are left unchanged. The ``get_raw`` method of a group
    parameter also forwards the call to the group, and the group makes sure
    that the command output is parsed correctly, and the value of the
    parameter of interest is returned.

    After initialization, the group parameters need to be added to a group.
    See :class:`.Group` for more information.

    Args:
        name: Name of the parameter.
        instrument: Instrument that this parameter belongs to; this instrument
            is used by the group to call its get and set commands.
        initial_value: Initial value of the parameter. Note that either none or
            all of the parameters in a :class:`.Group` should have an initial
            value.

        **kwargs: All kwargs used by the :class:`.Parameter` class, except
             ``set_cmd`` and ``get_cmd``.
    """

    def __init__(
        self,
        name: str,
        instrument: InstrumentBase | None = None,
        initial_value: float | int | str | None = None,
        **kwargs: Any,
    ) -> None:
        if "set_cmd" in kwargs or "get_cmd" in kwargs:
            raise ValueError(
                "A GroupParameter does not use 'set_cmd' or 'get_cmd' kwarg"
            )

        self._group: Group | None = None
        self._initial_value = initial_value
        super().__init__(name, instrument=instrument, **kwargs)

    @property
    def group(self) -> Group | None:
        """
        The group that this parameter belongs to.
        """
        return self._group

    def get_raw(self) -> ParamRawDataType:
        if self.group is None:
            raise RuntimeError("Trying to get Group value but no group defined")
        self.group.update()
        return self.cache.raw_value

    def set_raw(self, value: ParamRawDataType) -> None:
        if self.group is None:
            raise RuntimeError("Trying to set Group value but no group defined")
        self.group._set_one_parameter_from_raw(self, value)


class Group:
    """
    The group combines :class:`.GroupParameter` s that are to be gotten or set
    via the same command. The command has to be a string, for example,
    a VISA command.

    The :class:`Group`'s methods are used within :class:`GroupParameter` in
    order to properly implement setting and getting of a single parameter in
    the situation where one command sets or gets more than one parameter.

    The command used for setting values of parameters has to be a format
    string which contains the names of the parameters the group has been
    initialized with. For example, if a command has syntax ``CMD a_value,
    b_value``, where ``a_value`` and ``b_value`` are values of two parameters
    with names ``a`` and ``b``, then the command string has to be ``CMD {a},
    {b}``, and the group has to be initialized with two ``GroupParameter`` s
    ``a_param`` and ``b_param``, where ``a_param.name=="a"`` and
    ``b_param.name=="b"``.

    **Note** that by default, it is assumed that the command used for getting
    values returns a comma-separated list of values of parameters, and their
    order corresponds to the order of :class:`.GroupParameter` s in the list
    that is passed to the :class:`Group`'s constructor. Through keyword
    arguments of the :class:`Group`'s constructor, it is possible to change
    the separator, and even the parser of the output of the get command.

    The get and set commands are called via the instrument that the first
    parameter belongs to. It is assumed that all the parameters within the
    group belong to the same instrument.

    Example:
        ::

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

    Args:
        parameters: a list of :class:`.GroupParameter` instances which have
            to be gotten and set via the same command; the order of
            parameters in the list should correspond to the order of the
            values returned by the ``get_cmd``.
        set_cmd: Format string of the command that is used for setting the
            values of the parameters; for example, ``CMD {a}, {b}``.
        get_cmd: String of the command that is used for getting the values
            of the parameters; for example, ``CMD?``.
        separator: A separator that is used when parsing the output of the
            ``get_cmd`` in order to obtain the values of the parameters; it
            is ignored in case a custom ``get_parser`` is used.
        get_parser: A callable with a single string argument that is used to
            parse the output of the ``get_cmd``; the callable has to return a
            dictionary where parameter names are keys, and the values are the
            values (as directly obtained from the output of the get command;
            note that parsers within the parameters will take care of
            individual parsing of their values).
        single_instrument: A flag to indicate that all parameters belong to a
        single instrument, which in turn does additional checks. Defaults to True.
    """

    def __init__(
        self,
        parameters: Sequence[GroupParameter],
        set_cmd: str | None = None,
        get_cmd: str | None = None,
        get_parser: Callable[[str], Mapping[str, Any]] | None = None,
        separator: str = ",",
        single_instrument: bool = True,
    ) -> None:
        self._parameters = OrderedDict((p.name, p) for p in parameters)

        for p in parameters:
            p._group = self

        if single_instrument:
            if len({p.root_instrument for p in parameters}) > 1:
                raise ValueError("All parameters should belong to the same instrument")

        self._instrument = parameters[0].root_instrument

        self._set_cmd = set_cmd
        self._get_cmd = get_cmd

        if get_parser:
            self.get_parser = get_parser
        else:
            self.get_parser = self._separator_parser(separator)

        if single_instrument:
            self._check_initial_values(parameters)

    def _check_initial_values(self, parameters: Sequence[GroupParameter]) -> None:
        have_initial_values = [p._initial_value is not None for p in parameters]
        if any(have_initial_values):
            if not all(have_initial_values):
                params_with_initial_values = [
                    p.name for p in parameters if p._initial_value is not None
                ]
                params_without_initial_values = [
                    p.name for p in parameters if p._initial_value is None
                ]
                error_msg = (
                    f"Either none or all of the parameters in a "
                    f"group should have an initial value. Found "
                    f"initial values for "
                    f"{params_with_initial_values} but not for "
                    f"{params_without_initial_values}."
                )
                raise ValueError(error_msg)

            calling_dict = {
                name: p._from_value_to_raw_value(p._initial_value)
                for name, p in self.parameters.items()
            }

            self._set_from_dict(calling_dict)

    def _separator_parser(
        self, separator: str
    ) -> Callable[[str], dict[str, ParamRawDataType]]:
        """A default separator-based string parser"""

        def parser(ret_str: str) -> dict[str, Any]:
            keys = self.parameters.keys()
            values = ret_str.split(separator)
            return dict(zip(keys, values))

        return parser

    def set_parameters(self, parameters_dict: Mapping[str, ParamDataType]) -> None:
        """
        Sets the value of one or more parameters within a group to the given
        values by calling the ``set_cmd`` while updating rest.

        Args:
            parameters_dict: The dictionary of one or more parameters within
            the group with the corresponding values to be set.
        """
        if not parameters_dict:
            raise RuntimeError(
                "Provide at least one group parameter and its value to be set."
            )
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.cache.raw_value for name, p in self.parameters.items()}
        for parameter_name, value in parameters_dict.items():
            p = self.parameters[parameter_name]
            raw_value = p._from_value_to_raw_value(value)
            calling_dict[parameter_name] = raw_value

        self._set_from_dict(calling_dict)

    def _set_one_parameter_from_raw(
        self, set_parameter: GroupParameter, raw_value: ParamRawDataType
    ) -> None:
        """
        Sets the raw_value of the given parameter within a group to the given
        raw_value by calling the ``set_cmd``.

        Args:
            set_parameter: The parameter within the group to set.
            raw_value: The new raw_value for this parameter.
        """
        # TODO replace get latest with call to cache.invalid once that lands
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.cache.raw_value for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = raw_value

        self._set_from_dict(calling_dict)

    def _set_from_dict(self, calling_dict: Mapping[str, ParamRawDataType]) -> None:
        """
        Use ``set_cmd`` to parse a dict that maps parameter names to parameter
        raw values, and actually perform setting the values.
        """
        if self._set_cmd is None:
            raise RuntimeError("Calling set but no `set_cmd` defined")
        command_str = self._set_cmd.format(**calling_dict)
        if self.instrument is None:
            raise RuntimeError(
                "Trying to set GroupParameter not attached to any instrument."
            )
        self.instrument.write(command_str)
        for name, p in list(self.parameters.items()):
            p.cache._set_from_raw_value(calling_dict[name])

    def update(self) -> None:
        """
        Update the values of all the parameters within the group by calling
        the ``get_cmd``.
        """
        if self.instrument is None:
            raise RuntimeError(
                "Trying to update GroupParameter not attached to any instrument."
            )
        if self._get_cmd is None:
            parameter_names = ", ".join(p.full_name for p in self.parameters.values())
            raise RuntimeError(
                f"Cannot update values in the group with "
                f"parameters - {parameter_names} since it "
                f"has no `get_cmd` defined."
            )
        ret = self.get_parser(self.instrument.ask(self._get_cmd))
        for name, p in list(self.parameters.items()):
            p.cache._set_from_raw_value(ret[name])

    @property
    def parameters(self) -> OrderedDict[str, GroupParameter]:
        """
        All parameters in this group as a dict from parameter name to
        :class:`.Parameter`
        """
        return self._parameters

    @property
    def instrument(self) -> InstrumentBase | None:
        """
        The ``root_instrument`` that this parameter belongs to.
        """
        return self._instrument
