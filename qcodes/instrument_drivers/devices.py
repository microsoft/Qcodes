from typing import Union, cast
from operator import xor
import enum
import warnings

from qcodes import Parameter, ManualParameter, Instrument


class ParameterScaler(Parameter):
    """
    Parameter Scaler

    To be used when you use a physical voltage divider or an amplifier to set
    or get a quantity.

    Initialize the parameter by passing the parameter to be measured/set
    and the value of the division OR the gain.

    The scaling value can be either a scalar value or a Qcodes Parameter.

    The parameter scaler acts a your original parameter, but will set the right
    value, and store the gain/division in the metadata.

    Examples:
        Resistive voltage divider
        >>> vd = ParameterScaler(dac.chan0, division = 10)

        Voltage multiplier
        >>> vb = ParameterScaler(dac.chan0, gain = 30, name = 'Vb')

        Transimpedance amplifier
        >>> Id = ParameterScaler(multimeter.amplitude, division = 1e6, name = 'Id', unit = 'A')

    Args:
        output: Physical Parameter that need conversion
        division: the division value
        gain: the gain value
        label: label of this parameter, by default uses 'output' label
            but attaches _amplified or _attenuated depending if gain
            or division has been specified
        name: name of this parameter, by default uses 'output' name
            but attaches _amplified or _attenuated depending if gain
            or division has been specified
        unit: resulting unit. It uses the one of 'output' by default
    """

    class Role(enum.Enum):
        GAIN = enum.auto()
        DIVISION = enum.auto()


    def __init__(self,
                 output: Parameter,
                 division: Union[int, float, Parameter] = None,
                 gain: Union[int, float, Parameter] = None,
                 name: str=None,
                 label: str=None,
                 unit: str=None) -> None:
        self.wrapped_parameter = output
        self.wrapped_instrument = getattr(output, "_instrument", None)

        # Set the role, either as divider or amplifier
        # Raise an error if nothing is specified
        is_divider = division is not None
        is_amplifier = gain is not None

        if not xor(is_divider, is_amplifier):
            raise ValueError('Provide only division OR gain')

        if is_divider:
            self.role = ParameterScaler.Role.DIVISION
            self._multiplier = division
        elif is_amplifier:
            self.role = ParameterScaler.Role.GAIN
            self._multiplier = gain

        # Set the name
        if name:
            self.name = name
        else:
            self.name = "{}_scaled".format(self.wrapped_parameter.name)

        # Set label
        if label:
            self.label = label
        elif name:
            self.label = name
        else:
            self.label = "{}_scaled".format(self.wrapped_parameter.label)

        # Set the unit
        if unit:
            self.unit = unit
        else:
            self.unit = self.wrapped_parameter.unit

        super().__init__(
            name=self.name,
            label=self.label,
            unit=self.unit
            )

        # extend metadata
        self._meta_attrs.extend(["division"])
        self._meta_attrs.extend(["gain"])
        self._meta_attrs.extend(["role"])
        self._meta_attrs.extend(["wrapped_parameter"])
        self._meta_attrs.extend(["wrapped_instrument"])

    # Internal handling of the multiplier
    # can be either a Parameter or a scalar
    @property
    def _multiplier(self):
        return self._multiplier_parameter

    @_multiplier.setter
    def _multiplier(self, multiplier: Union[int, float, Parameter]):
        if isinstance(multiplier, Parameter):
            self._multiplier_parameter = multiplier
        else:
            self._multiplier_parameter = ManualParameter(
                'multiplier', initial_value=multiplier)

    # Division of the scaler
    @property
    def division(self):
        if self.role == ParameterScaler.Role.DIVISION:
            return self._multiplier()
        elif self.role == ParameterScaler.Role.GAIN:
            return 1 / self._multiplier()

    @division.setter
    def division(self, division: Union[int, float, Parameter]):
        self.role = ParameterScaler.Role.DIVISION
        self._multiplier = division

    # Gain of the scaler
    @property
    def gain(self):
        if self.role == ParameterScaler.Role.GAIN:
            return self._multiplier()
        elif self.role == ParameterScaler.Role.DIVISION:
            return 1 / self._multiplier()

    @gain.setter
    def gain(self, gain: Union[int, float, Parameter]):
        self.role = ParameterScaler.Role.GAIN
        self._multiplier = gain

    # Getter and setter for the real value
    def get_raw(self) -> Union[int, float]:
        """
        Returns:
            number: value at which was set at the sample
        """
        if self.role == ParameterScaler.Role.GAIN:
            value = self.wrapped_parameter() * self._multiplier()
        elif self.role == ParameterScaler.Role.DIVISION:
            value = self.wrapped_parameter() / self._multiplier()

        self._save_val(value)
        return value

    def get_wrapped_parameter_value(self) -> Union[int, float]:
        """
        Returns:
            number: value at which the attached parameter is (i.e. does
            not account for the scaling)
        """
        return self.wrapped_parameter.get()

    def set_raw(self, value: Union[int, float]) -> None:
        """
        Set the value on the wrapped parameter, accounting for the scaling
        """

        # disable type check due to https://github.com/python/mypy/issues/2128
        if self.role == ParameterScaler.Role.GAIN:
            instrument_value = value / self._multiplier() # type: ignore
        elif self.role == ParameterScaler.Role.DIVISION:
            instrument_value = value * self._multiplier() # type: ignore

        # don't leak unknow type
        instrument_value = cast(Union[int, float], instrument_value)

        self._save_val(value)
        self.wrapped_parameter.set(instrument_value)


class VoltageDivider(Parameter):
    """
    Resitive voltage divider

    To be used when you use a physical voltage divider to set or get a voltage.

    Initialize the voltage diveder by passing the parameter to be measured
    and the value of the division (which should be calibrated beforehand)

    >>> vd = VoltageDivider(dac.chan0, 10)

    The voltage divider acts a your original parameter, but will set the right
    value, and store the division_value in the metadata.

    Set the value you want to set your device at 10 V

    >>> vd(10)

    This will set the dac.cha0 at 10*10, but upon measuring the divider
    the value returned is the voltage at the sample.

    >>> vd()
    10

    To get the voltage that was actually set on the instrument:

    >>> vd.get_instrument_value()
    100



    Args:
        v1: Parameter physically attached to the divider as input
        division_value: the divsion value of the divider
        label: label of this parameter, by default uses v1 label
            but attaches _attenuated
        name: name of this parameter, by default uses v1 name
            but attaches _attenuated
    """

    def __init__(self,
                 v1: Parameter,
                 division_value: Union[int, float],
                 name: str=None,
                 label: str=None,
                 instrument: Union[None, Instrument]=None) -> None:
        self.v1 = v1
        self.division_value = division_value
        if label:
            self.label = label
        else:
            self.label = "{}_attenuated".format(self.v1.label)

        if name:
            self.name = name
        else:
            self.name = "{}_attenuated".format(self.v1.name)
        if not instrument:
            instrument = getattr(self.v1, "_instrument", None)
        super().__init__(
            name=self.name,
            instrument=instrument,
            label=self.label,
            unit=self.v1.unit,
            metadata=self.v1.metadata)

        # extend metadata
        self._meta_attrs.extend(["division_value"])
        warnings.warn('`VoltageDivider` is deprecated, use `ParameterScaler` instead')

    def set_raw(self, value: Union[int, float]) -> None:
        instrument_value = value * self.division_value # type: ignore
        # disable type check due to https://github.com/python/mypy/issues/2128
        instrument_value = cast(Union[int, float], instrument_value)

        self._save_val(value)
        self.v1.set(instrument_value)

    def get_raw(self) -> Union[int, float]:
        """
        Returns:
            number: value at which was set at the sample
        """
        value = self.v1.get() / self.division_value
        self._save_val(value)
        return value

    def get_instrument_value(self) -> Union[int, float]:
        """
        Returns:
            number: value at which the attached paraemter is (i.e. does
            not account for the scaling)
        """
        return self.v1.get()
