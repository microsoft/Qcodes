from __future__ import annotations

import enum
from operator import xor
from typing import cast

from .parameter import ManualParameter, Parameter


class ScaledParameter(Parameter):
    """
    :class:`.Parameter` Scaler

    To be used when you use a physical voltage divider or an amplifier to set
    or get a quantity.

    Initialize the parameter by passing the parameter to be measured/set
    and the value of the division OR the gain.

    The scaling value can be either a scalar value or a Qcodes Parameter.

    The parameter scaler acts a your original parameter, but will set the right
    value, and store the gain/division in the metadata.

    Examples:
        Resistive voltage divider
        >>> vd = ScaledParameter(dac.chan0, division = 10)

        Voltage multiplier
        >>> vb = ScaledParameter(dac.chan0, gain = 30, name = 'Vb')

        Transimpedance amplifier
        >>> Id = ScaledParameter(multimeter.amplitude,
        ...                      division = 1e6, name = 'Id', unit = 'A')

    Args:
        output: Physical Parameter that need conversion.
        division: The division value.
        gain: The gain value.
        label: Label of this parameter, by default uses 'output' label
            but attaches _amplified or _attenuated depending if gain
            or division has been specified.
        name: Name of this parameter, by default uses 'output' name
            but attaches _amplified or _attenuated depending if gain
            or division has been specified.
        unit: Resulting unit. It uses the one of 'output' by default.
    """

    class Role(enum.Enum):
        GAIN = enum.auto()
        DIVISION = enum.auto()

    def __init__(
        self,
        output: Parameter,
        division: float | Parameter | None = None,
        gain: float | Parameter | None = None,
        name: str | None = None,
        label: str | None = None,
        unit: str | None = None,
    ) -> None:

        # Set label
        if label:
            self.label = label
        elif name:
            self.label = name
        else:
            self.label = f"{output.label}_scaled"

        # Set the name
        if not name:
            name = f"{output.name}_scaled"

        # Set the unit
        if unit:
            self.unit = unit
        else:
            self.unit = output.unit

        super().__init__(name=name, label=self.label, unit=self.unit)

        self._wrapped_parameter = output
        self._wrapped_instrument = getattr(output, "_instrument", None)

        # Set the role, either as divider or amplifier
        # Raise an error if nothing is specified
        is_divider = division is not None
        is_amplifier = gain is not None

        if not xor(is_divider, is_amplifier):
            raise ValueError("Provide only division OR gain")

        if division is not None:
            self.role = ScaledParameter.Role.DIVISION
            # Unfortunately mypy does not support
            # properties where the setter has different types than
            # the actual property. We use that here to cast different inputs
            # to the same type.
            # https://github.com/python/mypy/issues/3004
            self._multiplier = division  # type: ignore[assignment]
        elif gain is not None:
            self.role = ScaledParameter.Role.GAIN
            self._multiplier = gain  # type: ignore[assignment]

        # extend metadata
        self._meta_attrs.extend(["division"])
        self._meta_attrs.extend(["gain"])
        self._meta_attrs.extend(["role"])
        self.metadata["wrapped_parameter"] = self._wrapped_parameter.name
        if self._wrapped_instrument:
            wrapped_instr_name = getattr(self._wrapped_instrument, "name", None)
            self.metadata["wrapped_instrument"] = wrapped_instr_name

    # Internal handling of the multiplier
    # can be either a Parameter or a scalar
    @property
    def _multiplier(self) -> Parameter:
        if self._multiplier_parameter is None:
            raise RuntimeError(
                "Cannot get multiplier when multiplier parameter in unknown."
            )
        return self._multiplier_parameter

    @_multiplier.setter
    def _multiplier(self, multiplier: float | Parameter) -> None:
        if isinstance(multiplier, Parameter):
            self._multiplier_parameter = multiplier
            multiplier_name = self._multiplier_parameter.name
            self.metadata["variable_multiplier"] = multiplier_name
        else:
            self._multiplier_parameter = ManualParameter(
                "multiplier", initial_value=multiplier
            )
            self.metadata["variable_multiplier"] = False

    # Division of the scaler
    @property
    def division(self) -> float:
        value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.DIVISION:
            return value
        elif self.role == ScaledParameter.Role.GAIN:
            return 1 / value
        else:
            raise ValueError(f"Unexpected role {self.role}")

    @division.setter
    def division(self, division: float | Parameter) -> None:
        self.role = ScaledParameter.Role.DIVISION
        self._multiplier = division  # type: ignore[assignment]

    # Gain of the scaler
    @property
    def gain(self) -> float:
        value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.GAIN:
            return value
        elif self.role == ScaledParameter.Role.DIVISION:
            return 1 / value
        else:
            raise ValueError(f"Unexpected role {self.role}")

    @gain.setter
    def gain(self, gain: float | Parameter) -> None:
        self.role = ScaledParameter.Role.GAIN
        self._multiplier = gain  # type: ignore[assignment]

    # Getter and setter for the real value
    def get_raw(self) -> float:
        """
        Returns:
            value at which was set at the sample
        """
        wrapped_value = cast(float, self._wrapped_parameter())
        multiplier = cast(float, self._multiplier())

        if self.role == ScaledParameter.Role.GAIN:
            value = wrapped_value * multiplier
        elif self.role == ScaledParameter.Role.DIVISION:
            value = wrapped_value / multiplier
        else:
            raise RuntimeError(
                f"ScaledParameter must be either a"
                f"Multiplier or Divisor; got {self.role}"
            )

        return value

    @property
    def wrapped_parameter(self) -> Parameter:
        """
        The attached unscaled parameter
        """
        return self._wrapped_parameter

    def get_wrapped_parameter_value(self) -> float:
        """
        Returns:
            value at which the attached parameter is (i.e. does
            not account for the scaling)
        """
        return self._wrapped_parameter.get()

    def set_raw(self, value: float) -> None:
        """
        Set the value on the wrapped parameter, accounting for the scaling
        """
        multiplier_value = cast(float, self._multiplier())
        if self.role == ScaledParameter.Role.GAIN:
            instrument_value = value / multiplier_value
        elif self.role == ScaledParameter.Role.DIVISION:
            instrument_value = value * multiplier_value
        else:
            raise RuntimeError(
                f"ScaledParameter must be either a"
                f"Multiplier or Divisor; got {self.role}"
            )

        self._wrapped_parameter.set(instrument_value)
