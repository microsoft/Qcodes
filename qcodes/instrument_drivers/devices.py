from typing import Union, Optional

from qcodes import Parameter, Instrument


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
                 name: Optional[str] = None,
                 label: Optional[str] = None,
                 instrument: Union[None, Instrument] = None) -> None:
        self.v1 = v1
        self.division_value = division_value
        if label:
            self.label = label
        else:
            self.label = f"{self.v1.label}_attenuated"

        if not name:
            name = f"{self.v1.name}_attenuated"

        if not instrument:
            instrument = getattr(self.v1, "_instrument", None)
        super().__init__(
            name=name,
            instrument=instrument,
            label=self.label,
            unit=self.v1.unit,
            metadata=self.v1.metadata)

        # extend metadata
        self._meta_attrs.extend(["division_value"])

    def set_raw(self, value: Union[int, float]) -> None:
        instrument_value = value * self.division_value
        self.v1.set(instrument_value)

    def get_raw(self) -> Union[int, float]:
        """
        Returns:
            value at which was set at the sample
        """
        value = self.v1.get() / self.division_value
        return value

    def get_instrument_value(self) -> Union[int, float]:
        """
        Returns:
            value at which the attached parameter is (i.e. does
            not account for the scaling)
        """
        return self.v1.get()
