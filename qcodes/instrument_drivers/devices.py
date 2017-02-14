from typing import Union

from qcodes import Parameter, StandardParameter


class VoltageDivider(Parameter):
    """
    Resitive voltage divider

    To be used when you use a voltage divider to measure a parameter.
    This allows logging of your "divider", and you get the right data
    and label back.

    >>> vd = VoltageDivider(dac.chan0, 10)

    set the value you want to set your sample at

    >>> vd(10)

    This will set the dac.cha0 at 10*10, but it will look

    >>> vd()
    10
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
                 v1: StandardParameter,
                 division_value: Union[int, float],
                 name: str=None,
                 label: str=None) -> None:
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

        super().__init__(
            name=self.name,
            instrument=getattr(self.v1, "_instrument", None),
            label=self.label,
            unit=self.v1.unit,
            metadata=self.v1.metadata)

        # extend metadata
        self._meta_attrs.extend(['v1', 'division_value'])

    def set(self, value: Union[int, float]) -> None:
        instrument_value = value * self.division_value
        self._save_val(value)
        self.v1.set(instrument_value)

    def get(self) -> Union[int, float]:
        """
        Returns:
            number: value at which was set at the sample
        """
        return self.v1.get() / self.division_value

    def get_instrument_value(self) -> Union[int, float]:
        """
        Returns:
            number: value at which the attached paraemter is (i.e. does
            not account for the scaling)
        """
        return self.v1.get()
