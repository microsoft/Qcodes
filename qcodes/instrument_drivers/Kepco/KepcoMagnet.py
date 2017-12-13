from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
import qcodes as qc

from typing import Union


class KepcoMagnet(Instrument):
    MAX_AMP = 20
    """
    This is the qcodes driver for controlling the field using the Kepco BOP 20-20M.

    This is a virtual driver only and will not talk to your instrument.
    """

    def __init__(self,
                 name,
                 parameter,
                 max_field,
                 volt_to_amp,
                 field_to_amp,
                 axis,
                 rate=0.2,
                 **kwargs):
        super().__init__(name, **kwargs)

        self.v1 = parameter

        self.max_field = max_field
        self.volt_to_amp = volt_to_amp
        self.field_to_amp = field_to_amp
        self.axis = axis
        self.rate = rate

        delay = 0.1
        self.v1.inter_delay = delay
        self.v1.step = (
            self.rate * self.field_to_amp / self.volt_to_amp) * delay / 60

        self.v1.vals = Numbers(-self.MAX_AMP / volt_to_amp,
                               self.MAX_AMP / volt_to_amp)

        self.add_parameter(
            'minmax', label='Maximum field value', unit='T', vals=Numbers())

        self.add_parameter(
            '{}_BField'.format(axis),
            label='{} magnetic field'.format(axis),
            unit='T',
            get_cmd=self.get_field,
            set_cmd=self.set_field,
            vals=Numbers(-max_field, max_field))

    def set_field(self, value: Union[int, float]) -> None:
        instrument_value = value * self.field_to_amp / self.volt_to_amp
        self.v1.set(instrument_value)

    def get_field(self) -> Union[int, float]:
        """
        Returns:
            number: value at which was set at the sample
        """
        value = self.v1.get() / self.field_to_amp * self.volt_to_amp
        return value

    def get_idn(self):
        vendor = 'Ithaco (DL Instruments)'
        model = '1211'
        serial = None
        firmware = None
        return {
            'vendor': vendor,
            'model': model,
            'serial': serial,
            'firmware': firmware
        }
