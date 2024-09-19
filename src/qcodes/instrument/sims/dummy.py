# a driver to interact with the dummy.yaml simulated instrument
#
# dummy = Dummy('dummy', 'GPIB::8::INSTR', visalib='dummy.yaml@sim',
#               terminator='\n', device_clear=False)
from typing import Any

import qcodes.validators as vals
from qcodes.instrument.visa import VisaInstrument


class Dummy(VisaInstrument):
    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address, **kwargs)

        self.connect_message()

        self.add_parameter(
            "frequency",
            set_cmd="FREQ {}",
            get_cmd="FREQ?",
            unit="Hz",
            get_parser=float,
            vals=vals.Numbers(10, 1000),
        )
