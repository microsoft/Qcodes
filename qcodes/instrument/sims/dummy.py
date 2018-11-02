# a driver to interact with the dummy.yaml simulated instrument
#
# dummy = Dummy('dummy', 'GPIB::8::INSTR', visalib='dummy.yaml@sim',
#               terminator='\n', device_clear=False)

from qcodes.instrument.visa import VisaInstrument
import qcodes.utils.validators as vals


class Dummy(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        self.connect_message()

        self.add_parameter('frequency',
                           set_cmd='FREQ {}',
                           get_cmd='FREQ?',
                           unit='Hz',
                           get_parser=float,
                           vals=vals.Numbers(10, 1000))
