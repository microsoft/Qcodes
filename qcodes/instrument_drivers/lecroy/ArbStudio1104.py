import pythonnet
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
NET.addAssembly('ArbStudioSDK.dll')

class ArbStudio1104(Instrument):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('core_clock',
                           label='Core clock',
                           units='MHz',
                           set_cmd=api.pb_core_clock,
                           vals=Numbers(0, 500))
