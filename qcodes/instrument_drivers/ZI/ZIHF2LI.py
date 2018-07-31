from .private.ZILI_generic import _ZILI_generic
import enum
from functools import partial

from qcodes.utils import validators as vals


class ZIHF2LI(_ZILI_generic):
    """
    QCoDeS driver for ZI HF2 Lockin.

    Currently implementing demodulator settings and the sweeper functionality.

    Requires ZI Lab One software to be installed on the computer running QCoDeS.
    Furthermore, the Data Server and Web Server must be running and a connection
    between the two must be made.

    """
    class DemodTrigger(enum.IntFlag):
        CONTINUOUS = 0
        DIO0_RISING = 1
        DIO0_FALLING = 2
        DIO1_RISING = 4
        DIO1_FALLING = 8
        DIO0_HIGH = 16
        DIO0_LOW = 32
        DIO1_HIGH = 64
        DIO1_LOW = 128


    def __init__(self, name: str, device_ID: str, **kwargs) -> None:
        super().__init__(name, device_ID, api_level=1, **kwargs)

        #TODO: Add check for MF option
        num_demod = 6
        num_osc = 2
        out_map = {1:6, 2:7}
        self._create_parameters(num_osc, num_demod, out_map)


        #Create HF2LI specific parameters

        ########################################
        # DEMODULATOR PARAMETERS

        for demod in range(1, num_demod+1):
            # val_mapping for the demodX_signalin parameter
            dmsigins = {'Signal input 0': 0,
                        'Signal input 1': 1,
                        'Aux Input 0': 2,
                        'Aux Input 1': 3,
                        'DIO 0': 4,
                        'DIO 1': 5}
            
            param = getattr(self, f'demod{demod}_signalin')
            param.val_mapping = dmsigins
            param.vals = vals.Enum(*list(dmsigins.keys()))

            param = getattr(self, f'demod{demod}_trigger')
            param.get_parser = ZIHF2LI.DemodTrigger
            param.set_parser = int


        ########################################
        # SIGNAL INPUTS

        for sigin in range(1, 3):
            param = getattr(self, f'signal_input{sigin}_range')
            param.vals = vals.Numbers(0.0001, 2)

            del self.parameters[f'signal_input{sigin}_scaling']
  
            param = getattr(self, f'signal_input{sigin}_diff')
            param.val_mapping = {'ON': 1, 'OFF': 0}
            param.vals = vals.Enum('ON', 'OFF')

        ########################################
        # SIGNAL OUTPUTS
        for sigout in range(1,3):
            del self.parameters[f'signal_output{sigout}_imp50']
            del self.parameters[f'signal_output{sigout}_autorange']

            param = getattr(self, f'signal_output{sigout}_range')
            param.vals = vals.Enum(0.01, 0.1, 1, 10)

            param = getattr(self, f'signal_output{sigout}_offset')
            param.vals = vals.Numbers(-1.0, 1.0)