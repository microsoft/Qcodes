from .private.ZILI_generic import _ZILI_generic
import enum
from functools import partial

from qcodes.utils import validators as vals


class ZIUHFLI(_ZILI_generic):
    """
    QCoDeS driver for ZI UHF Lockin.

    Currently implementing demodulator settings and the sweeper functionality.

    Requires ZI Lab One software to be installed on the computer running QCoDeS.
    Furthermore, the Data Server and Web Server must be running and a connection
    between the two must be made.

    """

    def __init__(self, name: str, device_ID: str, **kwargs) -> None:
        super().__init__(name, device_ID, api_level=5, **kwargs)


        #TODO: Add check for MF option
        num_demod = 8
        num_osc = 2
        out_map = {1:3, 2:7}
        self._create_parameters(num_osc, num_demod, out_map)


        #Create UHFLI specific parameters

        ########################################
        # DEMODULATOR PARAMETERS

        for demod in range(1, num_demod+1):
            # val_mapping for the demodX_signalin parameter
            dmsigins = {'Sig In 1': 0,
                        'Sig In 2': 1,
                        'Trigger 1': 2,
                        'Trigger 2': 3,
                        'Aux Out 1': 4,
                        'Aux Out 2': 5,
                        'Aux Out 3': 6,
                        'Aux Out 4': 7,
                        'Aux In 1': 8,
                        'Aux In 2': 9,
                        'Phi Demod 4': 10,
                        'Phi Demod 8': 11}
            param = getattr(self, f'demod{demod}_signalin')
            param.val_mapping = dmsigins
            param.vals = vals.Enum(*list(dmsigins.keys()))

            dmtrigs = {'Continuous': 0,
                       'Trigger in 3 Rise': 1,
                       'Trigger in 3 Fall': 2,
                       'Trigger in 3 Both': 3,
                       'Trigger in 3 High': 32,
                       'Trigger in 3 Low': 16,
                       'Trigger in 4 Rise': 4,
                       'Trigger in 4 Fall': 8,
                       'Trigger in 4 Both': 12,
                       'Trigger in 4 High': 128,
                       'Trigger in 4 Low': 64,
                       'Trigger in 3|4 Rise': 5,
                       'Trigger in 3|4 Fall': 10,
                       'Trigger in 3|4 Both': 15,
                       'Trigger in 3|4 High': 160,
                       'Trigger in 3|4 Low': 80}
            param = getattr(self, f'demod{demod}_trigger')
            param.val_mapping = dmtrigs
            param.vals = vals.Enum(*list(dmtrigs.keys()))

        ########################################
        # SIGNAL INPUTS

        for sigin in range(1, 3):
            #TODO: find the right range of ranges for UHF!
            param = getattr(self, f'signal_input{sigin}_range')
            param.vals = vals.Numbers(0.0001, 2)

            sigindiffs = {'Off': 0, 'Inverted': 1, 'Input 1 - Input 2': 2,
                          'Input 2 - Input 1': 3}         
            param = getattr(self, f'signal_input{sigin}_diff')
            param.val_mapping = sigindiffs
            param.vals = vals.Enum(*list(sigindiffs.keys()))

        ########################################
        # SIGNAL OUTPUTS
        for sigout in range(1,3):
            param = getattr(self, f'signal_output{sigout}_offset')
            param.vals = vals.Numbers(-1.5, 1.5)