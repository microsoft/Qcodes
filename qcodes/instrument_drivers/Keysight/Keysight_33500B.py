from qcodes import VisaInstrument, validators as vals
from pyvisa.errors import VisaIOError
import logging

log = logging.getLogger(__name__)


class Keysight_33500B(VisaInstrument):

    def __init__(self, name, address, **kwargs):

        super().__init__(name, address, **kwargs)

        # stupid helper functions
        def setcmd(channel, setting):
            return 'SOURce{}:'.format(channel) + setting + ' {}'

        def getcmd(channel, setting):
            return 'SOURce{}:'.format(channel) + setting + '?'

        for chan in [1, 2]:

            self.add_parameter('ch{}_function_type'.format(chan),
                               label='Channel {} function type'.format(chan),
                               set_cmd=setcmd(chan, 'FUNCtion'),
                               get_cmd=getcmd(chan, 'FUNCtion'),
                               vals=vals.Enum('SIN', 'SQU', 'TRI', 'RAMP',
                                              'PULS', 'PRBS', 'NOIS', 'ARB',
                                              'DC')
                               )

            self.add_parameter('ch{}_frequency_mode'.format(chan),
                               label='Channel {} frequency mode'.format(chan),
                               set_cmd=setcmd(chan, 'FREQuency:MODE'),
                               get_cmd=getcmd(chan, 'FREQuency:MODE'),
                               vals=vals.Enum('CW', 'LIST', 'SWEEP', 'FIXED')
                               )

            self.add_parameter('ch{}_frequency'.format(chan),
                               label='Channel {} frequency'.format(chan),
                               set_cmd=setcmd(chan, 'FREQuency'),
                               get_cmd=getcmd(chan, 'FREQUency'),
                               get_parser=float,
                               unit='Hz',
                               # TODO: max. freq. actually really tricky
                               vals=vals.Numbers(1e-6, 30e6)
                               )

            self.add_parameter('ch{}_amplitude_unit'.format(chan),
                               label='Channel {} amplitude unit'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage:UNIT'),
                               get_cmd=getcmd(chan, 'VOLTage:UNIT'),
                               vals=vals.Enum('VPP', 'VRMS', 'DBM')
                               )

            self.add_parameter('ch{}_amplitude'.format(chan),
                               label='Channel {} amplitude'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage'),
                               get_cmd=getcmd(chan, 'VOLTage'),
                               unit='',  # see amplitude_unit
                               get_parser=float)

            self.add_parameter('ch{}_offset'.format(chan),
                               label='Channel {} voltage offset'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage:OFFSet'),
                               get_cmd=getcmd(chan, 'VOLTage:OFFSet'),
                               unit='V',
                               get_parser=float
                               )
            self.add_parameter('ch{}_output'.format(chan),
                               label='Channel {} output state'.format(chan),
                               set_cmd='OUTPut{}'.format(chan) + ' {}',
                               get_cmd='OUTPut{}?'.format(chan),
                               val_mapping={'ON': 1, 'OFF': 0}
                               )

            self.add_parameter('ch{}_ramp_symmetry'.format(chan),
                               label='Channel {} ramp symmetry'.format(chan),
                               set_cmd=setcmd(chan, 'FUNCtion:RAMP:SYMMetry'),
                               get_cmd=getcmd(chan, 'FUNCtion:RAMP:SYMMetry'),
                               get_parser=float,
                               vals=vals.Numbers(0, 100)
                               )

        self.add_parameter('sync_source',
                           label='Source of sync function',
                           set_cmd='OUTPut:SYNC:SOURce {}',
                           get_cmd='OUTPut:SYNC:SOURce?',
                           val_mapping={1: 'CH1', 2: 'CH2'},
                           vals=vals.Enum(1, 2)
                           )

        self.add_parameter('sync_output',
                           label='Sync output state',
                           set_cmd='OUTPut:SYNC {}',
                           get_cmd='OUTPut:SYNC?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )
