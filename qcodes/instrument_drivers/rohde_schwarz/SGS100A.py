from qcodes import VisaInstrument, validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping


class RohdeSchwarz_SGS100A(VisaInstrument):
    """
    This is the qcodes driver for the Rohde & Schwarz SGS100A signal generator

    Status: beta-version.

    .. todo::

        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit

    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good idea to group all similar RS drivers together in one
    module.

    Tested working with

    - RS_SGS100A

    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='SOUR:FREQ?',
                           set_cmd='SOUR:FREQ {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(1e6, 20e9))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='SOUR:PHAS?',
                           set_cmd='SOUR:PHAS {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(0, 360))
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-120, 25))
        self.add_parameter('status',
                           label='RF Output',
                           get_cmd=':OUTP:STAT?',
                           set_cmd=':OUTP:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
        self.add_parameter('IQ_state',
                           label='IQ Modulation',
                           get_cmd=':IQ:STAT?',
                           set_cmd=':IQ:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
        self.add_parameter('pulsemod_state',
                           label='Pulse Modulation',
                           get_cmd=':SOUR:PULM:STAT?',
                           set_cmd=':SOUR:PULM:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
        self.add_parameter('pulsemod_source',
                           label='Pulse Modulation Source',
                           get_cmd='SOUR:PULM:SOUR?',
                           set_cmd='SOUR:PULM:SOUR {}',
                           vals=vals.Enum('INT', 'EXT', 'int', 'ext'))
        self.add_parameter('ref_osc_source',
                           label='Reference Oscillator Source',
                           get_cmd='SOUR:ROSC:SOUR?',
                           set_cmd='SOUR:ROSC:SOUR {}',
                           vals=vals.Enum('INT', 'EXT', 'int', 'ext'))
        # Define LO source INT/EXT (Only with K-90 option)
        self.add_parameter('LO_source',
                           label='Local Oscillator Source',
                           get_cmd='SOUR:LOSC:SOUR?',
                           set_cmd='SOUR:LOSC:SOUR {}',
                           vals=vals.Enum('INT', 'EXT', 'int', 'ext'))
        # Define output at REF/LO Output (Only with K-90 option)
        self.add_parameter('ref_LO_out',
                           label='REF/LO Output',
                           get_cmd='CONN:REFL:OUTP?',
                           set_cmd='CONN:REFL:OUTP {}',
                           vals=vals.Enum('REF', 'LO', 'OFF', 'ref', 'lo', 'off', 'Off'))
        # Frequency mw_source outputs when used as a reference
        self.add_parameter('ref_osc_output_freq',
                           label='Reference Oscillator Output Frequency',
                           get_cmd='SOUR:ROSC:OUTP:FREQ?',
                           set_cmd='SOUR:ROSC:OUTP:FREQ {}',
                           vals=vals.Enum('10MHz', '100MHz', '1000MHz'))
        # Frequency of the external reference mw_source uses
        self.add_parameter('ref_osc_external_freq',
                           label='Reference Oscillator External Frequency',
                           get_cmd='SOUR:ROSC:EXT:FREQ?',
                           set_cmd='SOUR:ROSC:EXT:FREQ {}',
                           vals=vals.Enum('10MHz', '100MHz', '1000MHz'))

        # IQ impairments
        self.add_parameter('IQ_impairments',
                           label='IQ Impairments',
                           get_cmd=':SOUR:IQ:IMP:STAT?',
                           set_cmd=':SOUR:IQ:IMP:STAT {}',
                           val_mapping=create_on_off_val_mapping(on_val='1',
                                                                 off_val='0'))
        self.add_parameter('I_offset',
                           label='I Offset',
                           get_cmd='SOUR:IQ:IMP:LEAK:I?',
                           set_cmd='SOUR:IQ:IMP:LEAK:I {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-10, 10))
        self.add_parameter('Q_offset',
                           label='Q Offset',
                           get_cmd='SOUR:IQ:IMP:LEAK:Q?',
                           set_cmd='SOUR:IQ:IMP:LEAK:Q {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-10, 10))
        self.add_parameter('IQ_gain_imbalance',
                           label='IQ Gain Imbalance',
                           get_cmd='SOUR:IQ:IMP:IQR?',
                           set_cmd='SOUR:IQ:IMP:IQR {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-1, 1))
        self.add_parameter('IQ_angle',
                           label='IQ Angle Offset',
                           get_cmd='SOUR:IQ:IMP:QUAD?',
                           set_cmd='SOUR:IQ:IMP:QUAD {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(-8, 8))

        self.add_function('reset', call_cmd='*RST')
        self.add_function('run_self_tests', call_cmd='*TST?')

        self.connect_message()

    def on(self):
        self.status('on')

    def off(self):
        self.status('off')
