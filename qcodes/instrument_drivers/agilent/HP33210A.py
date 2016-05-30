from qcodes import VisaInstrument, validators as vals


class Agilent_HP33210A(VisaInstrument):
    """
    This is the code for Agilent HP33210A function/arbitrary waveform generator






    """
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('IDN', get_cmd='*IDN?')
        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {:.3f} Hz',
                           get_parser=float,
                           vals=vals.Numbers(10e-3, 10e6))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           units='V',
                           get_cmd='VOLT?',
                           set_cmd='VOLT {:.3f} V',
                           get_parser=float,
                           vals=vals.Numbers(10e-3, 10))
        self.add_parameter(name='offset',
                           label='Offset',
                           units='V',
                           get_cmd='VOLT:OFFS?',
                           set_cmd='VOLT:OFFS {:.2f} V',
                           get_parser=float,
                           vals=vals.Numbers(-4.99, 4.99))
    # burst phase
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='BURS:PHAS?',
                           set_cmd='BURS:PHAS {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(0, 360))
        self.add_parameter(name='symmetry',
                           label='Symmetry',
                           units='%',
                           get_cmd='FUNC:RAMP:SYMM?',
                           set_cmd='FUNC:RAMP:SYMM {:.1f}',
                           get_parser=float,
                           vals=vals.Numbers(0, 100))
        self.add_parameter(name='duty',
                           label='Duty',
                           units='%',
                           get_cmd='FUNC:SQU:DCYC?',
                           set_cmd='FUNC:SQU:DCYC {:.1f}',
                           get_parser=float,
                           vals=vals.Numbers(0, 100))
        self.add_parameter(name='output_functions',
                           get_cmd='FUNC?',
                           set_cmd='FUNC {}',
                           vals=vals.Enum('SIN', 'SQU', 'RAMP',
                                          'PULS', 'NOIS', 'DC',
                                          'USER'))
        self.add_parameter(name='output_connector',
                           get_cmd='OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping={'OFF': '0', 'ON': '1'})
    # load impedence of the output
        self.add_parameter(name='load',
                           label='Load',
                           units='ohms',
                           get_cmd='OUTP:LOAD?',
                           set_cmd='OUTP:LOAD {:.1f}',
                           get_parser=float,
                           vals=vals.Numbers(1, 10e3))
    # invert the waveform relative to the offset voltage
        self.add_parameter(name='polarity',
                           get_cmd='OUTP:POL?',
                           set_cmd='OUTP:POL {}',
                           vals=vals.Enum('NORM', 'INV'))
    # disable/enable front-panel Sync connector
        self.add_parameter(name='sync_connector',
                           get_cmd='OUTP:SYNC?',
                           set_cmd='OUTP:SYNC {}',
                           val_mapping={'OFF': '0', 'ON': '1'})
        self.add_parameter(name='amplitude_units',
                           get_cmd='VOLT:UNIT?',
                           set_cmd='VOLT:UNIT {}',
                           vals=vals.Enum('VPP', 'VRMS', 'DBM'))
        self.add_parameter(name='volt_range',
                           get_cmd='VOLT:RANG:AUTO?',
                           set_cmd='VOLT:RANG:AUTO {}',
                           val_mapping={'OFF': '0', 'ON': '1'})
        self.add_parameter(name='volt_high',
                           get_cmd='VOLT:HIGH?',
                           set_cmd='VOLT:HIGH {:.3f}',
                           get_parser=float,
                           vals=vals.Numbers(10e-3, 10))
        self.add_parameter(name='volt_low',
                           get_cmd='VOLT:LOW?',
                           set_cmd='VOLT:LOW {:.3f}',
                           get_parser=float,
                           vals=vals.Numbers(10e-3, 10))
        self.connect_message('IDN')
