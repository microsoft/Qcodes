from typing import Any

from qcodes import VisaInstrument, validators as vals


class SRS_SG384(VisaInstrument):
    """
    This is the code for SG384 RF Signal Generator
    Status: beta version
    Includes the essential commands from the manual
    """
    def __init__(self, name: str, address: str,
                 reset: bool = False, **kwargs: Any):
        super().__init__(name, address, terminator='\n', **kwargs)
    # signal synthesis commands
        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {:.3f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=9.5e5,
                                             max_value=4.05e9))
        self.add_parameter(name='phase',
                           label='Carrier phase',
                           unit='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {:.1f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-360, max_value=360))
        self.add_parameter(name='amplitude_LF',
                           label='Power of BNC output',
                           unit='dBm',
                           get_cmd='AMPL?',
                           set_cmd='AMPL {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-47, max_value=13))
        self.add_parameter(name='amplitude_RF',
                           label='Power of type-N RF output',
                           unit='dBm',
                           get_cmd='AMPR?',
                           set_cmd='AMPR {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-110, max_value=16.5))
        self.add_parameter(name='amplitude_HF',
                           label='Power of RF doubler output',
                           unit='dBm',
                           get_cmd='AMPH?',
                           set_cmd='AMPH {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-10, max_value=13))
        self.add_parameter(name='amplitude_clock',
                           label='Rear clock output',
                           unit='Vpp',
                           get_cmd='AMPC?',
                           set_cmd='AMPC {:.2f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.4, max_value=1.00))
        self.add_parameter(name='noise_mode',
                           label='RF PLL loop filter mode',
                           get_cmd='NOIS?',
                           set_cmd='NOIS {}',
                           val_mapping={'Mode 1': 0,
                                        'Mode 2': 1})
        self.add_parameter(name='enable_RF',
                           label='Type-N RF output',
                           get_cmd='ENBR?',
                           set_cmd='ENBR {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='enable_LF',
                           label='BNC output',
                           get_cmd='ENBL?',
                           set_cmd='ENBL {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='enable_HF',
                           label='RF doubler output',
                           get_cmd='ENBH?',
                           set_cmd='ENBH {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='enable_clock',
                           label='Rear clock output',
                           get_cmd='ENBC?',
                           set_cmd='ENBC {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='offset_clock',
                           label='Rear clock offset voltage',
                           unit='V',
                           get_cmd='OFSC?',
                           set_cmd='OFSC {}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-2, max_value=2))
        self.add_parameter(name='offset_rearDC',
                           label='Rear DC offset voltage',
                           unit='V',
                           get_cmd='OFSD?',
                           set_cmd='OFSD {}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-10, max_value=10))
        self.add_parameter(name='offset_bnc',
                           label='Low frequency BNC output',
                           unit='V',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=-1.5, max_value=1.5))
    # Modulation commands
        self.add_parameter(name='modulation_coupling',
                           label='External modulation input coupling',
                           get_cmd='COUP?',
                           set_cmd='COUP {}',
                           val_mapping={'AC': 0,
                                        'DC': 1})
        self.add_parameter(name='FM_deviation',
                           label='Frequency modulation deviation',
                           unit='Hz',
                           get_cmd='FDEV?',
                           set_cmd='FDEV {:.1f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=0.1, max_value=32e6))
        self.add_parameter(name='modulation_function',
                           label='Modulation function for AM/FM/PhiM',
                           get_cmd='MFNC?',
                           set_cmd='MFNC {}',
                           val_mapping={'Sine': 0,
                                        'Ramp': 1,
                                        'Triangle': 2,
                                        'Square': 3,
                                        'Noise': 4,
                                        'External': 5})
        self.add_parameter(name='enable_modulation',
                           get_cmd='MODL?',
                           set_cmd='MODL {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='modulation_rate',
                           get_cmd='RATE?',
                           set_cmd='RATE {:.6f}',
                           get_parser=float,
                           vals=vals.Numbers(min_value=1e-6, max_value=50e3))
        self.add_parameter(name='modulation_type',
                           label='Current modulation type',
                           get_cmd='TYPE?',
                           set_cmd='TYPE {}',
                           val_mapping={'AM': 0,
                                        'FM': 1,
                                        'Phi': 2,
                                        'Sweep': 3,
                                        'Pulse': 4,
                                        'Blank': 5,
                                        'IQ': 6})
        self.connect_message()
