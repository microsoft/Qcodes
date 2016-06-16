from numpy import pi

from qcodes import VisaInstrument, validators as vals


class SRS_SR830(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research Systems SR830
    Lock-in Amplifier

    Status: beta-version.
        TODO: Add all parameters that are in the manual
        TODO: Complete getters, setters and validators for ranges and time constants
        TODO: Complete auto-offset operation
"""
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        # Identify
        self.add_parameter('IDN', get_cmd='*IDN?')

        # Source Parameters
        self.add_parameter(name='frequency',
                           label='Frequency',
                           units='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(0.001, 102e3))
        self.add_parameter(name='phase',
                           label='Phase',
                           units='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {:.2f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-180.0, 180.0))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           units='V',
                           get_cmd='SLVL?',
                           set_cmd='SLVL {:.3f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(0.004,5.000))
        self.add_parameter(name='harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           set_cmd='HARM {:d}',
                           get_parser=int,
                           set_parser=int,
                           vals=vals.Ints(min_value=0))

        # Input Parameters
        ISRC = {'A': 0, 'A-B': 1, 'I': 2, 'I100': 3}
        self.add_parameter(name='input_source',
                           label='Input Source',
                           get_cmd='ISRC?',
                           set_cmd='ISRC {:d}',
                           val_mapping=ISRC)
        IGND = {'Float': 0, 'Ground': 1}
        self.add_parameter(name='input_shield',
                           label='Input Shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {:d}',
                           val_mapping=IGND)
        ICPL = {'AC': 0, 'DC': 1}
        self.add_parameter(name='input_couple',
                           label='Input Couple',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {:d}',
                           val_mapping=ICPL)
        ILIN = {'Off': 0, 'Line': 1, '2xLine': 2, 'Both': 3}
        self.add_parameter(name='input_filter',
                           label='Input Filter',
                           get_cmd='ILIN?',
                           set_cmd='ILIN {:d}',
                           val_mapping=ILIN)

        self.add_parameter(name='sensitivity',
                           label='Sensitivity',
                           get_cmd='SENS?',
                           set_cmd='SENS {:d}',
                           get_parser=self.get_sensitivity,
                           set_parser=self.set_sensitivity,
                           vals=vals.MultiType(vals.Enum(*self.VOLT_SENS.keys()),
                                               vals.Enum(*self.CURR_SENS.keys()),
                                               vals.Numbers(2e-15, 1)))
        RMOD = {'High_Reserve': 0, 'Normal': 1, 'Low_Noise': 2}
        self.add_parameter(name='Reserve',
                           label='Reserve',
                           get_cmd='RMOD?',
                           set_cmd='RMOD {:d}',
                           val_mapping=RMOD)
        self.add_parameter(name='time_constant',
                           label='Time Constant',
                           units='s',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {:d}',
                           get_parser=self.get_tc,
                           set_parser=self.set_tc,
                           vals=vals.Enum(*self.TC_STR.keys()))
        OFSL = {6: 0, 12: 1, 18: 2, 24: 3}
        self.add_parameter(name='filter_slope',
                           label='Filter Slope',
                           units='dB',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {:d}',
                           val_mapping=OFSL)
        self.add_parameter(name="sync_filter",
                           label="Sync Filter",
                           get_cmd="SYNC?",
                           set_cmd="SYNC {:d}",
                           get_parser=self.get_bool,
                           set_parser=int,
                           vals=vals.Bool())

        # TODO: Add auxilliary output channels
        # TODO: Add auxilliary input channels
        # TODO: Add display settings

        # Measurement Parameters
        self.add_parameter(name='X',
                           label='X',
                           get_cmd='OUTP ? 1',
                           get_parser=float)
        self.add_parameter(name='Y',
                           label='Y',
                           get_cmd='OUTP ? 2',
                           get_parser=float)
        self.add_parameter(name='R',
                           label='R',
                           get_cmd='OUTP ? 3',
                           get_parser=float)
        self.add_parameter(name='theta',
                           label='Phase',
                           get_cmd='OUTP ? 4',
                           get_parser=float)

        self.connect_message('IDN')

    def get_bool(self, state):
        '''
        Map instrument returned state to On/Off
        '''
        if state.startswith('0'):
            state = True
        elif state.startswith('1'):
            state = False
        return state

    # TODO: Write sensitivity getters and setters and validators
    VOLT_SENS = {'2nV': 0, '5nV': 1, '10nV': 2, '20nV': 3, '50nV': 4,
                 '100nV': 5, '200nV': 6, '500nV': 7, '1μV': 8, '2μV': 9,
                 '5μV': 10, '10μV': 11, '20μV': 12, '50μV': 13, '100μV': 14,
                 '200μV': 15, '500μV': 16, '1mV': 17, '2mV': 18, '5mV': 19,
                 '10mV': 20, '20mV': 21, '50mV': 22, '100mV': 23, '200mV': 24,
                 '500mV': 25, '1V': 26}
    RVOLT_SENS = {v: k for k, v in VOLT_SENS.items()}
    CURR_SENS = {'2fA': 0, '5fA': 1, '10fA': 2, '20fA': 3, '50fA': 4,
                 '100fA': 5, '200fA': 6, '500fA': 7, '1pA': 8, '2pA': 9,
                 '5pA': 10, '10pA': 11, '20pA': 12, '50pA': 13, '100pA': 14,
                 '200pA': 15, '500pA': 16, '1nA': 17, '2nA': 18, '5nA': 19,
                 '10nA': 20, '20nA': 21, '50nA': 22, '100nA': 23, '200nA': 24,
                 '500nA': 25, '1μA': 26}
    RCURR_SENS = {v: k for k, v in CURR_SENS.items()}
    def get_sensitivity(self, val):
        try:
            val = int(val)
        except ValueError:
            raise ValueError("Failed to parse sensitivity: {!r}".format(val))
        if self.input_source() in ['A', 'A-B']: # Check whether we are looking at voltages or currents
            return self.RVOLT_SENS[val]
        else:
            return self.RCURR_SENS[val]

    def set_sensitivity(self, val):
        # If we have been passed a string, look up the code corresponding to the range in
        # the map
        if isinstance(val, str):
            # Map u to μ
            val = val.replace('u', 'μ')
            if self.input_source() in ['A', 'A-B']:
                return self.VOLT_SENS[val]
            else:
                return self.CURR_SENS[val]
        elif isinstance(val, float) or isinstance(val, int):
            # TODO: Implement numeric handling of sensitivity
            pass
        raise ValueError("Invalid sensitivity")

    TC_STR = {10e-6: 0, 30e-6: 1, 100e-6: 2, 300e-6: 3, 1e-3: 4,
              3e-3: 5, 10e-3: 6, 30e-3: 7, 100e-3: 8, 300e-3: 9,
              1.0: 10, 3.0: 11, 10.0: 12, 30.0: 13, 100.0: 14,
              300.0: 15, 1e3: 16, 3e3: 17, 10e3: 18, 30e3: 19}
    RTC_STR = {v: k for k, v in TC_STR.items()}
    def get_tc(self, val):
        try:
            val = int(val)
        except ValueError:
            raise ValueError("Failed to parse time constant: {!r}".format(val))
        return self.RTC_STR[val]

    def set_tc(self, val):
        val = float(val) # Convert seconds to float
        return self.TC_STR[val]

    def auto_gain(self):
        """
        Perform an auto-gain operation"
        """
        self.write("AGAN")

    def auto_reserve(self):
        """
        Perform an auto-reserve operation
        """
        self.write("ARSV")

    def auto_phase(self):
        """
        Perform an auto-phase operation
        """
        self.write("APHS")
    # TODO: Figure out syntax for auto-offset command