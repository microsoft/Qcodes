import numpy as np
from qcodes import VisaInstrument, validators as vals
from qcodes.utils.validators import Numbers
from qcodes.utils.helpers import create_on_off_val_mapping

from qcodes.utils.deprecate import deprecate_moved_to_qcd

def parse_on_off(stat):
    if stat.startswith('0'):
        stat = 'Off'
    elif stat.startswith('1'):
        stat = 'On'
    return stat


frequency_mode_docstring = """
This command sets the frequency mode of the signal generator.

*FIXed* and *CW* These choices are synonymous. Any currently running frequency sweeps are
turned off, and the current CW frequency settings are used to control the output
frequency.

*SWEep* The effects of this choice are determined by the sweep generation type selected. In analog sweep generation, the ramp
sweep frequency settings (start, stop, center, and span) control the output
frequency. In step sweep generation, the current step sweep frequency settings
control the output frequency. In both cases, this selection also activates the sweep.
This choice is available with Option 007 only.

*LIST* This choice selects the swept frequency mode. If sweep triggering is set to
immediate along with continuous sweep mode, executing the command starts the LIST or STEP frequency sweep. 
"""

IQsource_docstring = """
This command selects the I/Q modulator source for one of the two possible paths.

*EXTernal* This choice selects an external 50 ohm source as the I/Q input to I/Q modulator.
*INTernal* This choice is for backward compatibility with ESG E44xxB models and performs
the same function as the BBG1 selection.
*BBG1* This choice selects the baseband generator as the source for the I/Q modulator.
*EXT600* This choice selects a 600 ohm impedance for the I and Q input connectors and
routes the applied signals to the I/Q modulator.
*OFF* This choice disables the I/Q input.
"""


deprecate_moved_to_qcd(alternative="qcodes_contrib_drivers.drivers.Keysight.Keysight_E8267D.Keysight_E8267D")
class Keysight_E8267D(VisaInstrument):
    """
    This is the qcodes driver for the Keysight_E8267D signal generator

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual

    This driver will most likely work for multiple Agilent sources.

    This driver does not contain all commands available for the E8527D but
    only the ones most commonly used.
    """

    def __init__(self, name, address, step_attenuator=False, **kwargs):
        super().__init__(name, address, **kwargs)

        # Only listed most common spellings idealy want a
        # .upper val for Enum or string
        on_off_validator = vals.Enum('on', 'On', 'ON',
                                     'off', 'Off', 'OFF')
        on_off_mapping = create_on_off_val_mapping(1, 0)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ:CW?',
                           set_cmd='FREQ:CW' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(1e5, 20e9),
                           docstring='Adjust the RF output frequency')
        self.add_parameter(name='frequency_offset',
                           label='Frequency offset',
                           unit='Hz',
                           get_cmd='FREQ:OFFS?',
                           set_cmd='FREQ:OFFS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-200e9,
                                        max_value=200e9))
        self.add_parameter('frequency_mode',
                           label='Frequency mode',
                           set_cmd='FREQ:MODE {}',
                           get_cmd='FREQ:MODE?',
                           get_parser=lambda s: s.strip(),
                           vals=vals.Enum('FIX', 'CW', 'SWE', 'LIST'))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHASE?',
                           set_cmd='PHASE' + ' {:.8f}',
                           get_parser=self.rad_to_deg,
                           set_parser=self.deg_to_rad,
                           vals=vals.Numbers(-180, 180))
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='POW:AMPL?',
                           set_cmd='POW:AMPL' + ' {:.4f}',
                           get_parser=float,
                           set_parser=float,
                           vals=vals.Numbers(-130, 25))
        self.add_parameter('status',
                           get_cmd=':OUTP?',
                           set_cmd='OUTP {}',
                           get_parser=parse_on_off,
                           vals=on_off_validator)
        self.add_parameter(name='modulation_rf_enabled',
                           get_cmd='OUTP:MOD?',
                           set_cmd='OUTP:MOD {}',
                           val_mapping=on_off_mapping)
        self.add_parameter('IQmodulator_enabled',
                           get_cmd='DM:STATe?',
                           set_cmd='DM:STATe {}',
                           val_mapping=on_off_mapping,
                           docstring='Enables or disables the internal I/Q modulator. Source can be external or internal.')

        for source in [1, 2]:
            self.add_parameter(f'IQsource{source}',
                               get_cmd=f'DM:SOUR{source}?',
                               set_cmd=f'DM:SOUR{source} {{}}',
                               get_parser=lambda s: s.strip(),
                               vals=vals.Enum('OFF', 'EXT', 'EXT600', 'INT'),
                               docstring=IQsource_docstring)

        self.add_parameter(f'IQadjustments_enabled', get_cmd=f'DM:IQAD?', set_cmd=f'DM:IQAD {{}}', val_mapping=on_off_mapping, docstring='Enable or disable IQ adjustments')

        IQoffset_parameters = dict(get_parser=float, set_parser=float, vals=vals.Numbers(-100,100))
        self.add_parameter(f'I_offset', get_cmd=f'DM:IQAD:IOFF?', set_cmd=f'DM:IQAD:IOFF {{}}', **IQoffset_parameters, docstring='I channel offset in percentage')
        self.add_parameter(f'Q_offset', get_cmd=f'DM:IQAD:QOFF?', set_cmd=f'DM:IQAD:QOFF {{}}',  **IQoffset_parameters, docstring='Q channel offset in percentage')
        self.add_parameter(f'IQ_quadrature', get_cmd=f'DM:IQAD:QSK?', set_cmd=f'DM:IQAD:QSK {{}}', get_parser=float, set_parser=float, docstring='IQ quadrature offset', unit='deg')

        self.add_parameter(f'pulse_modulation_enabled', get_cmd=f'PULM:STATe?', set_cmd=f'PULM:STATe {{}}', val_mapping=on_off_mapping, docstring='Enable or disable pulse modulation path')
        self.add_parameter(f'pulse_modulation_source', get_cmd=f'PULM:SOURce?', set_cmd=f'PULM:SOURce {{}}', get_parser=lambda s: s.strip(), vals=vals.Enum('EXT', 'INT', 'SCAL'))

        self.add_parameter(f'wideband_amplitude_modulation_enabled', get_cmd=f'AM:WID:STATe?', set_cmd=f'AM:WID:STATe {{}}', val_mapping=on_off_mapping, docstring='This command enables or disables wideband amplitude modulation')

        self.connect_message()

    def on(self):
        self.set('status', 'on')

    def off(self):
        self.set('status', 'off')

    @staticmethod
    def deg_to_rad(angle_deg):
        return np.deg2rad(float(angle_deg))

    @staticmethod
    def rad_to_deg(angle_rad):
        return np.rad2deg(float(angle_rad))
