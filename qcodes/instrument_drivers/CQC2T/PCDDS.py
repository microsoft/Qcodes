from qcodes.instrument_drivers.Keysight.M3300A import Keysight_M3300A_FPGA
from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Bool, Ints

import struct
import numpy as np

try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')


class PCDDS(Instrument):
    """
    This class is the driver for the Phase Coherent Pulse Generation Module implemented on the FPGA onboard a Keysight
    PXI AWG card
    """
    def __init__(self, name, **kwargs):
        """ Constructor for the pulse generation modules """
        super().__init__(name, **kwargs)
        self.fpga = Keysight_M3300A_FPGA('FPGA')
        self.port = 0
        self.n_pointer_bits = 9
        self.n_op_bits = 10
        self.n_phase_bits = 45
        self.n_accum_bits = 45
        self.n_amp_bits = 45
        self.n_dac_bits = 16
        self.clk = 100e6
        self.v_max = 3.0
        self.f_max = 200e6

        self.add_parameter(
            'output_enable',
            set_cmd=self._set_output_enable,
            vals=Bool(),
            docstring='Whether the system has an enabled output'
        )

        self.add_parameter(
            'load_delay',
            set_cmd=self._set_load_delay,
            vals=Ints(0, 15),
            docstring='How long the delay should be during loading a new pulse to calculate the new coefficients'
        )

    def _set_output_enable(self, output_enable):
        """
        Set the output enable state of the module
        :param output_enable: (Bool) Is the output enabled
        :return: None
        """
        operation = int('0010000000', 2)
        if output_enable:
            operation += int('0001000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.port, instr, 0, 0, 1)

    def _set_load_delay(self, delay):
        """
        Set the delay that the system will apply to calculate all the phase and frequency coefficiences
        :param delay: (Int) Delay in clock cycles
        :return: None
        """
        operation = int('0000010000', 2) + delay
        # Construct and send instruction
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.port, instr, 0, 0, 1)

    def construct_instruction(self, operation, pointer):
        """
        Function to construct the int instruction packet from the operation and pointer
        :param operation: (Int) Operation that we want to do
        :param pointer: (Int) ID of the pulse that this instruction refers to
        :return: (Int) Instruction
        """
        # Check that the pointer is in the allowed range
        if pointer >= 2**self.n_pointer_bits:
            raise ValueError(
                'Pointer with value {} is outside of bounds [{}, {}]'.format(pointer, 0, 2**self.n_pointer_bits-1))
        # Convert and return
        return (operation << 22) + pointer

    def reset(self):
        """ Sends the reset signal to the FPGA """
        self.fpga.reset(reset_mode=keysightSD1.SD_ResetMode.RESET_PULSE)

    def write_pulse(self, pulse, phase, frequency, amplitude, next_pulse):
        """
        Write a normal sinusoidal pulse with the desired properties to the pulse memory.
        :param pulse: (Int) The location in pulse memory that this is to be written to
        :param phase: (Float) The phase of the signal in degrees
        :param frequency: (Float) The frequency of the signal in Hz
        :param amplitude: (Float) The desired output maximum amplitude in V
        :param next_pulse: (Int) The next pulse that the system is to go to after this one
        :return: None
        """
        phase_val = self.phase2val(phase)
        freq_val = self.freq2val(frequency)
        accum = 0
        amplitude_val = self.amp2val(amplitude)

    def write_dc_pulse(self):
        pass

    def write_chirp_pulse(self):
        pass

    def set_next_pulse(self):
        pass

    def send_trigger(self):
        pass

    def phase2val(self, phase):
        """
        Function to calculate the correct phase register values for a given phase
        :param phase: (Float) The desired phase in degrees.
        :return: (Int) The register value for the desired phase
        """
        # TODO: Add angle wrapping
        return int(np.round((2 ** self.n_phase_bits / 360.0) * phase))

    def freq2val(self, freq):
        """
        Function to calculate the correct frequency register values for a given frequency
        :param freq: (Float) The desired frequency in Hz
        :return: (Int) The register value for the desired frequency
        """
        if freq > self.f_max or freq < 0:
            raise ValueError('Frequency of {} is outside of allowed values [0, {}MHz]'.format(freq, self.f_max/1e6))
        return int(np.round((2 ** self.n_phase_bits / (5*self.clk)) * freq))

    def accum2val(self, accum):
        """
        Function to calculate the correct accumulation register values for a given accumulation
        :param accum: (Float) The desired accumulation in Hz/s
        :return: (Int) The register value for the desired accumulation
        """
        if accum < 0 or accum > (5*self.clk ** 2):
            raise ValueError('Frequency Accumulation of {} is outside of allowed values [0,{}Hz/s]'.format(
                accum, (5*self.clk ** 2)))
        return int(np.round(accum * 2 ** self.n_accum_bits / (5*self.clk) ** 2))

    def amp2val(self, amp):
        """
        Function to calculate the correct amplitude register values for a given amplitude
        :param amp: (Float) The desired amplitude in V
        :return: (Int) The register value for the desired amplitude
        """
        if amp < 0 or amp > self.v_max:
            raise ValueError('Amplitude of {} is outside of allowed values [0, {}V'.format(amp, self.v_max))
        return int(np.round(2**self.n_dac_bits * amp/self.v_max))
