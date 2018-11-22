from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Bool, Ints
from typing import List


import numpy as np

try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')

model_channels = {'M3201A': 4,
                  'M3300A': 4}

"""
Instruction Code Syntax:

Each 32-bit instruction code packet is split into a 10-bit command, 9-bit 
pulse pointer and 13 bits of unused space. They are currently laid out as 
follows:
[(10 bit) Command, (13 bit) unused, (9 bit) pulse pointer]

Within the command, we have the following command breakdown structure

Bit 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 | Instruction
-------------------------------------------------------------------------------
    1 | x | x | x | x | x | x | x | x | x | Software Trigger
    x | 1 | x | x | x | x | x | x | x | s | Set pcdds_enable to s
    x | x | 1 | a | x | x | x | x | x | x | Set output_enable to s
    x | 0 | x | x | 1 | x | x | x | x | x | Write pulse
    x | 0 | x | x | 0 | 1 | a | b | c | d | Set load_delay to abcd
    x | 0 | x | x | 0 | 0 | x | x | 1 | x | Set next pulse

"""


class PCDDSChannel(InstrumentChannel):
    """
    Class that holds all the actions related to a specific PCDDS channel
    """
    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.fpga = self._parent.fpga
        self.id = id
        self.n_pointer_bits = 9
        self.n_op_bits = 10
        self.n_phase_bits = 45
        self.n_accum_bits = 45
        self.n_amp_bits = 16
        self.clk = 100e6
        self.v_max = 1.5
        self.f_max = 200e6
        self.fpga_port = 1

        self.add_parameter(
            'output_enable',
            label=f'ch{self.id} output_enable',
            set_cmd=self._set_output_enable,
            vals=Bool(),
            docstring='Whether the system has an enabled output'
        )

        self.add_parameter(
            'load_delay',
            label=f'ch{self.id} load_delay',
            set_cmd=self._set_load_delay,
            vals=Ints(0, 15),
            docstring='How long the delay should be during loading a new pulse '
                      'to calculate the new coefficients (delay in samples).'
        )

        self.add_parameter(
            'pcdds_enable',
            label=f'ch{self.id} pcdds_enable',
            set_cmd=self._set_pcdds_enable,
            vals=Bool(),
            docstring='Set the output of the device to use the PCDDS system '
                      'and not the inbuilt functionality'
        )

        self.add_parameter(
            'instruction_sequence',
            label=f'ch{self.id} instruction sequence of pulses',
            initial_value=[],
            docstring='The instruction sequence for all pulses sent to the DDS.'
                      'Gets cleared during clear_memory()'
        )

        # Initially set load delay to 10 samples
        self.load_delay(10)

    def _set_output_enable(self, output_enable):
        """
        Set the output enable state of the module
        Args:
            output_enable: (Bool) Is the output enabled
        """
        operation = int('0010000000', 2)
        if output_enable:
            operation += int('0001000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def _set_pcdds_enable(self, pcdds_enable: bool):
        """
        Select the output between the PCDDS system and the passthrough
        connection
        Args:
            pcdds_enable: If true, select the PCDDS system. Otherwise
                          use passthrough
        """
        operation = int('0100000000', 2)
        if pcdds_enable:
            operation += int('0000000001', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def _set_load_delay(self, delay: int):
        """
        Set the delay that the system will apply to calculate all the phase
        and frequency coefficients
        Args:
            delay: Delay in clock cycles
        """
        operation = int('0000010000', 2) + delay
        # Construct and send instruction
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def construct_instruction(self, operation: int, pointer: int) -> int:
        """
        Function to construct the int instruction packet from the operation and
        pointer
        Args:
            operation: Operation that we want to do
            pointer: ID of the pulse that this instruction refers to

        Returns: Instruction

        """
        # Check that the pointer is in the allowed range
        assert pointer < 2**self.n_pointer_bits, \
            f'Pointer with value {pointer} is outside of bounds ' \
            f'[0, {2**self.n_pointer_bits-1}]'
        # Convert and return
        return (operation << 22) + pointer

    def write_zero_pulse(self, pulse: int):
        """
        Function to write all zeros to a specific memory location
        Args:
            pulse: The location where the zeros are to be written
        """
        self.write_pulse(pulse=pulse, phase=0, frequency=0,
                         frequency_accumulation=0, amplitude=0, offset=0,
                         next_pulse=0)

    def clear_memory(self):
        """
        Function to clear all memory
        """
        for i in np.arange(2**self.n_pointer_bits):
            self.write_zero_pulse(pulse=i)
        self.instruction_sequence().clear()

    def write_instr(self, instr: dict):
        """
        Function to write an instruction to the pulse memory.

        It takes a dictionary with the following entries:
        - instr: The instruction type. Valid instruction types are 'dc',
                 'sine' and 'chirp'
        - pulse_idx: The pulse id that this instruction is to be stored in
        - phase: The phase of the relevant pulse (in degrees).
            Only applicable to 'sine' and 'chirp' pulses
        - freq: The frequency of this pulse. Only applicable to 'sine' and
                'chirp' pulses
        - accum: The frequency accumulation rate (in Hz/s).
            Only applicable to 'chirp' pulses
        - amp: The amplitude of the pulse
        - next_pulse: The next pulse that should be played after this one
        Args:
            instr: The instruction to write to memory
        """
        if instr['instr'] == 'dc':
            self.write_dc_pulse(pulse=instr['pulse_idx'], voltage=instr['amp'],
                                next_pulse=instr['next_pulse'])
        elif instr['instr'] == 'sine':
            self.write_sine_pulse(pulse=instr['pulse_idx'],
                                  phase=instr['phase'],
                                  frequency=instr['freq'],
                                  amplitude=instr['amp'],
                                  offset=instr['offset'],
                                  next_pulse=instr['next_pulse'])
        elif instr['instr'] == 'chirp':
            self.write_chirp_pulse(pulse=instr['pulse_idx'],
                                   phase=instr['phase'],
                                   frequency=instr['freq'],
                                   frequency_accumulation=instr['accum'],
                                   amplitude=instr['amp'],
                                   offset=instr['offset'],
                                   next_pulse=instr['next_pulse'])
        else:
            raise ValueError(f'Unknown instruction type: {instr["instr"]}')

        self.instruction_sequence().append(instr)

    def write_sine_pulse(self, pulse: int, phase: float, frequency: float,
                         amplitude: float, offset: float, next_pulse: int):
        """
        Write a normal sinusoidal pulse with the desired properties to the
        pulse memory.
        Args:
            pulse: The location in pulse memory that this is to be written to
            phase: The phase of the signal in degrees
            frequency: The frequency of the signal in Hz
            amplitude: The desired output maximum amplitude in V
            offset: The desired pulse offset in V
            next_pulse: The next pulse that the system is to go to after this
                        one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. It should '
                            'be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. It '
                            'should be an int')
        assert np.abs(offset) + np.abs(amplitude) <= self.v_max + 1e-10, \
            'The combined output will be larger than the maximum output voltage'

        # Convert all the pulse parameters to the correct register values
        phase_val = self.phase2val(phase)
        freq_val = self.freq2val(frequency)
        accum_val = 0
        amplitude_val = self.amp2val(amplitude)
        offset_val = self.offset2val(offset)
        self.write_pulse(pulse, phase_val, freq_val, accum_val,
                         amplitude_val, offset_val, next_pulse)

    def write_dc_pulse(self, pulse: int, voltage: float, next_pulse: int):
        """
        Write a DC pulse to memory. This sets up a pulse with a phase offset
        of 90 or 270 degrees, 0 frequency and a certain amplitude
        Args:
            pulse: The location in pulse memory that this is to be
                   written to
            voltage: The desired DC voltage
            next_pulse: The next pulse that the system is to go to after this
                        one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. It '
                            'should be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. '
                            'It should be an int')
        # Convert the voltage to the correct register value
        offset_val = self.offset2val(voltage)
        amplitude_val = 0
        phase_val = 0
        accum_val = 0
        freq_val = 0
        self.write_pulse(pulse, phase_val, freq_val, accum_val,
                         amplitude_val, offset_val, next_pulse)

    def write_chirp_pulse(self, pulse: int, phase: float, frequency: float,
                          frequency_accumulation: float, amplitude: float,
                          offset: float, next_pulse: int):
        """
        Write a pulse to pulse memory which contains a frequency sweep
        Args:
            pulse: The location in pulse memory that this is to be written to
            phase: The phase value for this pulse in degrees
            frequency: The frequency value for this pulse in Hz
            frequency_accumulation: The frequency accumulation for this pulse
                                    in Hz/s
            amplitude: The amplitude for this pulse in V
            offset: The offset for this pulse in V
            next_pulse: The pulse that the system is to go to after this one
        """
        if not isinstance(pulse, int):
            raise TypeError('Incorrect type for function input pulse. '
                            'It should be an int')
        if not isinstance(next_pulse, int):
            raise TypeError('Incorrect type for function input next_pulse. '
                            'It should be an int')
        assert np.abs(offset) + np.abs(amplitude) <= self.v_max + 1e-10, \
            'The combined output will be larger than the maximum output voltage'

        phase_val = self.phase2val(phase)
        freq_val = self.freq2val(frequency)
        accum_val = self.accum2val(frequency_accumulation)
        amplitude_val = self.amp2val(amplitude)
        offset_val = self.offset2val(offset)
        self.write_pulse(pulse, phase_val, freq_val, accum_val,
                         amplitude_val, offset_val, next_pulse)

    def write_pulse(self, pulse: int, phase: int, frequency: int,
                    frequency_accumulation: int, amplitude: int, offset: int,
                    next_pulse: int):
        """
        Function to write a pulse with given register values to a given
        location in pulse memory
        Args:
            pulse: The location in pulse memory that this is to be written to
            phase: The phase register value for this pulse
            frequency: The frequency register value for this pulse
            frequency_accumulation: The frequency accumulation register for
                                    this pulse
            amplitude: The amplitude register for this pulse
            offset: The offset register value for this pulse
            next_pulse: The pulse that the system is to go to after this one
        """
        if pulse < 0 or pulse > 2**self.n_pointer_bits:
            raise ValueError(
                f'The pulse index is outside of memory. '
                f'It should be between 0 and {2**self.n_pointer_bits-1}')
        if next_pulse < 0 or next_pulse > 2**self.n_pointer_bits:
            raise ValueError(
                f'The next_pulse index is outside of memory. It should be '
                f'between 0 and {2 ** self.n_pointer_bits - 1}')
        # Construct the initial instruction to write a new pulse to memory
        operation = int('0000100000', 2)
        instr = self.construct_instruction(operation, pulse)
        # Construct the pulse parameter to be written to memory
        pulse_data = phase
        pulse_data += (frequency << self.n_phase_bits)
        pulse_data += (frequency_accumulation << 2 * self.n_phase_bits)
        pulse_data += (amplitude << (2 * self.n_phase_bits + self.n_accum_bits))
        pulse_data += (offset << (2 * self.n_phase_bits + self.n_accum_bits
                       + self.n_amp_bits))
        pulse_data += (next_pulse << (2 * self.n_phase_bits + self.n_accum_bits
                       + 2*self.n_amp_bits))
        pulse_data = self.split_value(pulse_data)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)
        # self.fpga.set_fpga_pc_port(self.fpga_port, pulse_data, self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[5]],
                                   self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[4]],
                                   self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[3]],
                                   self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[2]],
                                   self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[1]],
                                   self.id, 0, 1)
        self.fpga.set_fpga_pc_port(self.fpga_port, [pulse_data[0]],
                                   self.id, 0, 1)

    @staticmethod
    def split_value(value: int) -> List[int]:
        """
        Splits a 24 byte message up into 6x 32 bit messages
        Args:
            value: The message that is to be split

        Returns: List of 32 bit length ints to be sent as messages
        """
        if not isinstance(value, int):
            raise TypeError('Incorrect type passed to split_value')
        return [int(value & 0xFFFFFFFF),
                int((value >> 1*32) & 0xFFFFFFFF),
                int((value >> 2*32) & 0xFFFFFFFF),
                int((value >> 3*32) & 0xFFFFFFFF),
                int((value >> 4*32) & 0xFFFFFFFF),
                int((value >> 5*32) & 0xFFFFFFFF)]

    def set_next_pulse(self, pulse: int, update: bool):
        """
        Function to set the next pulse to be played. It is also possible to
        update to this new pulse via this function
        Args:
            pulse: Next pulse to be played
            update: Should the system update right now
        """
        operation = int('0000000010', 2)
        if update:
            operation += int('1000000000', 2)
        instr = self.construct_instruction(operation, pulse)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def send_trigger(self):
        """
        Send a trigger signal to the FPGA
        """
        operation = int('1000000000', 2)
        instr = self.construct_instruction(operation, 0)
        self.fpga.set_fpga_pc_port(self.fpga_port, [instr], self.id, 0, 1)

    def phase2val(self, phase: float) -> int:
        """
        Function to calculate the correct phase register values for a given
        phase
        Args:
            phase: The desired phase in degrees.

        Returns: The register value for the desired phase
        """
        phase = phase % 360.0
        return int(np.round((2 ** self.n_phase_bits / 360.0) * phase))

    def freq2val(self, freq: float) -> int:
        """Calculates correct frequency register values for a given frequency

        The frequency register is any int between 0 and 2**self.n_phase_bits-1,
        where 0 corresponds to 0 Hz, and 2**n_phase_bits corresponds to the
        maximum frequency (inverse of the maximum sampling rate including any
        interleaving), which is effectively 0 Hz.
        Negative frequencies are effectively:
        (max_frequency + frequency) % 2**n_phase_bits

        Args:
            freq: The desired frequency in Hz

        Returns: The register value for the desired frequency
        """
        assert -self.f_max < freq < self.f_max, \
            f'Frequency of {freq} is outside of allowed values: ' \
            f'[0, {self.f_max/1e6}MHz]'
        bit_value = int(np.round((2**self.n_phase_bits / (5*self.clk)) * freq))
        # If frequency is negative, bit value is also negative.
        # Ensure positive value by taking modulus
        bit_value %= 2**self.n_phase_bits
        return bit_value

    def accum2val(self, accum: float) -> int:
        """
        Function to calculate the correct accumulation register values for a
        given accumulation
        Args:
            accum: The desired accumulation in Hz/s

        Returns: The register value for the desired accumulation
        """
        if accum < 0 or accum > (5*self.clk ** 2):
            raise ValueError(f'Frequency Accumulation of {accum} is outside of '
                             f'allowed values [0,{(5*self.clk ** 2)}Hz/s]')
        return int(np.round(accum * 2 ** self.n_accum_bits / (5*self.clk) ** 2))

    def amp2val(self, amp: float) -> int:
        """
        Function to calculate the correct amplitude register values for a given
        amplitude
        Args:
            amp: The desired amplitude in V

        Returns: The register value for the desired amplitude
        """
        if amp < 0 - 1e-10 or amp > self.v_max + 1e-10:
            raise ValueError(f'Amplitude of {amp} is outside of allowed values '
                             f'[0, {self.v_max}V')
        value = int(np.round((2**self.n_amp_bits-1) * amp/self.v_max))
        # Limit offset
        value = min(value, 2 ** self.n_amp_bits - 1)
        value = max(value, 0)
        return value

    # Limit offset
    def offset2val(self, offset: float) -> int:
        """
        Function to calculate the correct offset register value for a given
        offset
        Args:
            offset: The desired offset in V

        Returns: The register value for the desired offset

        """
        assert -self.v_max -1e-10 <= offset <= self.v_max + 1e-10, \
            f'Offset value is not within allowed bounds of [-{self.v_max}, ' \
            f'{self.v_max}]'

        value = int(np.round(offset/self.v_max * 2**(self.n_amp_bits-1)))
        value = min(value, 2 ** (self.n_amp_bits - 1) - 1)
        value = max(value, -2 ** (self.n_amp_bits - 1))
        # Tricking python into not thinking it's a negative number
        value += 1 << 17
        value &= 0xFFFF
        return value


class PCDDS(Instrument):
    """
    This class is the driver for the Phase Coherent Pulse Generation Module
    implemented on the FPGA onboard a Keysight PXI AWG card
    """
    def __init__(self, name: str, FPGA, channels: int = None, **kwargs):
        """ Constructor for the pulse generation modules """
        super().__init__(name,  **kwargs)
        self.fpga = FPGA

        if channels is None:
            channels = model_channels[self.fpga.model]
        self.n_channels = channels

        channels = ChannelList(self,
                               name='channels',
                               chan_type=PCDDSChannel)
        for ch in range(self.n_channels):
            channel = PCDDSChannel(self, name=f'ch{ch}', id=ch)
            setattr(self, f'ch{ch}', channel)
            channels.append(channel)
        self.add_submodule('channels', channels)

    def reset(self):
        """ Sends the reset signal to the FPGA.

        Hasn't been tested if it actually does something.
        """
        self.fpga.reset(reset_mode=keysightSD1.SD_ResetMode.PULSE)

    def get_debug_messages(self):
        address_labels = ['front panel input',
                          'pxi trigger 0',
                          'pxi trigger 1',
                          'pxi trigger 2',
                          'pxi trigger 3',
                          'output enable ch0',
                          'delay length ch0',
                          'output enable ch1',
                          'delay length ch1',
                          'output enable ch2',
                          'delay length ch2',
                          'output enable ch3',
                          'delay length ch3']
        return {label: PCDDS.fpga.get_fpga_pc_port(port=1,
                                                   data_size=1,
                                                   address=debug_port + 1,
                                                   address_mode=1,
                                                   access_mode=0)[0]
                for debug_port, label in enumerate(address_labels)}